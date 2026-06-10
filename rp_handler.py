import os
import logging
import base64
import io
import tempfile
import requests

import runpod

import numpy as np
import rembg
import torch
from PIL import Image

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

# ---------------------------------------------------------------------------
# Global model cache – loaded once per container cold-start
# ---------------------------------------------------------------------------
MODEL = None
REMBG_SESSION = None
DEVICE = None


def get_device():
    global DEVICE
    if DEVICE is None:
        DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    return DEVICE


def load_model(pretrained_model_name_or_path: str = "stabilityai/TripoSR"):
    global MODEL, REMBG_SESSION
    if MODEL is not None:
        return MODEL

    device = get_device()
    logging.info(f"Loading TripoSR model on {device} ...")
    MODEL = TSR.from_pretrained(
        pretrained_model_name_or_path,
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    MODEL.renderer.set_chunk_size(8192)
    MODEL.to(device)
    logging.info("TripoSR model loaded.")

    REMBG_SESSION = rembg.new_session()
    return MODEL


def download_image(image_url: str) -> Image.Image:
    """Download an image from a URL and return a PIL RGB image."""
    response = requests.get(image_url, timeout=60)
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    return image


def preprocess_image(image: Image.Image, foreground_ratio: float = 0.85) -> Image.Image:
    """Remove background and resize foreground."""
    image = remove_background(image, REMBG_SESSION)
    image = resize_foreground(image, foreground_ratio)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
    image = Image.fromarray((image * 255.0).astype(np.uint8))
    return image


def generate_mesh(
    image: Image.Image,
    output_path: str,
    mc_resolution: int = 256,
    model_save_format: str = "glb",
    bake_texture: bool = False,
    texture_resolution: int = 2048,
) -> str:
    """Run TripoSR inference and export mesh."""
    device = get_device()
    model = load_model()

    with torch.no_grad():
        scene_codes = model([image], device=device)

    meshes = model.extract_mesh(
        scene_codes, not bake_texture, resolution=mc_resolution
    )
    mesh = meshes[0]

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if bake_texture:
        from tsr.bake_texture import bake_texture as bake_texture_fn
        import xatlas

        bake_output = bake_texture_fn(mesh, model, scene_codes[0], texture_resolution)
        xatlas.export(
            output_path,
            mesh.vertices[bake_output["vmapping"]],
            bake_output["indices"],
            bake_output["uvs"],
            mesh.vertex_normals[bake_output["vmapping"]],
        )
        texture_path = os.path.splitext(output_path)[0] + "_texture.png"
        Image.fromarray((bake_output["colors"] * 255.0).astype(np.uint8)).transpose(
            Image.FLIP_TOP_BOTTOM
        ).save(texture_path)
    else:
        mesh.export(output_path)

    return output_path


def handler(job):
    """RunPod serverless handler."""
    job_input = job.get("input", {})

    # -----------------------------------------------------------------------
    # 1. Parse inputs
    # -----------------------------------------------------------------------
    image_url = job_input.get("image_url")
    image_base64 = job_input.get("image_base64")
    no_remove_bg = job_input.get("no_remove_bg", False)
    foreground_ratio = job_input.get("foreground_ratio", 0.85)
    mc_resolution = job_input.get("mc_resolution", 256)
    model_save_format = job_input.get("model_save_format", "glb")
    bake_texture = job_input.get("bake_texture", False)
    texture_resolution = job_input.get("texture_resolution", 2048)
    return_mode = job_input.get("return_mode", "base64")
    # return_mode: "base64" (embed file) | "path" (local path, useful with network volumes)

    if not image_url and not image_base64:
        return {"status": "error", "message": "Either 'image_url' or 'image_base64' must be provided."}

    try:
        # -------------------------------------------------------------------
        # 2. Load image
        # -------------------------------------------------------------------
        if image_url:
            logging.info(f"Downloading image from {image_url} ...")
            image = download_image(image_url)
        else:
            logging.info("Decoding base64 image ...")
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # -------------------------------------------------------------------
        # 3. Preprocess
        # -------------------------------------------------------------------
        if not no_remove_bg:
            logging.info("Removing background ...")
            image = preprocess_image(image, foreground_ratio)
        else:
            logging.info("Skipping background removal (no_remove_bg=True).")

        # -------------------------------------------------------------------
        # 4. Run TripoSR
        # -------------------------------------------------------------------
        output_filename = f"output.{model_save_format}"
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, output_filename)
            generate_mesh(
                image,
                output_path,
                mc_resolution=mc_resolution,
                model_save_format=model_save_format,
                bake_texture=bake_texture,
                texture_resolution=texture_resolution,
            )

            # ---------------------------------------------------------------
            # 5. Return result
            # ---------------------------------------------------------------
            if return_mode == "base64":
                with open(output_path, "rb") as f:
                    file_bytes = f.read()
                file_b64 = base64.b64encode(file_bytes).decode("utf-8")
                return {
                    "status": "success",
                    "model_base64": file_b64,
                    "format": model_save_format,
                }
            else:
                # If using a RunPod network volume, you could copy to a persistent path here
                return {
                    "status": "success",
                    "model_path": output_path,
                    "format": model_save_format,
                }

    except Exception as e:
        logging.exception("Job failed")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
