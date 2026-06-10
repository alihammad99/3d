# TripoSR RunPod Serverless

[![RunPod](https://api.runpod.io/badge/alihammad99/3d)](https://console.runpod.io/hub/alihammad99/3d)

RunPod serverless worker that converts a 2D image into a 3D mesh using [TripoSR](https://github.com/VAST-AI-Research/TripoSR).

## Files

- `rp_handler.py` – RunPod handler (image download → preprocess → TripoSR → mesh export)
- `Dockerfile` – Docker image with CUDA, TripoSR, and all dependencies
- `requirements.txt` – Python dependencies for local development
- `build.sh` – Quick Docker build script

## API Input

```json
{
  "input": {
    "image_url": "https://example.com/photo.jpg",
    "no_remove_bg": false,
    "foreground_ratio": 0.85,
    "mc_resolution": 256,
    "model_save_format": "glb",
    "bake_texture": false,
    "texture_resolution": 2048,
    "return_mode": "base64"
  }
}
```

| Field                | Type   | Default    | Description                                      |
| -------------------- | ------ | ---------- | ------------------------------------------------ |
| `image_url`          | string | —          | URL of input image (alternative: `image_base64`) |
| `image_base64`       | string | —          | Base64-encoded image (alternative: `image_url`)  |
| `no_remove_bg`       | bool   | `false`    | Skip automatic background removal                |
| `foreground_ratio`   | float  | `0.85`     | Foreground size ratio after background removal   |
| `mc_resolution`      | int    | `256`      | Marching cubes grid resolution                   |
| `model_save_format`  | string | `"glb"`    | Output format: `obj` or `glb`                    |
| `bake_texture`       | bool   | `false`    | Bake texture atlas instead of vertex colors      |
| `texture_resolution` | int    | `2048`     | Texture atlas resolution (if baking)             |
| `return_mode`        | string | `"base64"` | `"base64"` (embed file) or `"path"` (local path) |

## API Output (base64 mode)

```json
{
  "status": "success",
  "model_base64": "<base64-string>",
  "format": "glb"
}
```

## Build & Deploy

### Option 1: RunPod Hub (Recommended)

1. Push this repo to GitHub.
2. In RunPod, choose **Deploy from GitHub Repo** and enter your repo URL.
3. RunPod will auto-build from the `Dockerfile`.

### Option 2: Manual Docker Build

1. Build locally:

   ```bash
   ./build.sh
   ```

2. Push to a container registry (Docker Hub, GitHub Container Registry, RunPod Registry, etc.).

3. Create a RunPod **Serverless Endpoint** and point it to your pushed image.

## Important Notes

- **Model caching:** The Dockerfile pre-downloads TripoSR weights during the build so the container cold-start is fast.
- **GPU required:** TripoSR needs a CUDA GPU. Use a RunPod GPU worker.
- **Storage:** If you prefer returning a URL instead of base64, modify `rp_handler.py` to upload the output mesh to S3 / Cloudflare R2 / MinIO inside the handler and return the public URL.
