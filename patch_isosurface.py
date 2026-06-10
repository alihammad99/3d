"""
Patches TripoSR's tsr/models/isosurface.py to use the pure-Python 'mcubes'
package instead of the CUDA-dependent 'torchmcubes'.
Run this script during Docker build AFTER cloning TripoSR.
"""
import os
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/TripoSR/tsr/models/isosurface.py"

with open(path, "r") as f:
    src = f.read()

# Replace the torchmcubes import and marching_cubes call
src = src.replace(
    "from torchmcubes import marching_cubes",
    "import mcubes as _mcubes\n\n\ndef marching_cubes(level, threshold):\n    verts, faces = _mcubes.marching_cubes(-level.cpu().numpy(), threshold)\n    import torch\n    return torch.from_numpy(verts).float(), torch.from_numpy(faces.astype('int64')).long()"
)

with open(path, "w") as f:
    f.write(src)

print(f"Patched {path} successfully.")
