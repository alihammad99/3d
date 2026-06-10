#!/usr/bin/env bash
set -e

IMAGE_NAME="triposr-runpod"
TAG="latest"

echo "Building Docker image: ${IMAGE_NAME}:${TAG} ..."
docker build -t ${IMAGE_NAME}:${TAG} .

echo ""
echo "Build complete. To push to a registry, run:"
echo "  docker tag ${IMAGE_NAME}:${TAG} your-registry.com/${IMAGE_NAME}:${TAG}"
echo "  docker push your-registry.com/${IMAGE_NAME}:${TAG}"
echo ""
echo "To test locally:"
echo "  docker run --rm -it --gpus all ${IMAGE_NAME}:${TAG}"
