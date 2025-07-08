#!/bin/bash
set -e

# Map architectures to prefixes
declare -A ARCHES=(
  ["cpu"]="cpu-"
  ["turing"]="turing-"
  ["ampere80"]=""
  ["ampere86"]="86-"
  ["adalovelace"]="89-"
  ["hopper"]="hopper-"
)

# Define version and target
VERSION="1.7.2"
TARGET="baseten/text-embeddings-inference-mirror"

# Build and push images
for ARCH in "${!ARCHES[@]}"; do
  ARCH_PREFIX=${ARCHES[$ARCH]}
  TAG="${TARGET}:${ARCH_PREFIX}${VERSION}"

  echo "Building and pushing image for $ARCH: $TAG"

  docker buildx build -t "$TAG" --build-arg TAG="${ARCH_PREFIX}${VERSION}" --push .
done

echo "All images have been built and pushed."
