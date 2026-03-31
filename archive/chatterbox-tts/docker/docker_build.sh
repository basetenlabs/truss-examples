#!/bin/bash

DOCKER_USERNAME="YOUR_DOCKER_USERNAME"
IMAGE_NAME="truss-numpy-1.26.0-gpu"
VERSION="0.1"

docker buildx build --platform linux/amd64 -t $IMAGE_NAME:$VERSION --load .
docker tag $IMAGE_NAME:$VERSION $DOCKER_USERNAME/$IMAGE_NAME:$VERSION
docker push $DOCKER_USERNAME/$IMAGE_NAME:$VERSION
