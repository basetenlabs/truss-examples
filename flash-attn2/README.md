### Building the Base image

[`base_image.Dockerfile`](./base_image.Dockerfile) describes the base image that's running at the truss.

To make modification or push a new version, you can run
```
docker buildx build -f base_image.Dockerfile -t YOUR_REPO/YOUR_IMAGE:YOUR_TAG --push
```
Make sure to update the `config.yaml` to point at this newly uploaded image.

TODO: we could make this docker image lighter by adopting a multistage build and moving the flash-attn package after it has been built on a lighter nvidia -runtime- image instead -devel- image, but that can be optimized later.

## The truss
The rest of the truss is as usual.