# Image Models

Truss configurations for image generation, editing, and segmentation models. Covers Stable Diffusion, Flux, and a variety of specialized image processing pipelines.

| Directory                                           | Variants | Description                                                                                                                   |
| --------------------------------------------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------- |
| [stable-diffusion](stable-diffusion/)               | 18       | Stable Diffusion family including SD 1.x, SDXL, SD 3, turbo, LCM, LoRA, ControlNet, inpainting, TensorRT, and video diffusion |
| [flux](flux/)                                       | 2        | Black Forest Labs Flux models (dev, schnell)                                                                                  |
| [flux-dev-trt-b200](flux-dev-trt-b200/)             | 1        | Flux Dev optimized with TensorRT for B200 GPUs                                                                                |
| [sana](sana/)                                       | 2        | Sana image generation models (600M, 1600M)                                                                                    |
| [comfyui](comfyui/)                                 | 1        | ComfyUI workflow server for node-based image generation                                                                       |
| [control-net-qrcode](control-net-qrcode/)           | 1        | ControlNet QR code art generation                                                                                             |
| [deepfloyd-xl](deepfloyd-xl/)                       | 1        | DeepFloyd IF XL text-to-image model                                                                                           |
| [fotographer](fotographer/)                         | 1        | Fotographer AI portrait generation model                                                                                      |
| [gfp-gan](gfp-gan/)                                 | 1        | GFPGAN face restoration and enhancement                                                                                       |
| [ip-adapter](ip-adapter/)                           | 1        | IP-Adapter for image-prompted generation                                                                                      |
| [magic-animate](magic-animate/)                     | 1        | MagicAnimate human image animation                                                                                            |
| [playground-v2-aesthetic](playground-v2-aesthetic/) | 1        | Playground v2 aesthetic image generation                                                                                      |
| [segment-anything](segment-anything/)               | 1        | Meta SAM universal image segmentation                                                                                         |
| [dis-segmentation](dis-segmentation/)               | 1        | Dichotomous image segmentation for high-accuracy cutouts                                                                      |
| [image-segmentation](image-segmentation/)           | 1        | General-purpose image segmentation                                                                                            |

## Deploying

Each image model can be deployed to Baseten with:

```bash
truss push
```
