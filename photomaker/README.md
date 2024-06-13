# PhotoMaker Truss

## Introduction and Credits

`PhotoMaker`, developed by TencentARC, represents a significant advancement in the domain of AI-based photo customization. It utilizes the `PhotoMakerStableDiffusionXLPipeline` for generating realistic human photos with high fidelity, offering unparalleled diversity and textual controllability. This marks a notable collaboration between TencentARC and open-source contributors worldwide. We extend our gratitude to TencentARC for spearheading this innovation. For deeper insights, visit the [official GitHub repository](https://github.com/TencentARC/PhotoMaker).

## Deploying PhotoMaker on Baseten

Deploying the PhotoMaker model on Baseten is streamlined for ease. Begin by cloning the desired repository with PhotoMaker's deployment configuration:

```bash
git clone https://github.com/basetenlabs/truss-examples/
cd truss-examples/photo_truss
truss push
```

During deployment, you will be prompted to provide your Baseten API key. This step is crucial for successful model integration.

## Input

PhotoMaker expects JSON-format input. The primary fields include `img_base64` for a base64 encoded image string or `prompt` for entering textual descriptions. Optional parameters like `size` enable output resolution adjustments, critical for tailoring the result to specific requirements.

## Output

The model generates a JSON containing the key `image`. This key maps to the base64 encoded string of the output image, facilitating direct web embedding and display.

## Example usage

Utilize the deployed PhotoMaker model through Baseten with the following pattern:

```bash
truss predict --input '{"prompt": "Generate a portrait with a vibrant sunset backdrop", "size": 1024}'
```

Executing the above command prompts PhotoMaker to craft a custom image based on the provided description and parameters, showcasing its text controllability and customization prowess.