# PhotoMaker Truss

## Introduction
The PhotoMaker model is a cutting-edge solution designed for customizing realistic human photos with high fidelity and text controllability. This remarkable model is the result of a collaborative effort led by Tencent ARC Lab and Nankai University's [MCG-NKU](https://mmcheng.net/cmm/) team. The work presents an innovative approach to rapidly customizing images while maintaining impressive identity fidelity and offering diversity. PhotoMaker serves as an adaptable tool that can be utilized alongside other Base Models with LoRA modules to enhance image modification capabilities. For detailed information, visit the [PhotoMaker GitHub Repository](https://github.com/TencentARC/PhotoMaker).

## Deploying PhotoMaker on Baseten
To utilize the capabilities of PhotoMaker within the Baseten platform, follow these deployment steps:
1. Obtain the truss-examples repository by executing: `git clone https://github.com/basetenlabs/truss-examples/`
2. Navigate to the directory for PhotoMaker Truss with: `cd truss-examples/photomaker_truss`
3. Deploy the model on Baseten using: `truss push`

This will host the PhotoMaker model on Baseten, making it accessible for online prediction.

## Input
The model expects input in JSON format carrying a base64 encoded string of the image that should be processed. The format of your input should look like this: `{"image": "<base64_encoded_image_string>"}`. This setup allows you to provide a diverse range of inputs for image customization and enhancement.

## Output
Following successful processing, PhotoMaker returns a prediction in the form of a base64 encoded image string encapsulated in a JSON object: `{"prediction": "<base64_encoded_image_string>"}`. This output can then be decoded to retrieve the enhanced image.

## Example Usage
Invoke the PhotoMaker model on Baseten as follows:
```
truss predict --input '{"image": "base64_encoded_image_string"}'
```
Expected Output:
```
{"prediction": "base64_encoded_modified_image_string"}
```
This command will send the input image to the PhotoMaker model hosted on Baseten and return a modified image in response, showcasing the model's capability to enhance and customize photos with remarkable fidelity and control.
