# PhotoMaker Truss

## Introduction
PhotoMaker Truss is an exceptional model developed through the collaboration of Tencent ARC Lab and the MCG-NKU team at Nankai University. It enables high-fidelity customization of human photos, leveraging advanced text controllability for varied and rapid imagery modifications. Honouring its innovative predecessors, PhotoMaker further amplifies the capabilities of base models through the integration of LoRA modules. Discover more about this transformative technology at the [PhotoMaker GitHub Repository](https://github.com/TencentARC/PhotoMaker).

## Deploying PhotoMaker on Baseten
Harness the power of the PhotoMaker model with these steps on the Baseten platform:
1. Begin by cloning the `truss-examples` repository:
   ```
   git clone https://github.com/basetenlabs/truss-examples/
   ```
2. Transition into the PhotoMaker Truss directory:
   ```
   cd truss-examples/photomaker_truss
   ```
3. Commence the model's deployment on Baseten with:
   ```
   truss push
   ```
This procedure stations the PhotoMaker model on Baseten, facilitating its immediate availability for online predictive processing and seamless image enhancement.

## Input
PhotoMaker necessitates a JSON-format input comprising a base64 encoded string of the to-be-processed image. Ensure your input mirrors the subsequent framework:
```
{"image": "<base64_encoded_image_string>"}
```
This architecture guarantees the comprehensive processability and customization feasibility of a broad image spectrum by the model.

## Output
Subsequent to processing, PhotoMaker articulates its predictions using a JSON object encasing a base64 encoded image string. This output is thereby structured as follows:
```
{"prediction": "<base64_encoded_image_string>"}
```
Decoding this output divulges the enhanced and meticulously customized image, demonstrating the model's exquisite capability to refine photos with unparalleled fidelity and control.

## Example Usage
Engage the PhotoMaker model on Baseten via the ensuing command:
```
truss predict --input '{"image": "base64_encoded_image_string"}'
```
Anticipated output is:
```
{"prediction": "base64_encoded_modified_image_string"}
```
This example succinctly illustrates submitting an image to the PhotoMaker model on Baseten and retrieving a modified version in return, underpinning the model's distinguishable proficiency in photo customization and enhancement.
