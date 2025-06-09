# Chatterbox TTS Example
## Overview
This is a basic example showing how to deploy [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) on truss. 

## Running the Example
The truss endpoint is set up to accept text and an optional base64 wav string, and it returns that base64 string audio.

For a more detailed example on how to run it, take a look at [`run_tts.py`](https://github.com/basetenlabs/truss-examples/blob/main/chatterbox-tts/run_tts.py). This script will take the `input_text`, apply the voice clone file `input/obama_8s.wav`, and output the audio file to `output/output_obama8s.wav`.

## Custom Docker Image
Currently, this example is built from an slightly modified baseten docker image. That image installs `numpy==1.26.0` on a truss base image in order to work with `chatterbox-tts==0.1.1`.

For more details, see [Creating a custom base image](https://docs.baseten.co/development/model/base-images#creating-a-custom-base-image).