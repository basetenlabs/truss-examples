# PersonaPlex 7V - Real-time Speech-to-Speech

This example demonstrates deploying NVIDIA's PersonaPlex 7B v1 model using Baseten's Bring Your Own Image (BYOI) feature with a custom Docker image containing a forked version of PersonaPlex.

## Overview

PersonaPlex is a speech-to-speech AI model that enables real-time voice conversations. This deployment:

1. **Custom Docker Image**: We built a Docker image (`basetenservice/personaplex-7v:fork`) that builds a fork of PersonaPlex from [basetenlabs/personaplex-baseten](https://github.com/basetenlabs/personaplex-baseten)
2. **Baseten BYOI**: The Docker image is deployed using Baseten's bring your own image capability, which allows us to use custom base images with the Baseten platform
3. **WebSocket Protocol**: The model communicates via WebSocket for low-latency bidirectional audio streaming

The dockerfile is included in this folder. The `config.yaml` brings this image and runs the Moshi server on port 8998.

## Server Setup

To deploy PersonaPlex 7V on Baseten, you'll use the Truss CLI to push this folder

### Requirements

- Truss CLI installed (`pip install truss`), cd into this directory, and `truss push --publish`
- Baseten account and API key
- Hugging Face account and Read Access Token
```

### Deploy the Model

From the `personaplex-7b-v1` directory, push your Truss:

```bash
truss push . --byoi --env HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN
```

This command builds your custom Docker image, passes the Hugging Face token as a runtime environment variable, and deploys your model to Baseten.

After deployment, your WebSocket server will be available and ready to accept client connections.

## Client Setup

### Prerequisites

The client to connect to the server requires the Opus audio codec and Python dependencies.

#### macOS

```bash
brew install opus
```

#### Linux (Ubuntu/Debian)

```bash
sudo apt-get install libopus-dev
```

### Install Python Dependencies

```bash
pip install -r requirements-client.txt
```

This installs:
- `websockets` - WebSocket client library
- `numpy` - Array operations
- `sphn` - Opus audio codec Python bindings
- `sounddevice` - Audio I/O

### Set API Key

```bash
export BASETEN_API_KEY="your_api_key_here"
```

## Running the Client

```bash
python client.py
```

The client will:
1. Connect to the deployed model via WebSocket
2. Start your microphone and speakers
3. Enable real-time voice conversation with the AI

## WebSocket Protocol

The communication protocol uses a simple binary format:

### Initial Handshake

1. **Client sends config** (first message): JSON configuration
   ```json
   {
     "voice_prompt": "NATF0.pt",
     "text_prompt": "You are a helpful assistant.",
     "seed": -1
   }
   ```

2. **Server responds**: Single byte `0x00` to confirm readiness

### Streaming Messages

After the handshake, all messages are binary with a single-byte header:

- **`0x01` + Opus audio data**: Audio frames (sent by both client and server)
- **`0x02` + UTF-8 text**: Text transcriptions/responses (sent by server)

The first byte of each message indicates the payload type, and the remaining bytes contain the Opus-encoded audio or UTF-8 text.

## Configuration

The model is configured in `config.yaml`:
- **Image**: Contains the built model at nvidia/personaplex-7b-v1
- **Accelerator**: H100 GPU
- **Transport**: WebSocket
- **Secrets**: Requires HuggingFace access token for model download
