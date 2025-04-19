# orpheus-tts

This is an implementation of Orpheus TTS that supports streaming and concurrent requests.

Source Code:
- https://huggingface.co/canopylabs/orpheus-3b-0.1-ft
- https://github.com/canopyai/Orpheus-TTS/tree/main
- https://github.com/canopyai/Orpheus-Speech-PyPi/blob/main/orpheus_tts/engine_class.py
- https://huggingface.co/spaces/MohamedRashad/Orpheus-TTS

# Voices

`["zoe", "zac", "jess", "leo", "mia", "julia", "leah"]`

# Performance
- Use H100_40GB, A100 or H100 GPUs as they give the optimal token/second performance
- `dtype` overwritten from `dtype=torch.dfloat16` to `dtype=torch.float16`
- According to [creators of Orpheus](https://github.com/canopyai/Orpheus-TTS/issues/53#issuecomment-2749433171), `The required generation speed for streaming is 83 toks/s as that is the number of tokens needed for 1s of audio. It seems like the A100 is generating faster than the necessary speed (~110 tok/s) as recorded in the logs (and the audio is generated in less time than its duration).`
