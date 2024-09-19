# whisper-trt

This package allow running TRTLLM Engines for Whisper in a performant way
that supports the same interface as other whisper packages, such as
openai/whisper and faster-whisper.

## build engine
When using the package, the engine will automatically be built for you. But if you need something specific...

Edit `fde/whisper/whisper-trt/src/whisper_trt/builder/__main__.py` to reference the file you want, and run
```
rye run python -m whisper_trt.builder
```