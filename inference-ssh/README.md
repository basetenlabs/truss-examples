# Inference SSH examples

These examples use Baseten's [SSH access](https://docs.baseten.co/inference/ssh) to
connect to a running model deployment and work inside the container.

- [tune-vllm-args](tune-vllm-args) — SSH into a running vLLM server, edit its launch
  args and chat template, and restart the engine in place.
