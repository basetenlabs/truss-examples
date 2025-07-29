# IMPORTANT NOTICE

Currently we're seeing an issue where using LoRAs with TensorRT-LLM shows a significant drop in performance. It is strongly recommended to go with the [vLLM example](../vllm-lora)

# Mistral 7B Instruct LoRA

This is an example of a truss that supports **dynamic swapping of LoRA adapters**—allowing you to serve multiple fine-tuned variants efficiently from a single GPU. In this example, we deploy a **Mistral 7B Instruct** on [TensorRT-LLM Engine Builder](https://docs.baseten.co/performance/examples/mistral-trt). This model will be an expert in finance, medicine and law.

- 📄 **TensorRT-LLM Details:** [Performance Example (Baseten Docs)](https://docs.baseten.co/performance/examples/mistral-trt)
- 💡 **LoRA Swapping Overview:** [Baseten Blog: Serving 10,000 Fine-Tuned LLMs from One GPU](https://www.baseten.co/blog/how-to-serve-10-000-fine-tuned-llms-from-a-single-gpu/)

---

## 🛠️ Implementing LoRA Swapping

Extending a base TensorRT-LLM deployment to support LoRA swapping requires three config changes:

### 1. Configure `lora_adapters`

List each LoRA adapter’s name and its download source. Supported sources are:
- `HF` for HuggingFace
- `GCS` for Google Cloud Storage
- `REMOTE_URL` for any direct download

**Example (`config.yaml`):**
```yaml
lora_adapters:
  legal:
    source: HF
    repo: Aretoss/Lexgen
  finance:
    source: HF
    repo: vaibhav1/lora-mistral-finance
  medical:
    source: HF
    repo: Imsachinsingh00/Fine_tuned_LoRA_Mistral_MTSDialog_Summarization
```

### 2. Set `served_model_name` (Optional)

Set this parameter if you wish to allow requests to the base model (without any LoRA applied).

---

### 3. Select Adapter or Base Model at Request Time

Specify the desired adapter or base model using the `model` field in your request payload.

**Example request body**

```json
{
  "model": "finance", // # Or legal, medical, or mistral
  "stream": true,
  "messages": [
    {"role": "user", "content": "What would you choose in 2008?"}
  ],
  "max_tokens": 1024,
  "temperature": 0.9
}
```

### For full details, see [documentation](https://docs.baseten.co/development/model/performance/engine-builder-config)
