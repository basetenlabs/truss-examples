# Mistral 7B Instruct LoRA

This is an example of a truss that supports **dynamic swapping of LoRA adapters**—allowing you to serve multiple fine-tuned variants efficiently from a single GPU. In this example, we deploy a **Mistral 7B Instruct** on vLLM's server. This model will be an expert in finance and law.

- 💡 **LoRA Swapping Overview:** [Baseten Blog: Serving 10,000 Fine-Tuned LLMs from One GPU](https://www.baseten.co/blog/how-to-serve-10-000-fine-tuned-llms-from-a-single-gpu/)

---

## 🛠️ Implementing LoRA Swapping

Extending a base vLLM deployment to support LoRA swapping requires three config changes:

### 1. Configure `lora_modules`

List each LoRA adapter’s name and its huggingface repo.

**Example (`config.yaml`):**
```
--enable-lora --lora-modules finance=vaibhav1/lora-mistral-finance legal=Aretoss/Lexgen
```

### 2. Set `served_model_name` (Optional)

Set this parameter if you wish to allow requests to the base model (without any LoRA applied).

---

### 3. Select Adapter or Base Model at Request Time

Specify the desired adapter or base model using the `model` field in your request payload.

**Example request body**

```json
{
  "model": "finance", // Or "legal" or "mistral"
  "stream": true,
  "messages": [
    {"role": "user", "content": "What would you choose in 2008?"}
  ],
  "max_tokens": 1024,
  "temperature": 0.9
}
```

### For full details, see [documentation](https://docs.vllm.ai/en/v0.9.1/features/lora.html#lora-model-lineage-in-model-card)
