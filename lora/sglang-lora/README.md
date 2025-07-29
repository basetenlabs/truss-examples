# IMPORTANT NOTICE

Currently, [support for openai compatible server with lora is not yet implemented](https://github.com/sgl-project/sglang/issues/2929). It is strongly recommended to go with the [vLLM example](../vllm-lora).

# Mistral 7B Instruct LoRA

This is an example of a truss that supports **dynamic swapping of LoRA adapters**—allowing you to serve multiple fine-tuned variants efficiently from a single GPU. In this example, we deploy a **Mistral 7B Instruct** with SGLang. This model will be an expert in finance, medicine and law.

- 💡 **LoRA Swapping Overview:** [Baseten Blog: Serving 10,000 Fine-Tuned LLMs from One GPU](https://www.baseten.co/blog/how-to-serve-10-000-fine-tuned-llms-from-a-single-gpu/)

---

## 🛠️ Implementing LoRA Swapping

Extending a base SGLang custom server deployment to support LoRA swapping requires two config changes:

### 1. Configure `lora_paths`

List each LoRA adapter’s name and its huggingface repo.

**Example (`start_command` in `config.yaml`):**
```
--enable-lora --lora-paths legal=Aretoss/Lexgen finance=vaibhav1/lora-mistral-finance medical=Imsachinsingh00/Fine_tuned_LoRA_Mistral_MTSDialog_Summarization --disable-radix-cache
```

Note that the `--disable-radix-cache` flag is necessary because lora with radix attention is not yet implemented in SGLang.

---

### 3. Select Adapter or Base Model at Request Time

This API is not openai compatible. Follow the example model input in the `README.md`.

**Example request body**

```json
{
"text": [
    "What would you choose in 2008?",
    "What would you choose in 2008?",
],
"sampling_params": {"max_new_tokens": 1000, "temperature": 1.0},
"lora_path": ["legal", "finance"],
}
```

### For full details, see SGLang's [documentation](https://docs.sglang.ai/backend/lora.html)
