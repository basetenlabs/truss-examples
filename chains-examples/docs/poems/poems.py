import asyncio
from typing import List

import pydantic
import truss_chains as chains
from truss import truss_config

PHI_HF_MODEL = "microsoft/Phi-3-mini-4k-instruct"
# This configures to cache model weights from the hunggingface repo
# in the docker image that is used for deploying the Chainlet.
PHI_CACHE = truss_config.ModelRepo(
    repo_id=PHI_HF_MODEL, allow_patterns=["*.json", "*.safetensors", ".model"]
)


class Messages(pydantic.BaseModel):
    messages: List[dict[str, str]]


class PhiLLM(chains.ChainletBase):
    # `remote_config` defines the resources required for this chainlet.
    remote_config = chains.RemoteConfig(
        docker_image=chains.DockerImage(
            # The phi model needs some extra python packages.
            pip_requirements=[
                "accelerate==0.30.1",
                "einops==0.8.0",
                "transformers==4.41.2",
                "torch==2.3.0",
            ]
        ),
        # The phi model needs a GPU and more CPUs.
        compute=chains.Compute(cpu_count=2, gpu="T4"),
        # Cache the model weights in the image
        assets=chains.Assets(cached=[PHI_CACHE]),
    )

    def __init__(self) -> None:
        # Note the imports of the *specific* python requirements are
        # pushed down to here. This code will only be executed on the
        # remotely deployed Chainlet, not in the local environment,
        # so we don't need to install these packages in the local
        # dev environment.
        import torch
        import transformers

        self._model = transformers.AutoModelForCausalLM.from_pretrained(
            PHI_HF_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            PHI_HF_MODEL,
        )
        self._generate_args = {
            "max_new_tokens": 512,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 50,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "use_cache": True,
            "do_sample": True,
            "eos_token_id": self._tokenizer.eos_token_id,
            "pad_token_id": self._tokenizer.pad_token_id,
        }

    async def run_remote(self, messages: Messages) -> str:
        import torch

        model_inputs = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(model_inputs, return_tensors="pt")
        input_ids = inputs["input_ids"].to("cuda")
        with torch.no_grad():
            outputs = self._model.generate(input_ids=input_ids, **self._generate_args)
            output_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output_text


@chains.mark_entrypoint
class PoemGenerator(chains.ChainletBase):
    def __init__(self, phi_llm: PhiLLM = chains.depends(PhiLLM)) -> None:
        self._phi_llm = phi_llm

    async def run_remote(self, words: list[str]) -> list[str]:
        tasks = []
        for word in words:
            messages = Messages(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are poet who writes short, "
                            "lighthearted, amusing poetry."
                        ),
                    },
                    {"role": "user", "content": f"Write a poem about {word}"},
                ]
            )
            tasks.append(asyncio.ensure_future(self._phi_llm.run_remote(messages)))
        return await asyncio.gather(*tasks)
