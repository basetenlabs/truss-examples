import asyncio
import copy
import inspect
import os
from pathlib import Path
from typing import AsyncGenerator, Dict

import numpy as np
from constants import (
    HF_AUTH_KEY_CONSTANT,
    LORA_CONFIG_PATH,
    LORA_CONVERSION_DTYPE,
    LORA_DIR,
    LORA_WEIGHTS_PATH,
    TLLM_LORA_DIR,
)
from hf_lora_convert import convert_hf_model
from huggingface_hub import snapshot_download
from schema import LoraAdapter, ModelInput
from triton_client import TritonClient


class LoraRegistryManager:
    def __init__(self, triton_client: TritonClient):
        self.triton_client = triton_client
        self._lora_registry: Dict[str, int] = {}
        self._lora_task_index: int = 0
        LORA_DIR.mkdir(parents=True, exist_ok=True)
        TLLM_LORA_DIR.mkdir(parents=True, exist_ok=True)

    def _get_hf_lora_path(self, lora_hf_repo: str) -> Path:
        return LORA_DIR / lora_hf_repo

    def _get_tllm_lora_path(self, lora_hf_repo: str) -> Path:
        return TLLM_LORA_DIR / lora_hf_repo

    def _download_lora(self, lora_hf_repo, auth_token=None) -> str:
        hf_lora_path = self._get_hf_lora_path(lora_hf_repo)
        snapshot_download(
            lora_hf_repo,
            local_dir=hf_lora_path,
            local_dir_use_symlinks=False,
            max_workers=4,
            **({"use_auth_token": auth_token} if auth_token is not None else {}),
        )
        return str(hf_lora_path)

    def _prepare_lora_inputs(self, model_input: ModelInput, lora_adapter: LoraAdapter):
        model_input.lora_task_id = lora_adapter.lora_task_id
        model_input.lora_weights = lora_adapter.lora_weights
        model_input.lora_config = lora_adapter.lora_config
        return model_input

    async def register_and_infer_lora(
        self, model_input: ModelInput
    ) -> AsyncGenerator[str, None]:
        self._lora_task_index += 1
        task_index = self._lora_task_index

        lora_hf_repo = model_input.lora_hf_repo
        lora_hf_dir = self._download_lora(
            lora_hf_repo, auth_token=os.environ.get(HF_AUTH_KEY_CONSTANT, None)
        )
        tllm_lora_path = self._get_tllm_lora_path(lora_hf_repo)
        convert_hf_model(lora_hf_dir, LORA_CONVERSION_DTYPE, str(tllm_lora_path))

        lora_weights_data = np.load(
            tllm_lora_path / LORA_WEIGHTS_PATH, allow_pickle=True
        )
        lora_config_data = np.load(tllm_lora_path / LORA_CONFIG_PATH, allow_pickle=True)
        try:
            lora_adapter = LoraAdapter(
                lora_task_id=task_index,
                lora_weights=lora_weights_data,
                lora_config=lora_config_data,
            )
            self._lora_registry[lora_hf_repo] = task_index
            model_input = self._prepare_lora_inputs(model_input, lora_adapter)

            # In case of error the followup element in gen will have the error message
            # We rely on that error message to convey error details, and ignore the
            # first element here.
            _, gen = await _extract_first_element(self.triton_client.infer(model_input))
            return gen
        # TODO(Abu): Add proper error handling here, maybe a specific LoRAFailedToRegister error
        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            print(tb)
            print(f"LoRA registration for {lora_hf_repo} failed with error: {e}")

    async def infer_lora(self, model_input: ModelInput) -> AsyncGenerator[str, None]:
        lora_hf_repo: str = model_input.lora_hf_repo
        if lora_hf_repo in self._lora_registry:
            print("Found LoRA in registry")
            lora_task_id = self._lora_registry[lora_hf_repo]
            model_input.lora_task_id = lora_task_id
            first_element_error, gen = await _extract_first_element(
                self.triton_client.infer(model_input)
            )
            if first_element_error is not None:
                # TODO(pankaj) Need a better way to detect lora eviction but don't
                # know of one right now other than this string check.
                if "not found in cache" in first_element_error:
                    return await self.register_and_infer_lora(model_input)
            return gen
        else:
            print("LoRA not found in registry, registering")
            return await self.register_and_infer_lora(model_input)


async def _extract_first_element(gen):
    first_element = await anext(gen)

    async def remaining():
        async for item in gen:
            yield item

    return first_element, remaining()
