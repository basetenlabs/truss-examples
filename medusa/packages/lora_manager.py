from huggingface_hub import snapshot_download
from hf_lora_convert import convert_hf_model
from pathlib import Path
from constants import LORA_LOCAL_CACHE_DIR, LORA_LOCAL_NPY_CACHE_DIR
import numpy as np
from typing import Dict
from triton_client import TritonClient
from schema import LoraAdapater

class LoraManager:
    def __init__(self, triton_client: TritonClient):
        self.triton_client = triton_client
        self.lora_cache_dir = LORA_LOCAL_CACHE_DIR
        self.converted_lora_dir = LORA_LOCAL_NPY_CACHE_DIR
        self.lora_cache_dir.mkdir(parents=True, exist_ok=True)
        self.converted_lora_dir.mkdir(parents=True, exist_ok=True)
        self._lora_map: Dict[str, LoraAdapater] = {}
            
    def _get_local_lora_path(self, lora_hf_id: str) -> Path:
        return self.lora_cache_dir / lora_hf_id
    
    def _get_converted_lora_path(self, lora_hf_id: str) -> Path:
        return self.converted_lora_dir / lora_hf_id
        
    def _download_lora(self, lora_hf_id, auth_token=None):
        snapshot_download(
            lora_hf_id,
            local_dir=self.lora_cache_dir,
            local_dir_use_symlinks=False,
            max_workers=4,
            **({"use_auth_token": auth_token} if auth_token is not None else {}),
        )
        return Path(self.lora_cache_dir / lora_hf_id)

    def register_lora(self, lora_hf_dir: str, lora_id: int = None) -> int:
        lora_id = len(self._lora_map) + 1
        local_lora_hf_path = self._download_lora(lora_hf_dir)
        converted_lora_path = self._get_converted_lora_path(lora_hf_dir)
        convert_hf_model(local_lora_hf_path, "float16", converted_lora_path)

        lora_weights_data = np.load(converted_lora_path / "model.lora_weights.npy", allow_pickle=True)
        lora_config_data = np.load(converted_lora_path / "model.lora_config.npy", allow_pickle=True)
        self._lora_map[lora_hf_dir] = LoraAdapater(
            lora_hf_path=lora_hf_dir,
            lora_id=lora_id,
            lora_weights=lora_weights_data,
            lora_config=lora_config_data,
        )
        
        try:
            self.triton_client.register_lora(self._lora_map[lora_hf_dir])
        # TODO(Abu): Add proper error handling here, maybe a specific LoRAFailedToRegister error
        except Exception as e:
            print(f"LoRA registeration for {lora_hf_dir} failed with error: {e}")
        return lora_id
    
    def get_lora_id(self, lora_hf_dir: str) -> int:
        if lora_hf_dir in self._lora_map:
            print("Found LoRA in cache")
            lora_adapter = self._lora_map[lora_hf_dir]
            return lora_adapter.lora_id
        else:
            print("LoRA not found in cache, registering")
            lora_id = self.register_lora(lora_hf_dir, lora_id)
            self._lora_map[lora_hf_dir] = lora_id
            