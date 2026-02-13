import torch
from cuda import cudart
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import engine_from_bytes
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import CLIPTextModelOutput


class TRTClip:
    def __init__(self, orig_clip, engine_path, stream=None, is_clip2=False):
        # Load trt engine
        self._orig_clip = orig_clip
        self._engine_path = engine_path
        self._stream = stream
        self._projection_dims = 1280 if is_clip2 else 768
        self._is_clip2 = is_clip2
        self._engine = None
        self._context = None
        self._tensors = {}

    def __getattr__(self, name):
        return getattr(self._orig_clip, name)

    def load(self):
        self._engine = engine_from_bytes(bytes_from_path(self._engine_path))
        self._context = self._engine.create_execution_context()
        if self._stream is None:
            self._stream = cudart.cudaStreamCreate()[1]

        self._tensors["input_ids"] = torch.empty(
            (1, 77), dtype=torch.int32, device="cuda"
        )
        text_embeddings_dims = (
            (1, self._projection_dims)
            if self._is_clip2
            else (1, 77, self._projection_dims)
        )
        self._tensors["text_embeddings"] = torch.empty(
            text_embeddings_dims, dtype=torch.float, device="cuda"
        )
        self._tensors["hidden_states"] = torch.empty(
            (1, 77, self._projection_dims), dtype=torch.float, device="cuda"
        )
        self._context.set_input_shape("input_ids", (1, 77))
        for name, tensor in self._tensors.items():
            self._context.set_tensor_address(name, tensor.data_ptr())

    def __call__(self, *args, **kwargs):
        text_input_ids = args[0]
        self._tensors["input_ids"].copy_(text_input_ids)
        noerror = self._context.execute_async_v3(self._stream)
        if not noerror:
            raise ValueError("ERROR: clip inference failed.")
        torch.cuda.synchronize()

        # [HACK] We should rebuild the clip trt engine to get rid of below hack.
        # HuggingFace model returns hidden states for all layers, but the engine
        # returns only the second last one, the one used by stable diffusion. So
        # create a corresponding tuple and only set the second last element as
        # the one returned by the engine. If diffusers implementation were to use
        # any other element then it would likely break.
        hf_model_hidden_states = [
            torch.empty(
                (1, 77, self._projection_dims), dtype=torch.float, device="cuda"
            )
        ] * 13
        hf_model_hidden_states[-2] = self._tensors["hidden_states"].clone()
        if self._is_clip2:
            return CLIPTextModelOutput(
                text_embeds=self._tensors["text_embeddings"].clone(),
                hidden_states=tuple(hf_model_hidden_states),
            )

        return BaseModelOutputWithPooling(
            hidden_states=tuple(hf_model_hidden_states),
        )
