import numpy as np
import tensorrt as trt
import torch
from cuda import cudart
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import engine_from_bytes

numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool


class TRTUnet:
    def __init__(self, orig_unet, engine_path):
        # Load trt engine
        self._orig_unet = orig_unet
        self._engine_path = engine_path
        self._engine = None
        self._context = None
        self._stream = None
        self._tensors = {}
        self._binding_indices = {}

    def __getattr__(self, name):
        return getattr(self._orig_unet, name)

    def load(self):
        self._engine = engine_from_bytes(bytes_from_path(self._engine_path))
        self._context = self._engine.create_execution_context()
        self._stream = cudart.cudaStreamCreate()[1]

        # Couldn't find an engine api to get binding id from name, also
        # calling engine could be expensive, so cache these upfront. These
        # wouldn't change after.
        for idx in range(self._engine.num_io_tensors):
            binding = self._engine[idx]
            self._binding_indices[binding] = idx

    def infer(self, feed_dict):
        # We need to make sure tensors are allocated for inputs and outputs
        # to the engine and that input bindings are set up correctly.
        # We go through the supplied input and make sure of above.
        for name, buf in feed_dict.items():
            if name not in self._tensors or self._tensors[name].shape != buf.shape:
                # If a tensor isn't allocated or if not the right size then
                # allocate a new one and update binding shape if it's input.
                self._tensors[name] = torch.empty_like(buf, device="cuda")
                if self._engine.binding_is_input(name):
                    binding_idx = self._binding_indices[name]
                    desired_shape = buf.shape
                    # Map scaler to tuple, trt binding seems to need this.
                    if buf.shape == torch.Size([]):
                        desired_shape = buf.unsqueeze(0).shape
                    self._context.set_binding_shape(binding_idx, desired_shape)
            self._tensors[name].copy_(buf)

        # Handle output tensore now, i.e. the latent.
        # Latent size is goverened by the input. It is sized as:
        # (xB*batch_size, 4, latent_height, latent_width)
        # Where xB is 1 or 2 based on guidance scale.
        # We pick these up from input dimensions:
        # latent_height and latent_width from sample
        # and first one from encoder_hidden_states.
        latent_shape = (
            self._tensors["encoder_hidden_states"].shape[0],
            4,
            self._tensors["sample"].shape[2],
            self._tensors["sample"].shape[3],
        )
        if (
            "latent" not in self._tensors
            or self._tensors["latent"].shape != latent_shape
        ):
            self._tensors["latent"] = torch.empty(
                latent_shape, dtype=torch.float16, device="cuda"
            )

        # Register these tensors into the context, engine will read from and
        # write to these tensors.
        for name, tensor in self._tensors.items():
            self._context.set_tensor_address(name, tensor.data_ptr())

        noerror = self._context.execute_async_v3(self._stream)
        if not noerror:
            raise ValueError(f"ERROR: inference failed.")

        # synchronize since above call is async.
        torch.cuda.synchronize()
        return self._tensors

    def __call__(self, *args, **kwargs):
        sample = args[0]
        sample_float = sample.float() if sample.dtype != torch.float32 else sample
        timestep = args[1]
        timestep_float = (
            timestep.float() if timestep.dtype != torch.float32 else timestep
        )
        encoder_hidden_states = kwargs["encoder_hidden_states"]
        text_embeds = kwargs["added_cond_kwargs"]["text_embeds"]
        time_ids = kwargs["added_cond_kwargs"]["time_ids"]
        result = self.infer(
            {
                "sample": sample_float,
                "timestep": timestep_float,
                "encoder_hidden_states": encoder_hidden_states,
                "text_embeds": text_embeds,
                "time_ids": time_ids,
            }
        )
        return UNet2DConditionOutput(sample=result["latent"])
