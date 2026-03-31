# <- download is invoked before here.
import pathlib


class Model:
    """example usage of `model_cache` in truss"""

    def __init__(self, *args, **kwargs):
        # `lazy_data_resolver` is passed as keyword-argument in init
        self._lazy_data_resolver = kwargs["lazy_data_resolver"]
        self.tensor_size = None

    def load(self):
        # work that does not require the download may be done beforehand
        # important to collect the download before using any incomplete data
        self._lazy_data_resolver.block_until_download_complete()
        # after the call, you may use the /app/model_cache directory and the contents
        # torch.load(
        #     "/app/model_cache/stable-diffusion-xl-base/vae_1_0/diffusion_pytorch_model.fp16.safetensors",
        #     weights_only=True
        # )
        self.tensor_size = (
            pathlib.Path("/app/model_cache/llama/model.safetensors").stat().st_size
        )
        print(
            "Model loaded successfully with size of {} bytes".format(self.tensor_size)
        )

    def predict(self, input_data):
        # this method will be called by the serving container
        # you may use the model here, after the download is complete
        return {"input": input_data, "tensor_size": self.tensor_size}
