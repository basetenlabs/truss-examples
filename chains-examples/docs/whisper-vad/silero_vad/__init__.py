from importlib.metadata import version

try:
    __version__ = version(__name__)
except:
    pass


def load_model(device_name: str):
    from pathlib import Path

    import torch
    from silero_vad.utils_vad import init_jit_model

    model_name = "silero_vad.jit"
    model_path = Path(__file__).parent / "data" / model_name
    gpu_model = init_jit_model(model_path, torch.device(device_name))
    return gpu_model


def load_cpu_model():
    return load_model("cpu")
