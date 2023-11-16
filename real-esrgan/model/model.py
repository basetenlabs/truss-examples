import base64
import io
import os
import subprocess
import sys
from io import BytesIO
from typing import Dict

import numpy as np
from PIL import Image

git_repo_url = "https://github.com/xinntao/Real-ESRGAN.git"
git_clone_command = ["git", "clone", git_repo_url]
commit_hash = "5ca1078535923d485892caee7d7804380bfc87fd"
original_working_directory = os.getcwd()

try:
    subprocess.run(git_clone_command, check=True)
    print("Git repository cloned successfully!")

    os.chdir(os.path.join(original_working_directory, "Real-ESRGAN"))
    checkout_command = ["git", "checkout", commit_hash]
    subprocess.run(checkout_command, check=True)
    subprocess.run([sys.executable, "setup.py", "develop"], check=True)

except Exception as e:
    print(e)
    raise Exception("Error cloning Real-ESRGAN repo :(")

sys.path.append(os.path.join(os.getcwd()))

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


class Model:
    def __init__(self, **kwargs):
        self._data_dir = kwargs["data_dir"]
        self.model_checkpoint_path = os.path.join(
            original_working_directory,
            self._data_dir,
            "weights",
            "RealESRGAN_x4plus.pth",
        )
        self.model = None

    def pil_to_b64(self, pil_img):
        buffered = BytesIO()
        pil_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    def load(self):
        rrdb_net_model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        netscale = 4

        self.model = RealESRGANer(
            scale=netscale,
            model_path=self.model_checkpoint_path,
            model=rrdb_net_model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True,
        )

    def predict(self, request: Dict) -> Dict:
        image = request.get("image")
        scale = 4

        pil_img = Image.open(io.BytesIO(base64.decodebytes(bytes(image, "utf-8"))))
        pil_image_array = np.asarray(pil_img)

        output, _ = self.model.enhance(pil_image_array, outscale=scale)
        output = Image.fromarray(output)
        output_b64 = self.pil_to_b64(output)
        return {"upscaled_image": output_b64}
