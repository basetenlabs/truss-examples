from typing import Any, Dict, List
import gc
import torch
from PIL import Image
import base64
import os
from io import BytesIO
from diffusers import DiffusionPipeline
from transformers import T5EncoderModel
import huggingface_hub


class Model:
    def __init__(self, data_dir: str, config: Dict, secrets: Dict) -> None:
        """
        Initialize the Model class.

        Parameters:
        - data_dir (str): Directory for the data.
        - config (dict): Configuration dictionary.
        - secrets (dict): Secrets dictionary.
        """
        self._data_dir = data_dir
        self._config = config
        self._secrets = secrets
        self.model_id = "DeepFloyd/IF-I-XL-v1.0"
        self.hf_token = self._secrets["hf_api_key"]
        self.generator = None
        self.text_encoder = None
        self.first_stage = None
        self.unet = None
        self.second_stage = None
        self.upscaler = None
    
    def login(self):
        huggingface_hub.login(self.hf_token)

    def flush(self):
        """
        Free up memory by calling garbage collector and emptying CUDA cache.
        """
        gc.collect()
        torch.cuda.empty_cache()

    def load_text_encoder(self):
        """
        Load the text encoder model.
        """
        self.text_encoder = T5EncoderModel.from_pretrained(
            self.model_id,
            subfolder="text_encoder",
            device_map="auto",
            load_in_8bit=True,
            variant="8bit"
        )
        torch.compile(self.text_encoder)

    def load_deepfloyd_first_stage_pipeline(self):
        """
        Load the first stage diffusion pipeline model.
        """
        self.first_stage = DiffusionPipeline.from_pretrained(
            self.model_id,
            text_encoder=None,
            device_map="auto",
            variant="fp16",
            torch_dtype=torch.float16,
        )
        self.unet = self.first_stage.unet
        torch.compile(self.unet)
        del self.first_stage.unet

    def load_deepfloyd_second_stage_pipeline(self):
        """
        Load the second stage diffusion pipeline model.
        """
        self.second_stage = DiffusionPipeline.from_pretrained(
            "DeepFloyd/IF-II-L-v1.0",
            text_encoder=None,
            variant="fp16",
            torch_dtype=torch.float16,
        )

    def load_upscaler(self):
        """
        Load the upscaler model.
        """
        self.upscaler = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler",
            torch_dtype=torch.float16,
        )

    def load(self):
        """
        Load all required models.
        """
        self.login()
        self.load_text_encoder()
        self.load_deepfloyd_first_stage_pipeline()
        self.load_deepfloyd_second_stage_pipeline()
        self.load_upscaler()
        self.generator = torch.Generator("cuda").manual_seed(1)

    def forward(self, 
                prompt: str, 
                negative_prompt: str, 
                encoder_kwargs: Dict = {},
                first_stage_kwargs: Dict = {},
                second_stage_kwargs: Dict = {},
                upscaler_kwargs: Dict = {}):
        """
        Forward pass through the model pipelines.

        Parameters:
        - prompt (str): The input prompt.
        - negative_prompt (str): The negative input prompt.
        - encoder_kwargs (dict): Additional arguments for encoder.
        - first_stage_kwargs (dict): Additional arguments for first stage.
        - second_stage_kwargs (dict): Additional arguments for second stage.
        - upscaler_kwargs (dict): Additional arguments for upscaler.

        Returns
        - image: The final output image.
        """
        # Run prompt through deepfloyd with text encoder without unet 
        self.first_stage.to("cuda")
        self.first_stage.text_encoder = self.text_encoder
        prompt_embeds, negative_embeds = self.first_stage.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            **encoder_kwargs,
        )
        self.text_encoder = self.first_stage.text_encoder
        self.first_stage.text_encoder = None
        self.first_stage.to("cpu")
        self.flush()

        # Run through deepfloyd with unet without text encoder
        self.unet = self.unet.to("cuda")
        self.first_stage.unet = self.unet
        image = self.first_stage(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds, 
            output_type="pt",
            generator=self.generator,
            num_inference_steps=50,
            **first_stage_kwargs,
        ).images

        self.unet = self.first_stage.unet.to("cpu")
        self.first_stage.unet = None
        self.first_stage.to("cpu")
        self.flush()

        # Run through deepfloyd 2nd stage with unet without text encoder
        self.second_stage = self.second_stage.to("cuda")
        image = self.second_stage(
            image=image, 
            prompt_embeds=prompt_embeds, 
            negative_prompt_embeds=negative_embeds, 
            output_type="pt",
            generator=self.generator,
            num_inference_steps=50,
            **second_stage_kwargs,
        ).images

        self.second_stage.to("cpu")
        self.flush()

        # Run through upscaler
        self.upscaler.to("cuda")
        image = self.upscaler(
            prompt,
            generator=self.generator, 
            image=image,
            **upscaler_kwargs,
        ).images

        self.upscaler.to("cpu")
        self.flush()

        return image
    
    def convert_to_b64(self, image: Image) -> str:
        """
        Convert the image to base64 format.

        Parameters:
        - image (PIL.Image): The image to convert.

        Returns:
        - img_b64 (str): The base64 representation of the image.
        """
        buffered = BytesIO()
        image[0].save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_b64 
    
    def predict(self, model_input: Dict) -> Dict:
        """
        Predict the output based on the model input.

        Parameters:
        - model_input (dict): The input for the model.

        Returns:
        - dict: A dictionary containing the status, data and message of the prediction.
        """
        prompt = model_input.pop("prompt")
        negative_prompt = model_input.pop("negative_prompt", None)
        encoder_kwargs = model_input.pop("encoder_kwargs", {})
        first_stage_kwargs = model_input.pop("first_stage_kwargs", {})
        second_stage_kwargs = model_input.pop("second_stage_kwargs", {})
        upscaler_kwargs = model_input.pop("upscaler_kwargs", {})
        random_seed = int.from_bytes(os.urandom(2), "big")
        seed = model_input.pop("seed", random_seed)
        
        self.generator = torch.Generator("cuda").manual_seed(seed)
        
        image = self.forward(
            prompt, 
            negative_prompt, 
            encoder_kwargs,
            first_stage_kwargs,
            second_stage_kwargs,
            upscaler_kwargs,
        )
        
        encoded_image = self.convert_to_b64(image)
        
        return {"status": "success", "data": [encoded_image], "message": None}
