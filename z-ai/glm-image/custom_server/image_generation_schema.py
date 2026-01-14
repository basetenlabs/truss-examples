from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(
        ...,
        description="The text description of the desired image(s). Maximum length is typically 4000 characters.",
        min_length=1,
    )
    model: Optional[str] = Field(
        None,
        description="The ID of the model to use for image generation. If not specified, the default model is used.",
    )
    n: int = Field(
        default=1,
        description="The number of images to generate. Must be between 1 and 10.",
        ge=1,
        le=10,
    )
    size: str = Field(
        default="1024x1024",
        description="The size of the generated images in the format WIDTHxHEIGHT. Common sizes: 256x256, 512x512, 1024x1024, 1792x1024, 1024x1792.",
        pattern=r"^\d+x\d+$",
    )
    quality: Optional[Literal["standard", "hd", "auto"]] = Field(
        default="auto",
        description="The quality of the image that will be generated. 'hd' creates images with finer details and greater consistency across the image.",
    )
    response_format: Literal["b64_json"] = Field(
        default="b64_json",
        description="The format in which the generated images are returned. Must be 'b64_json' for this endpoint.",
    )
    output_format: Optional[Literal["png", "jpeg", "webp"]] = Field(
        None,
        description="The format of the output image. If not specified, defaults to PNG.",
    )
    seed: Optional[int] = Field(
        default=1024,
        description="A random seed for reproducibility. Using the same seed with the same prompt will produce similar images.",
        ge=0,
    )
    generator_device: Optional[Literal["cuda", "cpu"]] = Field(
        default="cuda",
        description="The device to use for generation. 'cuda' for GPU, 'cpu' for CPU.",
    )
    user: Optional[str] = Field(
        None,
        description="A unique identifier representing the end-user. This can help the system to monitor and detect abuse.",
    )
    negative_prompt: Optional[str] = Field(
        None,
        description="A text description of what you do NOT want in the image. This can help guide the model away from unwanted elements.",
    )
    guidance_scale: Optional[float] = Field(
        None,
        description="The guidance scale for generation. Higher values encourage the model to generate images that are more closely linked to the text prompt.",
        ge=0.0,
        le=20.0,
    )
    num_inference_steps: Optional[int] = Field(
        None,
        description="The number of denoising steps. More steps typically result in higher quality but take longer to generate.",
        ge=1,
        le=100,
    )

    @field_validator("size")
    @classmethod
    def validate_size(cls, v: str) -> str:
        width, height = v.split("x")
        width_int = int(width)
        height_int = int(height)

        if width_int < 64 or height_int < 64:
            raise ValueError("Size must be at least 64x64")
        if width_int > 4096 or height_int > 4096:
            raise ValueError("Size must be at most 4096x4096")

        return v
