import os
import torch
from PIL import Image
import requests
import base64
import io
from transformers import AutoModel
from torchvision.transforms import v2
from typing import Dict, Any
import math


class Model:
    def __init__(self, **kwargs):
        """Initialize the Kaiko Midnight model."""
        self._model = None
        self._transform = None
        self._secrets = kwargs.get("secrets", {})
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Model configuration
        self.model_id = "kaiko-ai/midnight"
        self.input_size = 224
        self.normalization_mean = (0.5, 0.5, 0.5)
        self.normalization_std = (0.5, 0.5, 0.5)

        # Request headers for external URLs
        self.request_headers = {
            "User-Agent": "Kaiko-Midnight-Model/1.0 (https://github.com/kaiko-ai/midnight; contact@kaiko.ai) Python/3.12"
        }

        # Batch size optimization
        self.optimal_batch_size = self._calculate_optimal_batch_size()

    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on available GPU memory."""
        if self.device == "cpu":
            return 1

        try:
            # Get GPU memory info
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            # Conservative estimate: each 224x224 image needs ~0.5GB for processing
            # Leave 2GB for model weights and intermediate computations
            available_memory_gb = gpu_memory_gb - 2.0
            estimated_batch_size = int(available_memory_gb / 0.5)

            # Clamp between 1 and 16 for stability
            optimal_batch_size = max(1, min(16, estimated_batch_size))

            print(f"[I] GPU Memory: {gpu_memory_gb:.1f}GB")
            print(f"[I] Estimated optimal batch size: {optimal_batch_size}")

            return optimal_batch_size

        except Exception as e:
            print(f"[W] Could not determine optimal batch size: {e}")
            return 4  # Default fallback

    def load(self):
        """Load the Kaiko Midnight model."""
        print(f"[I] Loading Kaiko Midnight model: {self.model_id}")

        # Set up HuggingFace token if provided
        if self._secrets and "hf_access_token" in self._secrets:
            os.environ["HF_TOKEN"] = self._secrets["hf_access_token"]

        try:
            # Load the model
            self._model = AutoModel.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,
                device_map="auto" if self.device == "cuda" else None,
            )

            # Set up image transformation pipeline
            self._transform = v2.Compose(
                [
                    v2.Resize(self.input_size),
                    v2.CenterCrop(self.input_size),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(
                        mean=self.normalization_mean, std=self.normalization_std
                    ),
                ]
            )

            # Move model to device
            if self.device == "cuda":
                self._model = self._model.to(self.device)

            # Set model to evaluation mode
            self._model.eval()

            print(f"[I] Model loaded successfully on {self.device}")

        except Exception as e:
            print(f"[E] Failed to load model: {e}")
            raise

    def predict(self, model_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate embeddings for the input image(s).

        Args:
            model_input: Dictionary containing:
                - image_url: URL of the image to process (optional if image_base64 provided)
                - image_base64: Base64-encoded image data (optional if image_url provided)
                - image_urls: List of image URLs for batch processing (optional)
                - image_base64_list: List of base64-encoded images for batch processing (optional)
                - task: "classification" or "segmentation"
                - batch_size: Optional, defaults to optimal batch size for GPU

        Returns:
            Dictionary containing embeddings and metadata
        """
        try:
            # Extract parameters
            image_url = model_input.get("image_url")
            image_base64 = model_input.get("image_base64")
            image_urls = model_input.get("image_urls", [])
            image_base64_list = model_input.get("image_base64_list", [])
            task = model_input.get("task", "classification")
            batch_size = model_input.get("batch_size", self.optimal_batch_size)

            # Debug logging
            print(f"[D] image_url: {type(image_url)} = {image_url}")
            print(
                f"[D] image_base64: {type(image_base64)} = {str(image_base64)[:50] if image_base64 else None}..."
            )
            print(f"[D] image_urls: {type(image_urls)} = {image_urls}")
            print(
                f"[D] image_base64_list: {type(image_base64_list)} = {len(image_base64_list) if image_base64_list else 0} items"
            )
            print(f"[D] task: {task}")
            print(f"[D] batch_size: {batch_size}")

            # Determine input type and load images
            images = []

            # Handle case where image_url contains a list (should be image_urls)
            if isinstance(image_url, list) and len(image_url) > 0:
                print("[I] Detected list in image_url, treating as image_urls")
                image_urls = image_url
                image_url = None

            if image_urls:
                # Batch processing with multiple URLs
                print(f"[I] Processing {len(image_urls)} URLs in batch")
                if not isinstance(image_urls, list):
                    raise ValueError(
                        f"image_urls must be a list, got {type(image_urls)}: {image_urls}"
                    )
                for i, url in enumerate(image_urls):
                    print(f"[I] Loading URL {i + 1}/{len(image_urls)}: {url}")
                    images.append(self._load_image(url))
            elif image_base64_list:
                # Batch processing with multiple base64 images
                print(f"[I] Processing {len(image_base64_list)} base64 images in batch")
                if not isinstance(image_base64_list, list):
                    raise ValueError(
                        f"image_base64_list must be a list, got {type(image_base64_list)}"
                    )
                for i, base64_img in enumerate(image_base64_list):
                    print(f"[I] Loading base64 image {i + 1}/{len(image_base64_list)}")
                    images.append(self._load_base64_image(base64_img))
            elif image_url:
                # Single image from URL
                print(f"[I] Processing single URL: {image_url}")
                images.append(self._load_image(image_url))
            elif image_base64:
                # Single image from base64
                print("[I] Processing single base64 image")
                images.append(self._load_base64_image(image_base64))
            else:
                raise ValueError(
                    "No valid image input provided. Use image_url, image_base64, image_urls, or image_base64_list"
                )

            # Validate batch size
            if len(images) > batch_size:
                print(
                    f"[W] Warning: {len(images)} images provided but batch_size is {batch_size}. Processing first {batch_size} images."
                )
                images = images[:batch_size]
            elif len(images) < batch_size:
                print(
                    f"[W] Warning: {len(images)} images provided but batch_size is {batch_size}. This may not be optimal for GPU utilization."
                )

            # Transform all images
            transformed_images = []
            for image in images:
                transformed_images.append(self._transform(image))

            # Stack into batch
            if len(transformed_images) == 1:
                batch = transformed_images[0].unsqueeze(dim=0)
            else:
                batch = torch.stack(transformed_images, dim=0)

            # Move to device
            batch = batch.to(self.device)

            # Generate embeddings
            with torch.no_grad():
                outputs = self._model(batch)
                last_hidden_state = outputs.last_hidden_state

                # Extract embeddings based on task
                if task == "classification":
                    embeddings = self._extract_classification_embedding(
                        last_hidden_state
                    )
                elif task == "segmentation":
                    embeddings = self._extract_segmentation_embedding(last_hidden_state)
                else:
                    raise ValueError(
                        f"Unsupported task: {task}. Use 'classification' or 'segmentation'"
                    )

                # Convert to numpy for serialization
                embeddings_np = embeddings.cpu().numpy()

                return {
                    "embeddings": embeddings_np.tolist(),
                    "embedding_shape": list(embeddings_np.shape),
                    "task": task,
                    "model_id": self.model_id,
                    "input_size": self.input_size,
                    "actual_batch_size": len(images),
                    "requested_batch_size": batch_size,
                    "optimal_batch_size": self.optimal_batch_size,
                    "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory
                    / (1024**3)
                    if self.device == "cuda"
                    else 0,
                }

        except Exception as e:
            print(f"[E] Prediction failed: {e}")
            raise

    def _load_image(self, image_url: str) -> Image.Image:
        """Load image from URL."""
        try:
            # Validate URL format
            if not isinstance(image_url, str):
                raise ValueError(
                    f"image_url must be a string, got {type(image_url)}: {image_url}"
                )

            if not image_url.startswith(("http://", "https://")):
                raise ValueError(f"Invalid URL format: {image_url}")

            print(f"[I] Loading image from URL: {image_url}")

            response = requests.get(
                image_url, stream=True, timeout=30, headers=self.request_headers
            )
            response.raise_for_status()
            image = Image.open(response.raw)

            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")

            print(f"[I] Successfully loaded image: {image.size}")
            return image

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                raise ValueError(
                    f"Access forbidden (403) for {image_url}. This may be due to missing User-Agent or access restrictions. Consider using image_base64 instead."
                )
            else:
                raise ValueError(
                    f"HTTP error {e.response.status_code} for {image_url}: {e}"
                )
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to load image from {image_url}: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load image from {image_url}: {e}")

    def _load_base64_image(self, image_base64: str) -> Image.Image:
        """Load image from base64-encoded string."""
        try:
            # Remove data URL prefix if present
            if image_base64.startswith("data:image"):
                image_base64 = image_base64.split(",")[1]

            # Decode base64 and open image
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))

            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")

            return image

        except Exception as e:
            raise ValueError(f"Failed to load base64 image: {e}")

    def _extract_classification_embedding(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings for classification tasks.

        Args:
            tensor: Model output tensor of shape (batch_size, seq_len, hidden_dim)

        Returns:
            Classification embeddings of shape (batch_size, hidden_dim * 2)
        """
        cls_embedding = tensor[:, 0, :]  # CLS token
        patch_embeddings = tensor[:, 1:, :]  # Patch tokens
        patch_mean = patch_embeddings.mean(dim=1)  # Average pooling over patches

        # Concatenate CLS token with mean patch embeddings
        return torch.cat([cls_embedding, patch_mean], dim=-1)

    def _extract_segmentation_embedding(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings for segmentation tasks.

        Args:
            tensor: Model output tensor of shape (batch_size, seq_len, hidden_dim)

        Returns:
            Segmentation embeddings of shape (batch_size, hidden_dim, height, width)
        """
        features = tensor[:, 1:, :].permute(0, 2, 1)  # Remove CLS token and transpose
        batch_size, hidden_size, patch_grid = features.shape

        # Reshape to spatial dimensions (16x16 for 224x224 input)
        height = width = int(math.sqrt(patch_grid))

        return features.view(batch_size, hidden_size, height, width)
