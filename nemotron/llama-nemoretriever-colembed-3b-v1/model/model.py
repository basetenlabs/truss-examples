"""
NVIDIA Llama NemoRetriever ColEmbed 3B V1 - Traditional Truss Implementation

This model provides high-quality embeddings using NVIDIA's ColEmbed approach for:
- Cross-modal retrieval (text queries with image documents)
- Text-to-text retrieval
- Image document embedding
- RAG applications with multimodal content
"""

import base64
import io
import os
from typing import Dict, Any

import requests
import torch
from PIL import Image
from transformers import AutoModel


class Model:
    def __init__(self, **kwargs):
        """Initialize the model. This runs once when the container starts."""
        self._model = None
        self._device = None

    def load(self):
        """
        Load the model into memory.
        This function runs once when the model server starts.
        """
        # Set up device
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self._device}")

        # Try to read the HuggingFace token from the /secrets/hf_access_token file if it exists
        hf_token = None
        hf_token_path = "/secrets/hf_access_token"
        if os.path.exists(hf_token_path):
            with open(hf_token_path, "r") as f:
                hf_token = f.read().strip()

        if hf_token is None:
            raise ValueError("HuggingFace token not found in /secrets/hf_access_token")

        # Load model with specific revision and flash attention
        print("Loading model...")
        self._model = AutoModel.from_pretrained(
            "nvidia/llama-nemoretriever-colembed-3b-v1",
            device_map="cuda",
            token=hf_token,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            revision="50c36f4d5271c6851aa08bd26d69f6e7ca8b870c",
        ).eval()

        print(
            "Model loaded successfully! Supports text queries and text/image passages."
        )

    def predict(self, model_input: Dict) -> Dict[str, Any]:
        """
        Generate embeddings for queries and passages, or compute similarity scores.

        Args:
            model_input: Dictionary containing:
              - queries (List[str]): Text queries to encode
              - passages (List[str] or List[Dict]): Text passages or images to encode.
                Can be text strings or dicts with {"type": "image", "url": "..."} or {"type": "image", "content": "base64..."}
              - batch_size (int, optional): Batch size for encoding. Default: 8
              - compute_scores (bool, optional): Whether to compute similarity scores. Default: False

        Returns:
            - query_embeddings: List of query embedding tensors
            - passage_embeddings: List of passage embedding tensors
        """
        batch_size = model_input.get("batch_size", 8)
        compute_scores = model_input.get("compute_scores", False)

        result = {}
        query_embeddings = None
        passage_embeddings = None

        # Process queries if provided
        if "queries" in model_input:
            queries = model_input["queries"]
            if not isinstance(queries, list):
                raise ValueError(
                    f"Queries must be a list of strings, got {type(queries)}: {queries}"
                )

            print(f"Encoding {len(queries)} queries...")
            query_embeddings = self._model.forward_queries(
                queries, batch_size=batch_size
            )

            # Convert to list for JSON serialization
            result["query_embeddings"] = query_embeddings.cpu().float().tolist()

        # Process passages if provided
        if "passages" in model_input:
            passages = model_input["passages"]
            if not isinstance(passages, list):
                raise ValueError(
                    f"Passages must be a list of strings or dicts, got {type(passages)}: {passages}"
                )

            # Process passages - can be text or images
            processed_passages = []
            for passage in passages:
                if isinstance(passage, str):
                    # Text passage
                    processed_passages.append(passage)
                elif isinstance(passage, dict):
                    # Check if it's an image
                    if passage.get("type") == "image":
                        # Load and add image
                        image = self._load_image(passage)
                        if image is None:
                            raise ValueError(f"Failed to load image: {passage}")
                        processed_passages.append(image)
                    else:
                        raise ValueError(f"Unsupported passage type: {type(passage)}")
                else:
                    raise ValueError(f"Unsupported passage type: {type(passage)}")

            print(f"Encoding {len(processed_passages)} passages...")

            image_passages = [
                p for p in processed_passages if isinstance(p, Image.Image)
            ]
            text_passages = [p for p in processed_passages if isinstance(p, str)]

            # forward_passages expects a list of images
            tensors = []
            if len(image_passages) > 0:
                img_out = self._model.forward_passages(
                    image_passages, batch_size=batch_size
                )
                tensors.append(self._norm_embs(img_out))
            if len(text_passages) > 0:
                txt_out = self._model.forward_queries(
                    text_passages, batch_size=batch_size
                )
                tensors.append(self._norm_embs(txt_out))

            if len(tensors) > 0:
                passage_embeddings = torch.cat(tensors, dim=0)
            else:
                passage_embeddings = torch.empty(
                    (0, getattr(self._model.config, "hidden_size", 0))
                )

            # # forward_queries expects a list of strings
            # text_embs = self._model.forward_queries(text_passages, batch_size=batch_size)
            # # combine above two lists into a single list
            # passage_embeddings = torch.cat([text_embs, image_embs], dim=0)

            # Convert to list for JSON serialization

            result["passage_embeddings"] = passage_embeddings.cpu().float().tolist()
            #
            # # forward_documents
            # image_embs = self._model.forward_passages(image_passages, batch_size=batch_size)
            # text_embs = self._model.forward_passages(text_passages, batch_size=batch_size)

            # # passage_embeddings = self._model.forward_passages(
            # #     processed_passages, batch_size=batch_size
            # # )

            # passage_embeddings = torch.cat([image_embs, text_embs], dim=0)

            # # Convert to list for JSON serialization
            # result["passage_embeddings"] = passage_embeddings.cpu().float().tolist()

        # Compute scores if requested
        if compute_scores:
            if query_embeddings is None or passage_embeddings is None:
                raise ValueError(
                    "Query embeddings and passage embeddings must be provided to compute scores"
                )

            def _as_list_of_2d(x):
                if isinstance(x, list):
                    return x
                if isinstance(x, torch.Tensor):
                    # Squeeze any singleton batch dimension at the front
                    while x.ndim > 3 and x.shape[0] == 1:
                        x = x.squeeze(0)
                    if x.ndim == 3:  # [B, N, D]
                        return [x[i] for i in range(x.shape[0])]
                    if x.ndim == 2:  # [N, D]
                        return [x]
                    if x.ndim == 1:  # [D]
                        return [x.unsqueeze(0)]
                raise TypeError(
                    f"Unexpected type for embeddings: {type(x)}, shape={getattr(x, 'shape', None)}"
                )

            q_list = _as_list_of_2d(query_embeddings)
            p_list = _as_list_of_2d(passage_embeddings)

            scores = self._model.get_scores(q_list, p_list, batch_size=batch_size)
            result["scores"] = scores.cpu().float().tolist()

        # # Compute scores if requested
        # if compute_scores:
        #     if (
        #         result["query_embeddings"] is None
        #         or result["passage_embeddings"] is None
        #     ):
        #         raise ValueError(
        #             "Query embeddings and passage embeddings must be provided to compute scores"
        #         )
        #     scores = self._model.get_scores(
        #         result["query_embeddings"], result["passage_embeddings"]
        #     )
        #     result["scores"] = scores.cpu().float().tolist()

        return result

    def _load_image(self, image_input: Dict) -> Image.Image:
        """
        Load an image from URL or base64 string.

        Args:
            image_input: Dict with either "url" or "content" (base64)

        Returns:
            PIL Image
        """
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

        if "url" in image_input:
            url = image_input["url"]
            if url.startswith("data:image"):
                # Base64 encoded image in data URL
                base64_str = url.split(",")[1]
                image_data = base64.b64decode(base64_str)
                img = Image.open(io.BytesIO(image_data))
                if img.mode != "RGB":
                    img = img.convert("RGB")
                return img
            else:
                # Regular URL
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                img = Image.open(io.BytesIO(response.content))
                if img.mode != "RGB":
                    img = img.convert("RGB")
                return img
        elif "content" in image_input:
            # Base64 string
            base64_str = image_input["content"]
            if "," in base64_str:
                base64_str = base64_str.split(",")[1]
            image_data = base64.b64decode(base64_str)
            img = Image.open(io.BytesIO(image_data))
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        else:
            raise ValueError("Image input must contain 'url' or 'content' field")

    def _norm_embs(self, out):
        if isinstance(out, list):
            # pool each item to a single vector, then stack -> [N, D]
            pooled = []
            for t in out:
                if t.ndim == 2:  # [seq_len, dim] -> mean over tokens
                    pooled.append(t.mean(dim=0))
                elif t.ndim == 1:  # [dim]
                    pooled.append(t)
                else:
                    pooled.append(t.reshape(-1))
            return torch.stack(pooled, dim=0)
        elif isinstance(out, torch.Tensor):
            return out if out.ndim == 2 else out.unsqueeze(0)
        else:
            raise TypeError(f"Unexpected embedding output type: {type(out)}")

    # def _as_tokens_3d(self, x):
    #     if isinstance(x, list):
    #         # list of [tokens, D] -> pad to max tokens and stack -> [B, N, D]
    #         N = max(t.shape[0] if t.ndim == 2 else 1 for t in x)
    #         D = x[0].shape[-1] if x and x[0].ndim >= 1 else 0
    #         out = []
    #         for t in x:
    #             if t.ndim == 1:          # [D] -> [1, D]
    #                 t = t.unsqueeze(0)
    #             if t.shape[0] < N:       # pad tokens
    #                 pad = torch.zeros(N - t.shape[0], D, device=t.device, dtype=t.dtype)
    #                 t = torch.cat([t, pad], dim=0)
    #             out.append(t)
    #         return torch.stack(out, dim=0)  # [B, N, D]
    #     # tensor
    #     if x.ndim == 2:   # [B, D] -> add token dim: [B, 1, D]
    #         return x.unsqueeze(1)
    #     return x          # already [B, N, D]
