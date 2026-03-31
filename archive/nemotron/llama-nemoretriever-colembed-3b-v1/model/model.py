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
from typing import Dict, Any, Tuple

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
            query_embeddings = self._process_queries(model_input["queries"], batch_size)
            result["query_embeddings"] = query_embeddings.cpu().float().tolist()

        # Process passages if provided
        if "passages" in model_input:
            passage_embeddings = self._process_passages(
                model_input["passages"], batch_size
            )
            result["passage_embeddings"] = [
                t.cpu().float().tolist() for t in passage_embeddings
            ]

        # Compute scores if requested
        if compute_scores:
            scores = self._compute_similarity_scores(
                query_embeddings, passage_embeddings, batch_size
            )
            result["scores"] = scores.cpu().float().tolist()

        return result

    def _process_queries(self, queries: list, batch_size: int) -> torch.Tensor:
        """
        Process and encode text queries.

        Args:
            queries: List of query strings
            batch_size: Batch size for encoding

        Returns:
            Query embeddings tensor
        """
        if not isinstance(queries, list):
            raise ValueError(
                f"Queries must be a list of strings, got {type(queries)}: {queries}"
            )

        print(f"Encoding {len(queries)} queries...")
        return self._model.forward_queries(queries, batch_size=batch_size)

    def _process_passages(self, passages: list, batch_size: int) -> list:
        """
        Process and encode passages (text, images, or documents with both).

        Args:
            passages: List of passage strings or dicts
            batch_size: Batch size for encoding

        Returns:
            List of passage embedding tensors
        """
        if not isinstance(passages, list):
            raise ValueError(
                f"Passages must be a list of strings or dicts, got {type(passages)}: {passages}"
            )

        processed_passages = self._parse_passages(passages)
        print(f"Encoding {len(processed_passages)} passages...")

        # Separate passages by modality
        docs, images, texts = self._separate_passages_by_type(processed_passages)

        # Encode each modality
        doc_embeddings = (
            self._model.forward_documents(docs, batch_size=batch_size) if docs else []
        )
        img_embeddings = (
            self._model.forward_passages(images, batch_size=batch_size)
            if images
            else []
        )
        txt_embeddings = (
            self._model.forward_queries(texts, batch_size=batch_size) if texts else []
        )

        # Merge outputs preserving original order
        return self._merge_passage_embeddings(
            processed_passages, doc_embeddings, img_embeddings, txt_embeddings
        )

    def _parse_passages(self, passages: list) -> list:
        """
        Parse passages into a standardized format with type information.

        Args:
            passages: List of passage strings or dicts

        Returns:
            List of tuples (kind, payload) where kind is "text", "image", or "doc"
        """
        processed_passages = []
        for passage in passages:
            if isinstance(passage, str):
                # Text passage
                processed_passages.append(("text", passage))
            elif isinstance(passage, dict):
                if passage.get("type") == "image":
                    # Load and add image
                    image = self._load_image(passage)
                    if image is None:
                        raise ValueError(f"Failed to load image: {passage}")
                    text = passage.get("text", "") or ""
                    kind = "doc" if text else "image"
                    processed_passages.append((kind, (image, text)))
                else:
                    raise ValueError(f"Unsupported passage type: {type(passage)}")
            else:
                raise ValueError(f"Unsupported passage type: {type(passage)}")
        return processed_passages

    def _separate_passages_by_type(
        self, processed_passages: list
    ) -> Tuple[list, list, list]:
        """
        Separate processed passages into lists by modality.

        Args:
            processed_passages: List of (kind, payload) tuples

        Returns:
            Tuple of (docs, images, texts) lists
        """
        docs = []
        images = []
        texts = []
        for kind, payload in processed_passages:
            if kind == "doc":
                image, text = payload
                docs.append({"image": image, "text": text})
            elif kind == "image":
                image, _ = payload
                images.append(image)
            elif kind == "text":
                texts.append(payload)
        return docs, images, texts

    def _merge_passage_embeddings(
        self,
        processed_passages: list,
        doc_embeddings: list,
        img_embeddings: list,
        txt_embeddings: list,
    ) -> list:
        """
        Merge embeddings from different modalities while preserving original order.

        Args:
            processed_passages: List of (kind, payload) tuples in original order
            doc_embeddings: List of document embeddings
            img_embeddings: List of image embeddings
            txt_embeddings: List of text embeddings

        Returns:
            List of embeddings in original passage order
        """
        out_docs = iter(doc_embeddings)
        out_images = iter(img_embeddings)
        out_texts = iter(txt_embeddings)

        tensors = []
        for kind, payload in processed_passages:
            if kind == "doc":
                tensors.append(next(out_docs))
            elif kind == "image":
                tensors.append(next(out_images))
            else:  # text
                tensors.append(next(out_texts))
        return tensors

    def _compute_similarity_scores(
        self,
        query_embeddings: torch.Tensor,
        passage_embeddings: list,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Compute similarity scores between query and passage embeddings.

        Args:
            query_embeddings: Query embeddings tensor
            passage_embeddings: List of passage embedding tensors
            batch_size: Batch size for score computation

        Returns:
            Similarity scores tensor
        """
        if query_embeddings is None or passage_embeddings is None:
            raise ValueError(
                "Query embeddings and passage embeddings must be provided to compute scores"
            )

        q_list = self._as_list_of_2d(query_embeddings)
        p_list = self._as_list_of_2d(passage_embeddings)

        # Normalize embeddings to fix text/image scale mismatch
        q_list = [torch.nn.functional.normalize(q, dim=-1) for q in q_list]
        p_list = [torch.nn.functional.normalize(p, dim=-1) for p in p_list]

        return self._model.get_scores(q_list, p_list, batch_size=batch_size)

    def _as_list_of_2d(self, x) -> list:
        """
        Convert embeddings to a list of 2D tensors.

        Args:
            x: Embeddings as list, tensor, or other format

        Returns:
            List of 2D tensors
        """
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
        # else:
        #     raise ValueError("Image input must contain 'url' or 'content' field")
