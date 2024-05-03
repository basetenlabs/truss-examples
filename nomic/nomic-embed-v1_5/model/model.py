import torch.nn.functional as F
from huggingface_hub import login as hf_login
from sentence_transformers import SentenceTransformer


class Model:
    def __init__(self, **kwargs):
        self._model = None
        self._hf_access_token = kwargs["secrets"]["hf_access_token"]

    def preprocess(self, model_input):
        task_type = model_input.get("task_type", "search_document")
        model_input["texts"] = [
            task_type + ": " + text for text in model_input["texts"]
        ]
        return model_input

    def load(self):
        hf_login(token=self._hf_access_token)
        self._model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
        )

    def predict(self, model_input):
        texts = model_input.get("texts")
        m_dim = model_input.get(
            "dimensionality", 768
        )  # m_dim must be in range [64, 768]

        embeddings = self._model.encode(texts, convert_to_tensor=True)
        embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
        if m_dim < 768:
            embeddings = embeddings[:, :m_dim]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.tolist()
