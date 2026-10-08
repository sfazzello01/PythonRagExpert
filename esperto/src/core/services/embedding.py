"""BGE-M3 model lifecycle and encoding operations."""

from __future__ import annotations

import logging
import pickle
import threading
import zipfile
from collections import OrderedDict
from typing import Callable, Sequence

import numpy as np
import onnxruntime  # type: ignore
from huggingface_hub import hf_hub_download, snapshot_download
from langchain_core.documents import Document
from tokenizers import Tokenizer

from config import DENSE_MODEL_NAME, EMBEDDING_MAX_LENGTH
from ..shared.vectors import QueryVectors, lexical_weights_to_sparse_vector

logger = logging.getLogger("EspertoEmbedding")

# <s>, <pad>, </s>, <unk>: esclusi dai pesi lessicali, come in FlagEmbedding.
_SPECIAL_TOKEN_IDS = frozenset({0, 1, 2, 3})


def _load_sparse_head() -> tuple[np.ndarray, float]:
    """Read BGE-M3's sparse head (a 1024->1 linear layer saved by torch) without torch."""
    with zipfile.ZipFile(hf_hub_download(DENSE_MODEL_NAME, "sparse_linear.pt")) as archive:
        pickle_name = next(name for name in archive.namelist() if name.endswith("data.pkl"))

        class TensorUnpickler(pickle.Unpickler):
            def find_class(self, module: str, name: str):
                if name == "OrderedDict":
                    return OrderedDict
                if name == "_rebuild_tensor_v2":
                    return lambda storage, offset, size, *_: storage[offset:].reshape(size)
                return {"HalfStorage": np.float16, "FloatStorage": np.float32}[name]

            def persistent_load(self, pid: tuple):
                data = archive.read(f"{pickle_name.removesuffix('data.pkl')}data/{pid[2]}")
                return np.frombuffer(data, dtype=pid[1])

        state = TensorUnpickler(archive.open(pickle_name)).load()
    return state["weight"][0].astype(np.float32), float(state["bias"][0])


class BGEM3Encoder:
    """BGE-M3 tokenizer, dense pooling and sparse head around a backend computing the token embeddings."""

    def __init__(self, forward: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        self._forward = forward
        self._tokenizer = Tokenizer.from_file(hf_hub_download(DENSE_MODEL_NAME, "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=EMBEDDING_MAX_LENGTH)
        self._tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
        self._sparse_weight, self._sparse_bias = _load_sparse_head()

    def encode(self, texts: Sequence[str]) -> dict:
        """Return normalized CLS vectors and, per text, the highest weight of each token id."""
        encodings = self._tokenizer.encode_batch(list(texts))
        input_ids = np.array([encoding.ids for encoding in encodings], dtype=np.int64)
        attention_mask = np.array([encoding.attention_mask for encoding in encodings], dtype=np.int64)
        token_embeddings = self._forward(input_ids, attention_mask)

        dense = token_embeddings[:, 0]
        dense = dense / np.linalg.norm(dense, axis=1, keepdims=True)
        token_weights = token_embeddings @ self._sparse_weight + self._sparse_bias
        lexical_weights = []
        for ids, weights in zip(input_ids.tolist(), token_weights.tolist()):
            lexical: dict[str, float] = {}
            for token_id, weight in zip(ids, weights):
                if token_id not in _SPECIAL_TOKEN_IDS and weight > lexical.get(str(token_id), 0.0):
                    lexical[str(token_id)] = weight
            lexical_weights.append(lexical)
        return {"dense_vecs": dense, "lexical_weights": lexical_weights}


_query_model: BGEM3Encoder | None = None
_query_model_lock = threading.Lock()


def load_indexing_model() -> BGEM3Encoder:
    """Create the BGE-M3 model (ONNX Runtime, CPU) used by the one-shot indexing process."""
    logger.info("Caricamento BGE-M3 su CPU con ONNX Runtime (%s).", DENSE_MODEL_NAME)
    model_dir = snapshot_download(DENSE_MODEL_NAME, allow_patterns=["onnx/*"])
    session = onnxruntime.InferenceSession(f"{model_dir}/onnx/model.onnx", providers=["CPUExecutionProvider"])
    return BGEM3Encoder(
        lambda input_ids, attention_mask: session.run(
            ["token_embeddings"], {"input_ids": input_ids, "attention_mask": attention_mask}
        )[0]
    )


def get_query_model() -> BGEM3Encoder:
    """Lazily load the BGE-M3 model shared by all query requests."""
    global _query_model
    with _query_model_lock:  # richieste concorrenti non devono caricare il modello piu' volte
        if _query_model is None:
            _query_model = load_indexing_model()
    return _query_model


def encode_documents(model: BGEM3Encoder, documents: Sequence[Document]) -> dict:
    """Encode an indexing batch with the dense and sparse BGE-M3 representations."""
    return model.encode([document.page_content for document in documents])


def encode_queries(texts: Sequence[str]) -> list[QueryVectors]:
    """Encode all expanded queries in one CPU BGE-M3 forward pass."""
    output = get_query_model().encode(texts)
    return [
        QueryVectors(
            dense=dense_vector.tolist(),
            sparse=lexical_weights_to_sparse_vector(lexical_weights),
        )
        for dense_vector, lexical_weights in zip(
            output["dense_vecs"], output["lexical_weights"]
        )
    ]