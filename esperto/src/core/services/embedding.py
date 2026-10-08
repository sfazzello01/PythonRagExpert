"""BGE-M3 model lifecycle and encoding operations."""

from __future__ import annotations

import logging
import os
import pickle
import shutil
import tempfile
import threading
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import onnxruntime  # type: ignore
from huggingface_hub import constants, hf_hub_download, snapshot_download
from langchain_core.documents import Document
from tokenizers import Tokenizer

from config import DENSE_MODEL_NAME, EMBEDDING_INT8, EMBEDDING_MAX_LENGTH
from ..shared.vectors import QueryVectors, lexical_weights_to_sparse_vector
from .embedding_cache import EmbeddingCache

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


def _quantized_model(model_dir: Path) -> Path:
    """Return the int8 copy of the ONNX model, created next to the HuggingFace cache on first use."""
    target = Path(constants.HF_HOME) / f"{DENSE_MODEL_NAME.replace('/', '--')}-int8.onnx"
    if not target.exists():
        from onnxruntime.quantization import QuantType, quantize_dynamic  # type: ignore

        logger.info("Quantizzazione int8 di BGE-M3 (solo al primo avvio).")
        with tempfile.TemporaryDirectory(dir=target.parent) as work:
            # onnx rifiuta i link della cache HuggingFace: la quantizzazione lavora su una copia.
            shutil.copytree(model_dir, f"{work}/onnx")
            quantize_dynamic(
                f"{work}/onnx/model.onnx", f"{work}/int8.onnx", weight_type=QuantType.QInt8, per_channel=True
            )
            os.replace(f"{work}/int8.onnx", target)
    return target


def _execution_providers() -> list[str]:
    """OpenVINO on Intel CPUs when the runtime ships it, the default CPU provider otherwise."""
    try:
        intel_cpu = "GenuineIntel" in Path("/proc/cpuinfo").read_text()
    except OSError:
        intel_cpu = False
    # Il modello int8 resta sul provider CPU: con OpenVINO e' piu' lento del modello originale.
    if intel_cpu and not EMBEDDING_INT8 and "OpenVINOExecutionProvider" in onnxruntime.get_available_providers():
        return ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def load_indexing_model() -> BGEM3Encoder:
    """Create the BGE-M3 model (ONNX Runtime, CPU) used by the one-shot indexing process."""
    model_dir = Path(snapshot_download(DENSE_MODEL_NAME, allow_patterns=["onnx/*"])) / "onnx"
    model_file = str(_quantized_model(model_dir) if EMBEDDING_INT8 else model_dir / "model.onnx")
    providers = _execution_providers()
    try:
        session = onnxruntime.InferenceSession(model_file, providers=providers)
    except Exception as error:
        if providers == ["CPUExecutionProvider"]:
            raise
        logger.warning("Provider %s non utilizzabile (%s): uso il provider CPU.", providers[0], error)
        session = onnxruntime.InferenceSession(model_file, providers=["CPUExecutionProvider"])
    logger.info(
        "BGE-M3 caricato (%s, int8=%s, provider=%s).", DENSE_MODEL_NAME, EMBEDDING_INT8, session.get_providers()[0]
    )
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


def encode_documents(model: BGEM3Encoder, documents: Sequence[Document], cache: EmbeddingCache) -> dict:
    """Encode an indexing batch with the dense and sparse BGE-M3 representations, reusing cached vectors."""
    texts = [document.page_content for document in documents]
    vectors = cache.get(texts)
    missing = [index for index, cached in enumerate(vectors) if cached is None]
    if missing:
        output = model.encode([texts[index] for index in missing])
        encoded = list(zip(output["dense_vecs"], output["lexical_weights"]))
        cache.put([texts[index] for index in missing], encoded)
        for index, fresh in zip(missing, encoded):
            vectors[index] = fresh
    return {
        "dense_vecs": [dense for dense, _ in vectors],
        "lexical_weights": [lexical for _, lexical in vectors],
    }


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