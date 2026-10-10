"""BGE-M3 model lifecycle and encoding operations."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import onnxruntime  # type: ignore
from huggingface_hub import constants, hf_hub_download, snapshot_download
from langchain_core.documents import Document
from tokenizers import Tokenizer

from config import DENSE_MODEL_NAME, EMBEDDING_INT8, EMBEDDING_MAX_LENGTH, INDEXER_GPU
from .embedding_cache import EmbeddingCache

logger = logging.getLogger("EspertoEmbedding")

class BGEM3Encoder:
    """BGE-M3 tokenizer and dense pooling around a backend computing the token embeddings."""

    def __init__(self, forward: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        self._forward = forward
        self._tokenizer = Tokenizer.from_file(hf_hub_download(DENSE_MODEL_NAME, "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=EMBEDDING_MAX_LENGTH)
        self._tokenizer.enable_padding(pad_id=1, pad_token="<pad>")

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return the normalized CLS vector of each text."""
        encodings = self._tokenizer.encode_batch(list(texts))
        input_ids = np.array([encoding.ids for encoding in encodings], dtype=np.int64)
        attention_mask = np.array([encoding.attention_mask for encoding in encodings], dtype=np.int64)
        dense = self._forward(input_ids, attention_mask)[:, 0]
        return dense / np.linalg.norm(dense, axis=1, keepdims=True)


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


def _load_gpu_model() -> BGEM3Encoder | None:
    """Create BGE-M3 on the NVIDIA GPU (PyTorch, fp16), or None where PyTorch or CUDA are missing."""
    try:  # torch e transformers sono installati solo nell'immagine costruita con INDEXER_GPU=1
        import torch
        from transformers import AutoModel  # type: ignore
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    # Caricato da cartella locale: dato il nome del repository, che ha solo pytorch_model.bin,
    # transformers ne scaricherebbe in background una conversione safetensors bloccando l'uscita.
    model_dir = snapshot_download(DENSE_MODEL_NAME, allow_patterns=["config.json", "pytorch_model.bin"])
    model = AutoModel.from_pretrained(model_dir).eval().half().to("cuda")

    def forward(input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            output = model(
                input_ids=torch.from_numpy(input_ids).to("cuda"),
                attention_mask=torch.from_numpy(attention_mask).to("cuda"),
            )
        return output.last_hidden_state.float().cpu().numpy()

    logger.info("BGE-M3 caricato (%s, GPU, fp16).", DENSE_MODEL_NAME)
    return BGEM3Encoder(forward)


def load_indexing_model() -> BGEM3Encoder:
    """Create the BGE-M3 model used by the one-shot indexing process, on GPU when INDEXER_GPU asks for it."""
    if INDEXER_GPU:
        model = _load_gpu_model()
        if model is not None:
            return model
        logger.warning("INDEXER_GPU attivo ma PyTorch o CUDA non disponibili: indicizzazione su CPU.")
    return _load_cpu_model()


def _load_cpu_model() -> BGEM3Encoder:
    """Create the BGE-M3 model on CPU with ONNX Runtime."""
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
            _query_model = _load_cpu_model()
    return _query_model


def encode_documents(model: BGEM3Encoder, documents: Sequence[Document], cache: EmbeddingCache) -> list[np.ndarray]:
    """Encode an indexing batch into BGE-M3 dense vectors, reusing cached vectors."""
    texts = [document.page_content for document in documents]
    vectors = cache.get(texts)
    missing = [index for index, cached in enumerate(vectors) if cached is None]
    if missing:
        encoded = list(model.encode([texts[index] for index in missing]))
        cache.put([texts[index] for index in missing], encoded)
        for index, fresh in zip(missing, encoded):
            vectors[index] = fresh
    return vectors


def encode_queries(texts: Sequence[str]) -> list[list[float]]:
    """Encode all expanded queries into dense vectors, in one CPU BGE-M3 forward pass."""
    return get_query_model().encode(texts).tolist()
