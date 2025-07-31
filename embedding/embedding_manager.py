# embedding_manager.py
# =============================================================
"""
ABS Embedding Manager
---------------------------------------------------------------
Handles prompt-based embedding bundles in:

    <embeddings>/cached_embeddings/

Each bundle comprises:
    • <md5>.safetensors — tensor data
    • <md5>.json        — metadata incl. dims, prompt_text, etc.

Supports prompt similarity matching using TF-IDF char n-gram vectors.
"""
import hashlib, json, os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from folder_paths import get_folder_paths

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingManager:
    def __init__(
        self,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        root = Path(get_folder_paths("embeddings")[0])
        self.path = str(root / "cached_embeddings")

        self.cache: Dict[str, torch.Tensor] = {}
        self.meta:  Dict[str, dict]         = {}
        self.dtype  = dtype
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.prompt_keys = []
        self.prompt_index = []
        self.prompt_matrix = None
        self.prompt_vectorizer = None

    # ---------- disk I/O -------------------------------------------------
    def load_dir(self, folder: str):
        for meta_path in Path(folder).glob("*.json"):
            md5 = meta_path.stem
            tensor_path = meta_path.with_suffix(".safetensors")
            if not tensor_path.exists():
                continue
            self.meta[md5]  = json.load(meta_path.open())
            self.cache[md5] = load_file(tensor_path)["conditioning"].cpu().to(self.dtype)
        if self.cache:
            print(f"[EmbeddingManager] loaded {len(self.cache)} bundles")

    def save_bundle(
            self,
            trigger: str,
            conditioning: List[Tuple[torch.Tensor, dict]],
            prompt_text: Optional[str] = None,
            folder: Optional[str] = None,
            meta_extra: Optional[dict] = None
    ) -> str:
        folder = folder or self.path
        Path(folder).mkdir(parents=True, exist_ok=True)

        # Construct tensor and extras dicts
        tensors = {}
        extras = {}
        for i, (tensor, extra) in enumerate(conditioning):
            key = f"conditioning_{i}"
            tensors[key] = tensor
            extras[key] = extra

        # Use first tensor as hash base
        base_tensor = tensors["conditioning_0"]
        raw = base_tensor.cpu().contiguous().view(-1)[:256_000].numpy().tobytes()
        md5 = hashlib.sha256(raw).hexdigest()

        # Save tensor data
        save_file(
            {k: v.cpu().to(self.dtype) for k, v in tensors.items()},
            str(Path(folder, md5 + ".safetensors"))
        )

        # Metadata
        meta = {
            "prompt_trigger": trigger,
            "tensor_keys": list(tensors.keys()),
            "dims": {k: list(v.shape) for k, v in tensors.items()},
            "created_at": datetime.utcnow().isoformat() + "Z",
            "prompt_text": prompt_text or "",
            "conditioning_extras": extras
        }

        if meta_extra:
            meta.update(meta_extra)

        # Save metadata
        Path(folder, md5 + ".json").write_text(json.dumps(meta, indent=2))

        self.cache[md5] = base_tensor.cpu().to(self.dtype)
        self.meta[md5] = meta
        return md5

    # ---------- tensor access -------------------------------------------
    def load_bundle(self, md5: str) -> Dict[str, torch.Tensor]:
        return load_file(Path(self.path, md5 + ".safetensors"))

    def get_tensor(self, md5: str, key: str = "conditioning") -> torch.Tensor:
        return self.load_bundle(md5)[key]

    # ---------- masked EMA update ---------------------------------------
    def update_tensor(self, md5: str, key: str, new: torch.Tensor,
                      mask=None, lr=0.2, ema=0.9):
        bundle = self.load_bundle(md5)
        old = bundle[key].to(new.device)

        if mask is None:
            mask = torch.ones_like(old[..., :1])

        grad = (F.normalize(old, dim=-1) - F.normalize(new, dim=-1)) * mask
        mid  = old - lr * grad
        nxt  = F.normalize(ema * old + (1 - ema) * mid, p=2, dim=-1)

        bundle[key] = nxt.cpu().to(self.dtype)
        save_file(bundle, str(Path(self.path, md5 + ".safetensors")))

        if key == "conditioning":
            self.cache[md5] = nxt.cpu().to(self.dtype)

        self.meta[md5]["updated_at"] = datetime.utcnow().isoformat() + "Z"
        return nxt

    def wrap_for_conditioning_pipeline(
        self,
        md5: str,
        key: str = "conditioning",
        dtype: Optional[torch.dtype] = None
    ):
        dtype = dtype or self.dtype
        core  = self.get_tensor(md5, key).to(dtype)
        pool  = self.get_tensor(md5, "pooled_output") if "pooled_output" in self.meta[md5]["tensor_keys"] else None
        return [core, {"pooled_output": pool}]

    # ---------- prompt similarity search --------------------------------
    def build_prompt_matrix(self):
        self.prompt_keys = []
        self.prompt_index = []
        for md5, meta in self.meta.items():
            prompt = meta.get("prompt_text", "").strip()
            if prompt:
                self.prompt_keys.append(md5)
                self.prompt_index.append(prompt)

        if not self.prompt_index:
            self.prompt_matrix = None
            return

        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6))
        self.prompt_matrix = vec.fit_transform(self.prompt_index)
        self.prompt_vectorizer = vec
        print(f"[EmbeddingManager] built prompt matrix with {len(self.prompt_index)} entries.")

    def lookup_prompt(self, query: str, top_k=5, thresh=0.25):
        if not self.prompt_matrix or not self.prompt_vectorizer:
            raise RuntimeError("Prompt matrix not built. Call `build_prompt_matrix()` first.")

        q_vec = self.prompt_vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.prompt_matrix).flatten()
        results = [(self.prompt_keys[i], sims[i]) for i in range(len(sims)) if sims[i] >= thresh]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]



# ---------- singleton accessor -----------------------------------------
import builtins as _bi


def _init_bank(path: str):
    bank = EmbeddingManager()
    if os.path.isdir(path):
        bank.load_dir(path)
    return bank

def get_bank(path: Optional[str] = None, *, force_reload=False) -> EmbeddingManager:
    if path is None:
        path = str(Path(get_folder_paths("embeddings")[0]) / "cached_embeddings")
    if force_reload or not hasattr(_bi, "_ABS_BANK"):
        _bi._ABS_BANK = _init_bank(path)
    return _bi._ABS_BANK
