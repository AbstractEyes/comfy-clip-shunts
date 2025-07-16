# embedding_manager.py
# =============================================================
import hashlib, json, os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import folder_paths

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

# -------------------------------------------------------------
class EmbeddingManager:
    """
    Conditioning-bank with:
      • .cache[md5] -> main “conditioning” tensor  (CPU fp16)
      • .meta[md5]  -> metadata dict
      • bag-of-tokens matrix for lightning-fast fuzzy lookup
    """

    def __init__(self,
                 dtype: torch.dtype = torch.float16,
                 device: str = "cuda",
                 vram_limit_gb: float = 4.0):
        self.path = folder_paths.get_directory_by_type("embeddings") + "/" + "cached_embeddings"
        self.cache: Dict[str, torch.Tensor] = {}
        self.meta:  Dict[str, dict]         = {}
        self.dtype  = dtype
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.vram_limit_gb = vram_limit_gb
        self.bit_matrix: Optional[torch.Tensor] = None
        self.token2col: Dict[int, int] = {}

    # ---------- helpers --------------------------------------------------
    def _id_from_tokens(self, tokens: Optional[List[int]],
                        tensor: torch.Tensor) -> str:
        if tokens:
            h = hashlib.md5()
            for t in tokens:
                h.update(t.to_bytes(4, "little"))
            return h.hexdigest()
        # fallback: hash first 256 kB of raw bytes
        raw = tensor.cpu().contiguous().view(-1)[:256_000].numpy().tobytes()
        return hashlib.sha256(raw).hexdigest()

    # ---------- disk I/O -------------------------------------------------
    def load_dir(self, folder: str):
        for meta_path in Path(folder).glob("*.json"):
            md5 = meta_path.stem
            safe_path = meta_path.with_suffix(".safetensors")
            if not safe_path.exists():
                continue
            self.meta[md5] = json.load(meta_path.open())
            self.cache[md5] = (
                load_file(safe_path)["conditioning"].cpu().to(self.dtype)
            )
        if self.cache:
            print(f"[EmbeddingManager] loaded {len(self.cache)} bundles")

    def save_bundle(self,
                    trigger: str,
                    tensors: Dict[str, torch.Tensor],   # {"conditioning": T, ...}
                    tokens: Optional[List[int]] = None,
                    folder: str = "embeddings",
                    meta_extra: Optional[dict] = None) -> str:
        md5 = self._id_from_tokens(tokens, next(iter(tensors.values())))
        Path(folder).mkdir(exist_ok=True, parents=True)
        safe_path = Path(folder, md5 + ".safetensors")
        save_file({k: v.cpu().to(self.dtype) for k, v in tensors.items()},
                  str(safe_path))

        meta = {
            "prompt_trigger": trigger,
            "tensor_keys": list(tensors.keys()),
            "token_ids": tokens or [],
            "dims": {k: list(v.shape) for k, v in tensors.items()},
            "created_at": datetime.utcnow().isoformat()+"Z",
        }
        if meta_extra:
            meta.update(meta_extra)
        Path(folder, md5 + ".json").write_text(json.dumps(meta, indent=2))

        self.cache[md5] = tensors["conditioning"].cpu().to(self.dtype)
        self.meta[md5]  = meta
        self.bit_matrix = None                          # matrix now stale
        return md5

    # ---------- bag-of-tokens matrix ------------------------------------
    def build_matrix(self):
        """Create/recreate [N,V] bool matrix with VRAM guard."""
        if not self.cache:
            return
        vocab = sorted({tid for m in self.meta.values() for tid in m["token_ids"]})
        self.token2col = {t: i for i, t in enumerate(vocab)}
        N, V = len(self.cache), len(vocab)
        req_gb = (N * V) / 8 / 1e9
        dev = self.device if (self.device.type == "cuda" and req_gb <= self.vram_limit_gb) else torch.device("cpu")
        M = torch.zeros(N, V, dtype=torch.bool, device=dev)
        for r, md5 in enumerate(self.meta):
            cols = [self.token2col[t] for t in self.meta[md5]["token_ids"]
                    if t in self.token2col]
            M[r, cols] = True
        self.bit_matrix = M
        print(f"[EmbeddingManager] bag matrix {N}×{V} on {dev} ({req_gb:.2f} GB)")

    def free_gpu(self):
        if self.bit_matrix is not None and self.bit_matrix.device.type == "cuda":
            self.bit_matrix = self.bit_matrix.cpu()
            torch.cuda.empty_cache()
            print("[EmbeddingManager] freed GPU VRAM")

    # ---------- lookup ---------------------------------------------------
    def _lookup_brute(self, tokens: List[int], top_k=5, thresh=0.8):
        tset = set(tokens)
        scored = []
        for k, meta in self.meta.items():
            j = len(tset & set(meta["token_ids"])) / len(tset | set(meta["token_ids"]) or {1})
            if j >= thresh:
                scored.append((k, j))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [k for k, _ in scored[:top_k]]

    def lookup(self, tokens: List[int], top_k=5, thresh=0.8):
        md5_exact = self._id_from_tokens(tokens, torch.empty(0))
        if md5_exact in self.cache:
            return [md5_exact]
        if self.bit_matrix is None:
            return self._lookup_brute(tokens, top_k, thresh)
        q = torch.zeros(self.bit_matrix.size(1), dtype=torch.bool, device=self.bit_matrix.device)
        for t in tokens:
            if t in self.token2col:
                q[self.token2col[t]] = True
        inter = (self.bit_matrix & q).sum(1).float()
        union = (self.bit_matrix | q).sum(1).float().clamp_min(1e-6)
        jacc  = inter / union
        vals, idx = torch.topk(jacc, k=min(top_k, len(jacc)))
        return [list(self.meta.keys())[i] for i, v in zip(idx.tolist(), vals.tolist()) if v >= thresh]

    # ---------- tensor access -------------------------------------------
    def load_bundle(self, md5: str) -> Dict[str, torch.Tensor]:
        return load_file(Path("embeddings", md5 + ".safetensors"))

    def get_tensor(self, md5: str, key: str = "conditioning") -> torch.Tensor:
        return self.load_bundle(md5)[key]

    # ---------- masked EMA update on a single tensor --------------------
    def update_tensor(self, md5: str, key: str, new: torch.Tensor,
                      mask=None, lr=0.2, ema=0.9):
        bundle = self.load_bundle(md5)
        old = bundle[key].to(new.device)
        if mask is None:
            mask = torch.ones_like(old[..., :1])
        grad = (F.normalize(old, dim=-1) - F.normalize(new, dim=-1)) * mask
        mid  = old - lr * grad
        nxt  = F.normalize(ema*old + (1-ema)*mid, p=2, dim=-1)
        bundle[key] = nxt.cpu().to(self.dtype)
        save_file(bundle, f"embeddings/{md5}.safetensors")
        if key == "conditioning":
            self.cache[md5] = nxt.cpu().to(self.dtype)
            self.bit_matrix = None
        self.meta[md5]["updated_at"] = datetime.utcnow().isoformat()+"Z"
        return nxt

    def wrap_for_conditioning_pipeline(self,
                                       md5: str,
                                       key: str = "conditioning",
                                       dtype: Optional[torch.dtype] = None) -> List:
        """
        Returns a tensor suitable for conditioning pipelines.
        If dtype is specified, it will convert the tensor to that dtype.
        """
        if dtype is None:
            dtype = self.dtype
        core = self.get_tensor(md5, key)
        pool = self.get_tensor(md5, "pooled_output") if "pooled_output" in self.meta[md5]["tensor_keys"] else None
        return [core, {"pooled_output": pool}]


# ---------- singleton accessor -----------------------------------------
import builtins as _bi

def _init_bank(path: str):
    bank = EmbeddingManager()
    if os.path.isdir(path):
        bank.load_dir(path)
        bank.build_matrix()
    return bank

def get_bank(path: str = None, *, force_reload=False) -> EmbeddingManager:
    if path is None:
        path = folder_paths.get_directory_by_type("embeddings") + "/" + "cached_embeddings"
    if force_reload or not hasattr(_bi, "_ABS_BANK"):
        _bi._ABS_BANK: EmbeddingManager = _init_bank(path)
    return _bi._ABS_BANK
