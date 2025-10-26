# embedding_manager.py
# ============================================================
from __future__ import annotations

import json
import os
import threading
import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
from safetensors.torch import save_file as st_save_file, load_file as st_load_file
from folder_paths import get_folder_paths

from ..utils.conditioning_helper import ConditioningHelper, UsefulConditioning


# ============================================================
# Constants / paths
# ------------------------------------------------------------
_EMBED_ROOT = Path(get_folder_paths("embeddings")[0])

SCHEMA_VERSION = 1
CORE_KEY_FMT = "conditioning_{i}"
POOLED_SUFFIX = "_pooled"
COMBINED_POOLED_KEY = "pooled_output"


# ============================================================
# Helpers
# ------------------------------------------------------------
def _cpu_clone(t: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Distinct CPU storage for safetensors: detach → cpu → dtype → contiguous → clone."""
    if dtype is not None:
        return t.detach().to("cpu", dtype=dtype).contiguous().clone()
    return t.detach().to("cpu").contiguous().clone()


def _json_safe_meta(meta: dict) -> dict:
    """Remove tensors / non-serializables; keep plain Python types only."""
    out = {}
    for k, v in meta.items():
        if torch.is_tensor(v):
            continue
        out[k] = v
    return out


class EmbeddingManager:
    """
    Disk-backed store for conditioning bundles:
      • tensors in <bundle_id>.safetensors
      • sidecar metadata in <bundle_id>.json
    Correctness guarantees:
      - Bundle ID is derived from (tensor structure + trigger + prompt) → no overwrites
      - All tensors are CPU-cloned for safetensors (Windows-safe; no shared storage)
      - Loads return list[[tensor, meta], ...] where pooled is meta['pooled_output']
      - In-memory indices/caches update immediately after save
    """

    # ----- schema & keying -----
    SCHEMA_VERSION: int = 1
    CORE_KEY_FMT: str = "conditioning_{i}"
    POOLED_SUFFIX: str = "_pooled"
    COMBINED_POOLED_KEY: str = "pooled_output"

    def __init__(
        self,
        dtype: torch.dtype = torch.float16,
        device: str = "cpu",
        subdir: str = "cached_embeddings",
    ):
        self.dtype: torch.dtype = dtype
        self.device: torch.device = torch.device(device)

        # paths
        root = Path(get_folder_paths("embeddings")[0])
        self.paths = type("Paths", (), {})()
        self.paths.root = root
        self.paths.dir = root / subdir
        self.paths.dir.mkdir(parents=True, exist_ok=True)

        # indices/caches
        self.meta: Dict[str, dict] = {}                               # bundle_id -> sidecar
        self.cache: Dict[str, List[List[Union[torch.Tensor, dict]]]] = {}  # bundle_id -> list[[tensor, meta], ...]
        self._loaded_tensors: Dict[str, Dict[str, torch.Tensor]] = {} # bundle_id -> {key: tensor}
        self._trigger_to_id: Dict[str, List[str]] = {}                # trigger -> [bundle_id,...]

        # derived/search artifacts (if you use them)
        self._prompt_matrix: Optional[torch.Tensor] = None
        self._bundle_ids_for_matrix: List[str] = []
        self._vectorizer: Any = None

        self._cache_lock = threading.RLock()
        self._prime_from_disk()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_dir(self, path: Union[str, Path]) -> None:
        """Switch working directory and re-index."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        with self._cache_lock:
            self.paths.dir = p
        self._prime_from_disk()

    def reload(self, dir_path: Optional[Union[str, Path]] = None) -> None:
        """Re-index current (or new) directory."""
        if dir_path:
            self.load_dir(dir_path)
        else:
            self._prime_from_disk()

    def clear_cache(self) -> None:
        """Drop RAM caches (sidecars remain)."""
        with self._cache_lock:
            self.cache.clear()
            self._loaded_tensors.clear()
            self._prompt_matrix = None
            self._bundle_ids_for_matrix = []
            self._vectorizer = None

    def list_bundles(self) -> List[str]:
        """Return bundle_ids newest-first by sidecar created_at."""
        with self._cache_lock:
            items = list(self.meta.items())
        def _k(it):
            return it[1].get("created_at", "")
        return [bid for bid, _ in sorted(items, key=_k, reverse=True)]

    def info(self, ident: str) -> dict:
        """Return a copy of the sidecar for id/trigger/prefix."""
        bid = self._resolve_ident(ident)
        with self._cache_lock:
            return dict(self.meta[bid])

    def save_bundle(
        self,
        trigger: str,
        conditioning,
        *,
        prompt_text: str = "",
        folder: Optional[Union[str, Path]] = None,
        meta_extra: Optional[dict] = None,
        notes: Optional[str] = None,
    ) -> str:
        """
        Save a bundle atomically. ID = sha256(structure + trigger + prompt).
        Returns: bundle_id (hex).
        """
        # Normalize to your canonical format
        try:
            uc = conditioning if isinstance(conditioning, UsefulConditioning) \
                 else ConditioningHelper.convert_conditioning(conditioning)
        except Exception as e:
            raise TypeError(f"[EmbeddingManager.save_bundle] invalid conditioning: {e}")
        if len(uc) == 0:
            raise ValueError("Empty conditioning")

        tensors: Dict[str, torch.Tensor] = {}
        extras: List[dict] = []
        pooled_entries: List[torch.Tensor] = []

        # Build tensors/extras payload
        for i in range(len(uc)):
            core = uc.get_tensor(i)
            meta = dict(uc.get_all_metadata()[i])

            core_key = self.CORE_KEY_FMT.format(i=i)
            core_cpu = core.detach().to("cpu", dtype=self.dtype).contiguous().clone()
            tensors[core_key] = core_cpu

            pooled = meta.get("pooled_output", None)
            if isinstance(pooled, torch.Tensor):
                pooled_key = f"{core_key}{self.POOLED_SUFFIX}"
                pooled_cpu = pooled.detach().to("cpu", dtype=self.dtype).contiguous().clone()
                tensors[pooled_key] = pooled_cpu
                pooled_entries.append(pooled_cpu)

            # strip tensors from meta → JSON-safe
            clean_meta = {k: (None if torch.is_tensor(v) else v)
                          for k, v in meta.items() if k != "pooled_output"}
            extras.append(clean_meta)

        # Combined pooled (clone again → no shared storage with per-entry pooled)
        if pooled_entries:
            tensors[self.COMBINED_POOLED_KEY] = pooled_entries[0].clone().contiguous()

        # Deterministic id: structure + trigger + prompt
        h = hashlib.sha256()
        for k in sorted(tensors.keys()):
            t = tensors[k]
            h.update(str((k, tuple(t.shape), str(t.dtype))).encode("utf-8"))
        h.update((trigger or "").encode("utf-8"))
        h.update((prompt_text or "").encode("utf-8"))
        bundle_id = h.hexdigest()

        # Resolve target paths; guard against accidental on-disk collision
        base = Path(folder or self.paths.dir)
        base.mkdir(parents=True, exist_ok=True)
        tpath = base / f"{bundle_id}.safetensors"
        jpath = base / f"{bundle_id}.json"
        if tpath.exists() or jpath.exists():
            salt = uuid.uuid4().hex[:8]
            h.update(salt.encode("utf-8"))
            bundle_id = h.hexdigest()
            tpath = base / f"{bundle_id}.safetensors"
            jpath = base / f"{bundle_id}.json"

        # Atomic writes
        self._atomic_safetensors_write(tpath, tensors)

        sidecar = {
            "schema_version": self.SCHEMA_VERSION,
            "bundle_id": bundle_id,
            "created_at": datetime.utcnow().isoformat(timespec="seconds"),
            "prompt_trigger": trigger or "",
            "prompt_text": prompt_text or "",
            "tensor_keys": list(tensors.keys()),
            "dtype": str(self.dtype),
            "device": str(self.device),
            "conditioning_extras": extras,   # index-aligned to conditioning_{i}
            "notes": notes or "",
        }
        if meta_extra:
            # keep user extras separate to avoid key collisions
            sidecar["user_meta"] = {k: v for k, v in meta_extra.items()
                                    if not torch.is_tensor(v)}

        self._atomic_json_write(jpath, sidecar)

        # Update in-memory indices/caches
        with self._cache_lock:
            self.meta[bundle_id] = sidecar
            self.cache.pop(bundle_id, None)
            self._loaded_tensors.pop(bundle_id, None)

        self._rebuild_trigger_index()
        self.build_prompt_matrix(force=True)
        return bundle_id

    def load_by_id(
        self,
        ident: str,
        to_device: Optional[Union[str, torch.device]] = None,
        to_dtype: Optional[torch.dtype] = None,
    ) -> List[List[Union[torch.Tensor, dict]]]:
        """
        Load a bundle and return list[[tensor, meta], ...] with pooled in meta.
        """
        bid = self._resolve_ident(ident)
        return self._load_core(bid, to_device=to_device, to_dtype=to_dtype)

    def load_by_trigger(
        self,
        trigger: str,
        to_device: Optional[Union[str, torch.device]] = None,
        to_dtype: Optional[torch.dtype] = None,
    ) -> List[List[Union[torch.Tensor, dict]]]:
        bids = self._trigger_to_id.get(trigger) or []
        if not bids:
            raise KeyError(f"No bundle for trigger '{trigger}'")
        return self.load_by_id(bids[0], to_device=to_device, to_dtype=to_dtype)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _prime_from_disk(self) -> None:
        """Scan sidecars into self.meta (tensors are lazy)."""
        with self._cache_lock:
            self.meta.clear()
            self.cache.clear()
            self._loaded_tensors.clear()
            self._trigger_to_id.clear()
            self._prompt_matrix = None
            self._bundle_ids_for_matrix = []
            self._vectorizer = None

            for p in self.paths.dir.glob("*.json"):
                try:
                    side = json.loads(p.read_text(encoding="utf-8"))
                    if int(side.get("schema_version", 0)) != self.SCHEMA_VERSION:
                        continue
                    bid = side.get("bundle_id") or p.stem
                    self.meta[bid] = side
                except Exception:
                    continue

        self._rebuild_trigger_index()

    def _rebuild_trigger_index(self) -> None:
        with self._cache_lock:
            self._trigger_to_id.clear()
            for bid, side in self.meta.items():
                trig = side.get("prompt_trigger") or ""
                if trig:
                    self._trigger_to_id.setdefault(trig, []).append(bid)

    def _resolve_ident(self, ident: str) -> str:
        """Support exact id, exact trigger, or id prefix."""
        with self._cache_lock:
            if ident in self.meta:
                return ident
            if ident in self._trigger_to_id and self._trigger_to_id[ident]:
                return self._trigger_to_id[ident][0]
            for bid in self.meta.keys():
                if bid.startswith(ident):
                    return bid
        raise KeyError(f"bundle '{ident}' not found")

    def _lazy_load_tensors(self, bundle_id: str) -> Dict[str, torch.Tensor]:
        with self._cache_lock:
            if bundle_id in self._loaded_tensors:
                return self._loaded_tensors[bundle_id]
        tpath = self.paths.dir / f"{bundle_id}.safetensors"
        if not tpath.exists():
            raise FileNotFoundError(f"Missing tensor file for {bundle_id}")
        td = st_load_file(str(tpath))  # CPU tensors
        with self._cache_lock:
            self._loaded_tensors[bundle_id] = td
        return td

    def _load_core(
        self,
        bundle_id: str,
        to_device: Optional[Union[str, torch.device]] = None,
        to_dtype: Optional[torch.dtype] = None,
    ) -> List[List[Union[torch.Tensor, dict]]]:
        """
        Materialize list[[tensor, meta], ...]; pooled restored into meta['pooled_output'].
        """
        with self._cache_lock:
            side = self.meta.get(bundle_id)
        if not side:
            raise FileNotFoundError(f"Missing sidecar for {bundle_id}")

        tensors = self._lazy_load_tensors(bundle_id)
        device = torch.device(to_device) if to_device is not None else self.device
        dtype = to_dtype if to_dtype is not None else self.dtype

        # Keep original order by conditioning_{i}
        keys = [k for k in side.get("tensor_keys", [])
                if k.startswith("conditioning_") and not k.endswith(self.POOLED_SUFFIX)]
        def idx(k: str) -> int:
            try: return int(k.split("_")[1])
            except Exception: return 0
        keys.sort(key=idx)

        extras_list: List[dict] = side.get("conditioning_extras", [])
        out: List[List[Union[torch.Tensor, dict]]] = []

        for i, core_key in enumerate(keys):
            if core_key not in tensors:
                continue
            core = tensors[core_key].to(device=device, dtype=dtype, non_blocking=True).contiguous()
            meta: dict = {}

            pooled_key = f"{core_key}{self.POOLED_SUFFIX}"
            if pooled_key in tensors:
                meta["pooled_output"] = tensors[pooled_key].to(device=device, dtype=dtype, non_blocking=True).contiguous()

            if i < len(extras_list) and isinstance(extras_list[i], dict):
                for k, v in extras_list[i].items():
                    if not torch.is_tensor(v):
                        meta[k] = v

            out.append([core, meta])

        # Backfill pooled from combined if per-entry pooled missing
        if self.COMBINED_POOLED_KEY in tensors and out:
            combined = tensors[self.COMBINED_POOLED_KEY].to(device=device, dtype=dtype, non_blocking=True).contiguous()
            for _, m in out:
                m.setdefault("pooled_output", combined)

        with self._cache_lock:
            self.cache[bundle_id] = out
        return out

    # ------------------------------------------------------------------
    # Atomic writers
    # ------------------------------------------------------------------
    def _atomic_safetensors_write(self, path: Union[str, Path], tensors: Dict[str, torch.Tensor]) -> None:
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        st_save_file(tensors, str(tmp))
        os.replace(str(tmp), str(path))

    def _atomic_json_write(self, path: Union[str, Path], payload: dict) -> None:
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(str(tmp), str(path))

    # ------------------------------------------------------------------
    # Optional search/index (no-op unless you use it)
    # ------------------------------------------------------------------
    def build_prompt_matrix(self, force: bool = False) -> None:
        if not force and self._prompt_matrix is not None:
            return
        self._prompt_matrix = None
        self._bundle_ids_for_matrix = []
        self._vectorizer = None


# ============================================================
# Singleton factory
# ------------------------------------------------------------
_MANAGER_LOCK = threading.Lock()
_MANAGER_SINGLETON: Optional[EmbeddingManager] = None

def get_bank(
    force_reload: bool = False,
    dtype: torch.dtype = torch.float16,
    device: str = "cpu",
    subdir: str = "cached_embeddings",
) -> EmbeddingManager:
    """
    Retrieve the process-local EmbeddingManager singleton.
    • force_reload=True → re-index sidecars and clear derived caches
    • dtype/device/subdir are applied on first creation; later calls can update device/dtype
    """
    global _MANAGER_SINGLETON
    with _MANAGER_LOCK:
        if _MANAGER_SINGLETON is None:
            _MANAGER_SINGLETON = EmbeddingManager(dtype=dtype, device=device, subdir=subdir)
        else:
            # apply dynamic config changes
            changed = False
            if str(_MANAGER_SINGLETON.device) != str(device):
                _MANAGER_SINGLETON.device = torch.device(device)
                changed = True
            if _MANAGER_SINGLETON.dtype != dtype:
                _MANAGER_SINGLETON.dtype = dtype
                changed = True
            expected_dir = _EMBED_ROOT / subdir
            if _MANAGER_SINGLETON.paths.dir != expected_dir:
                _MANAGER_SINGLETON.load_dir(expected_dir)
                changed = True
            if force_reload and not changed:
                _MANAGER_SINGLETON.reload()
        return _MANAGER_SINGLETON
