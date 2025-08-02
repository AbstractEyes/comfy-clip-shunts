# embedding_manager.py
# =============================================================
"""
ABS Embedding Manager - Fixed for Windows file locking issues
"""
import hashlib, json, os
import threading
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Union

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from folder_paths import get_folder_paths

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..utils.conditioning_helper import ConditioningHelper, UsefulConditioning


class SingletonMeta(type):
    """Thread-safe singleton metaclass implementation."""
    _instances = {}
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]


class EmbeddingManager(metaclass=SingletonMeta):
    """
    Thread-safe singleton embedding manager with Windows file handling fixes.
    """
    _initialized = False
    _init_lock = threading.Lock()

    def __init__(
            self,
            dtype: torch.dtype = torch.float16,
            device: str = "cuda",
    ):
        with self._init_lock:
            if self._initialized:
                return

            root = Path(get_folder_paths("embeddings")[0])
            self.path = str(root / "cached_embeddings")

            self.cache: Dict[str, UsefulConditioning] = {}
            self.meta: Dict[str, dict] = {}
            self.dtype = dtype
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")

            self.prompt_keys = []
            self.prompt_index = []
            self.prompt_matrix = None
            self.prompt_vectorizer = None

            # Locks for thread safety
            self._cache_lock = threading.RLock()
            self._matrix_lock = threading.Lock()
            self._file_locks: Dict[str, threading.Lock] = {}
            self._file_lock_manager = threading.Lock()

            # Track loaded tensors to avoid memory-mapped file issues
            self._loaded_tensors: Dict[str, Dict[str, torch.Tensor]] = {}

            self._initialized = True

    def _get_file_lock(self, filepath: str) -> threading.Lock:
        """Get or create a lock for a specific file."""
        with self._file_lock_manager:
            if filepath not in self._file_locks:
                self._file_locks[filepath] = threading.Lock()
            return self._file_locks[filepath]

    @staticmethod
    def _sanitize(obj):
        """Convert anything that JSON can't handle into a lightweight stub."""
        import torch
        if isinstance(obj, torch.Tensor):
            return f"<tensor:{tuple(obj.shape)}>"
        if isinstance(obj, torch.device):
            return str(obj)
        if isinstance(obj, dict):
            return {k: EmbeddingManager._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [EmbeddingManager._sanitize(x) for x in obj]
        return obj

    def _safe_save_file(self, tensors: Dict[str, torch.Tensor], filepath: str):
        """
        Save tensors to file safely, handling Windows file locking issues.
        Uses atomic write with temporary file.
        """
        # Clear any cached tensors for this file to release memory maps
        bundle_id = Path(filepath).stem
        if bundle_id in self._loaded_tensors:
            del self._loaded_tensors[bundle_id]

        # Create temporary file in same directory for atomic move
        temp_fd, temp_path = tempfile.mkstemp(
            dir=os.path.dirname(filepath),
            prefix='.tmp_',
            suffix='.safetensors'
        )

        try:
            os.close(temp_fd)  # Close the file descriptor

            # Save to temporary file
            save_file(tensors, temp_path)

            # Atomic move (on Windows, this might fail if target exists)
            if os.path.exists(filepath):
                # On Windows, we need to remove the target first
                try:
                    os.remove(filepath)
                except OSError:
                    # If removal fails, try backup approach
                    backup_path = f"{filepath}.backup"
                    if os.path.exists(backup_path):
                        os.remove(backup_path)
                    os.rename(filepath, backup_path)

            # Now move temp file to target
            shutil.move(temp_path, filepath)

        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            raise e

    def _safe_load_file(self, filepath: str) -> Dict[str, torch.Tensor]:
        """
        Load tensors from file safely, caching to avoid repeated memory mapping.
        """
        bundle_id = Path(filepath).stem

        # Check if already loaded
        if bundle_id in self._loaded_tensors:
            # Return copies to avoid modifications affecting cache
            return {k: v.clone().detach().contiguous() for k, v in self._loaded_tensors[bundle_id].items()}

        # Load and cache
        tensors = load_file(filepath)

        # Store CPU copies to avoid device issues
        cpu_tensors = {k: v.cpu().clone().detach().contiguous() for k, v in tensors.items()}
        self._loaded_tensors[bundle_id] = cpu_tensors

        # Return copies
        return {k: v.clone().detach().contiguous() for k, v in cpu_tensors.items()}

    def load_dir(self, folder: str):
        """Thread-safe directory loading."""
        with self._cache_lock:
            self.cache.clear()
            self.meta.clear()
            self._loaded_tensors.clear()

        for meta_path in Path(folder).glob("*.json"):
            sha_id = meta_path.stem
            try:
                self.load_bundle(sha_id, folder_override=folder, _priming=True)
            except Exception as e:
                print(f"[EmbeddingManager] ⇢ skip {sha_id[:8]}  ({e})")

        if self.cache:
            print(f"[EmbeddingManager] loaded {len(self.cache)} bundles.")
            self.build_prompt_matrix()

    def save_bundle(
            self,
            trigger: str,
            conditioning: Union[List[Tuple[torch.Tensor, dict]], UsefulConditioning],
            *,
            prompt_text: Optional[str] = None,
            folder: Optional[str] = None,
            meta_extra: Optional[dict] = None,
    ) -> str:
        """Thread-safe bundle saving with Windows file handling."""
        # Normalize input
        if not isinstance(conditioning, UsefulConditioning):
            if not ConditioningHelper.verify(conditioning, silent=True):
                raise ValueError("Conditioning input is invalid.")
            conditioning = ConditioningHelper.convert_conditioning(conditioning)
        conditioning = conditioning.clone(device="cpu")
        folder = Path(folder or self.path)
        folder.mkdir(parents=True, exist_ok=True)

        # Collect tensors and metadata
        tensors: Dict[str, torch.Tensor] = {}
        extras: Dict[str, dict] = {}
        pooled_collect: List[torch.Tensor] = []

        for idx in range(len(conditioning)):
            core_key = f"conditioning_{idx}"
            pooled_key = f"{core_key}_pooled"

            core_tensor = conditioning.get_tensor(idx).clone().detach().to(self.dtype).cpu().contiguous()
            meta = conditioning.get_all_metadata()[idx]

            pooled_tensor = conditioning.get_pooled(idx)
            if pooled_tensor is not None:
                pooled_tensor = pooled_tensor.clone().detach().to(self.dtype).cpu().contiguous()
                tensors[pooled_key] = pooled_tensor
                pooled_collect.append(pooled_tensor)

            tensors[core_key] = core_tensor
            extras[core_key] = self._sanitize(meta)

        if pooled_collect:
            if len(pooled_collect) == 1:
                # Clone to avoid shared memory reference
                tensors["pooled_output"] = pooled_collect[0].clone()
            else:
                try:
                    tensors["pooled_output"] = torch.cat(pooled_collect, dim=1)
                except Exception:
                    # Clone in fallback case too
                    tensors["pooled_output"] = pooled_collect[0].clone()

        # Generate bundle ID
        ref_bytes = tensors[next(iter(tensors))].contiguous().view(-1)[:256_000].numpy().tobytes()
        sha256_id = hashlib.sha256(ref_bytes).hexdigest()

        # File paths
        tensor_path = str(folder / f"{sha256_id}.safetensors")
        meta_path = str(folder / f"{sha256_id}.json")

        # Get file-specific locks
        tensor_lock = self._get_file_lock(tensor_path)
        meta_lock = self._get_file_lock(meta_path)


        # Write tensor file with safe method
        with tensor_lock:
            self._safe_save_file(tensors, tensor_path)

        # Prepare metadata
        meta = {
            "prompt_trigger": trigger,
            "tensor_keys": list(tensors.keys()),
            "dims": {k: list(v.shape) for k, v in tensors.items()},
            "dtype": str(self.dtype),
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "prompt_text": prompt_text or "",
            "conditioning_extras": extras,
        }
        if meta_extra:
            meta.update(meta_extra)

        # Write metadata with atomic write
        with meta_lock:
            temp_meta = f"{meta_path}.tmp"
            Path(temp_meta).write_text(json.dumps(meta, indent=2))
            if os.path.exists(meta_path):
                os.remove(meta_path)
            os.rename(temp_meta, meta_path)

        # Update cache
        with self._cache_lock:
            self.cache[sha256_id] = UsefulConditioning(conditioning.clone(device="cpu"))
            self.meta[sha256_id] = meta

        self.build_prompt_matrix()
        return sha256_id

    def load_bundle(
            self,
            ident: str,
            *,
            folder_override: Optional[str] = None,
            _priming: bool = False,
    ) -> UsefulConditioning:
        """Thread-safe bundle loading with Windows file handling."""
        sha_id = self._resolve_md5(ident) if not _priming else ident

        # Check cache first
        with self._cache_lock:
            if sha_id in self.cache:
                return self.cache[sha_id]

        base_dir = Path(folder_override or self.path)
        meta_path = base_dir / f"{sha_id}.json"
        tensor_path = base_dir / f"{sha_id}.safetensors"

        if not meta_path.exists() or not tensor_path.exists():
            raise FileNotFoundError(f"Missing bundle files for {sha_id}")

        # Thread-safe file reading
        meta_lock = self._get_file_lock(str(meta_path))
        tensor_lock = self._get_file_lock(str(tensor_path))

        with meta_lock:
            meta = json.loads(meta_path.read_text())

        with tensor_lock:
            raw_tensors = self._safe_load_file(str(tensor_path))

        # Reconstruct UsefulConditioning
        extras = meta.get("conditioning_extras", {})
        out_entries: List[List[Union[torch.Tensor, dict]]] = []

        core_keys = [k for k in meta["tensor_keys"] if k.startswith("conditioning_") and not k.endswith("_pooled")]
        core_keys.sort(key=lambda k: int(k.split("_")[1]))

        for core_key in core_keys:
            pooled_key = f"{core_key}_pooled"
            core_tensor = raw_tensors[core_key].cpu().to(self.dtype)

            meta_dict = extras.get(core_key, {})
            if pooled_key in raw_tensors:
                meta_dict = dict(meta_dict)
                meta_dict["pooled_output"] = raw_tensors[pooled_key].cpu().to(self.dtype)

            out_entries.append([core_tensor, meta_dict])

        if "pooled_output" in raw_tensors and out_entries:
            combined = raw_tensors["pooled_output"].cpu().to(self.dtype)
            for entry in out_entries:
                if "pooled_output" not in entry[1]:
                    entry[1]["pooled_output"] = combined

        useful = UsefulConditioning(out_entries)

        # Update cache
        with self._cache_lock:
            self.cache[sha_id] = useful
            self.meta[sha_id] = meta

        if not _priming:
            self.build_prompt_matrix()

        return useful

    def _resolve_md5(self, ident: str) -> str:
        """Thread-safe bundle resolution."""
        with self._cache_lock:
            if ident in self.meta:
                return ident
            for md5, meta in self.meta.items():
                if ident == meta.get("prompt_trigger") or md5.startswith(ident):
                    return md5
        raise KeyError(f"[EmbeddingManager] bundle '{ident}' not found")


    # Thread-safe prompt matrix building
    def build_prompt_matrix(self):
        """Thread-safe prompt matrix building."""
        with self._matrix_lock:
            with self._cache_lock:
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
        """Thread-safe prompt lookup."""
        with self._matrix_lock:
            if not self.prompt_matrix or not self.prompt_vectorizer:
                raise RuntimeError("Prompt matrix not built. Call `build_prompt_matrix()` first.")

            q_vec = self.prompt_vectorizer.transform([query])
            sims = cosine_similarity(q_vec, self.prompt_matrix).flatten()

        results = [(self.prompt_keys[i], sims[i]) for i in range(len(sims)) if sims[i] >= thresh]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    # Add other methods with appropriate locking...




# Singleton accessor functions
_manager_instance = None
_manager_lock = threading.Lock()


def clear_cache(self):
    """Clear all caches including loaded tensors."""
    with self._cache_lock:
        self.cache.clear()
        self.meta.clear()
        self._loaded_tensors.clear()


# Singleton accessor
def get_bank(path: Optional[str] = None, *, force_reload: bool = False) -> EmbeddingManager:
    """Thread-safe singleton accessor for EmbeddingManager."""
    if path is None:
        path = str(Path(get_folder_paths("embeddings")[0]) / "cached_embeddings")

    manager = EmbeddingManager()

    if force_reload or not manager.cache:
        if os.path.isdir(path):
            manager.load_dir(path)

    return manager