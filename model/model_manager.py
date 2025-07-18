from typing import Dict, Optional, Any, Union, Tuple
import os
import torch
import torch.nn as nn
import logging
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from safetensors.torch import load_file
from torch.nn import Module
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, BertModel, BertTokenizer, \
    PreTrainedTokenizerFast, T5TokenizerFast, T5EncoderModel

from .custom.t5_encoder_with_projection import T5EncoderWithProjection

logger = logging.getLogger(__name__)
# --------------------------------------------------------------------------- #
# Helper for namespaced cache keys
def _make_key(model_type: str, model_id: str) -> str:
    """
    Produce a unique key for the internal cache.

    Example
    -------
    >>> _make_key("bert", "bert-base")
    'bert:bert-base'
    """
    return f"{model_type}:{model_id}"


# Thread-safe registry wrapper
class _SafeDict(dict):
    """A dict protected by a re-entrant lock for thread-safe writes."""
    def __init__(self):
        super().__init__()
        import threading
        self._lock = threading.RLock()

    def safe_set(self, key, value):
        with self._lock:
            super().__setitem__(key, value)

    def safe_get(self, key, default=None):
        with self._lock:
            return super().get(key, default)

    def safe_del(self, key):
        with self._lock:
            if key in self:
                super().__delitem__(key)
                return True
            return False


# -------------------------------------------------------------------------------------------------------------------- #
# WARNING: ENABLING THIS TRUST_REMOTE_CODE FLAG WILL ALLOW EXECUTION OF ARBITRARY CODE FROM THE MODEL REPOSITORY.
# USE WITH EXTREME CAUTION, AS IT CAN POTENTIALLY EXECUTE MALICIOUS CODE FROM UNTRUSTED SOURCES.

TRUST_REMOTE_CODE = False  # Set to True only if you trust the source of the models you are loading.

# I advise leaving this OFF for any production or sensitive environments, and for any government or enterprise use.
# Ensure you fully trust the model repository and its maintainers and reviewing the code thoroughly.
# You cannot ONLY trust an AI's response to the question of whether it is safe to enable this flag,
#   as it may not have the full context of security implications or the specific model's behavior.
# -------------------------------------------------------------------------------------------------------------------- #
# COMFYUI operates within a form of sandbox, but enabling remote code execution can still pose many unseen risks.
# -------------------------------------------------------------------------------------------------------------------- #


class ModelType(Enum):
    """Enum for different model types"""
    SHUNT_ADAPTER = "shunt_adapter"
    T5_MODEL = "t5_model"
    BERT_MODEL = "bert"
    NOMIC_BERT_MODEL = "nomic_bert"
    GENERIC = "generic"
    TOKENIZER = "tokenizer"

@dataclass
class ModelInfo:
    """Container for model information"""
    model: nn.Module
    model_type: ModelType
    config: Dict[str, Any]
    device: torch.device
    dtype: torch.dtype
    metadata: Dict[str, Any] = None
    trust_remote_code: bool = TRUST_REMOTE_CODE  # Use global setting by default


class ModelManager:
    """
    Centralized model loader / cache with thread-safety and namespaced keys.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        # Thread-safe model cache
        self.models: _SafeDict = _SafeDict()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = self._setup_cache_dir(cache_dir)

    # be VERY careful with huggingface keys, remote code execution, and model downloads.
    # If you are using private models or need to authenticate, set the HuggingFace API key.
    def set_huggingface_key(self, key: str):
        """
        Set the HuggingFace API key for model downloads.
        This is useful if you have a private model or need to authenticate.
        """
        os.environ["HF_TOKEN"] = key
        logger.info("HuggingFace API key set successfully.")

    def get_huggingface_key(self) -> Optional[str]:
        """
        Get the HuggingFace API key if set.
        This is useful for debugging or checking if authentication is needed.
        """
        return os.environ.get("HF_TOKEN")

    def set_huggingface_cache_directory(self, directory: str):
        """
        Set the cache directory for HuggingFace model downloads.
        This is useful if you want to change the cache location.
        This will not move your models, it only sets the new default directory.
        """
        os.environ["HF_HOME"] = directory
        logger.info(f"HuggingFace default directory set to: {directory}")

    def get_huggingface_cache_directory(self) -> Optional[str]:
        """
        Get the cache directory for HuggingFace model downloads.
        This is useful for debugging or checking where models are stored.
        """
        return os.environ.get("HF_HOME", str(self.cache_dir))

    # --------------------------------------------------------------------- #
    # Internal helpers
    def _store(self, key: str, info: "ModelInfo") -> None:
        """Thread-safe insertion into the model cache."""
        self.models.safe_set(key, info)


    def _setup_cache_dir(self, cache_dir: Optional[str]) -> Path:
        """Setup and validate cache directory"""
        if cache_dir:
            cache_path = Path(cache_dir)
        else:
            # Use default HuggingFace cache location
            cache_path = Path.home() / ".cache" / "huggingface" / "transformers"

        cache_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using cache directory: {cache_path}")
        return cache_path

    def get_model(self, key: str) -> Optional["ModelInfo"]:
        """Retrieve a model by its namespaced key."""
        return self.models.safe_get(key)

    def is_loaded(self, key: str) -> bool:
        """Return True if the namespaced key is present in the cache."""
        return self.models.safe_get(key) is not None


    def move_model(
        self,
        namespaced_key: str,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Optional[nn.Module]:
        """
        Convert device/dtype of a cached model and return the updated object.
        """
        model = self._maybe_convert_dtype(namespaced_key, dtype, device)
        if model is None:
            logger.warning("move_model: %s not found", namespaced_key)
        return model


    def load_tokenizer(
        self,
        id: str,
        tokenizer_name_or_path: str,
        target_output_device: Optional[torch.device] = None,
        force_reload: bool = False,
        trust_remote_code: Optional[bool] = None,
    ) -> Optional[tuple[PreTrainedTokenizerFast, dict[str, Any]]]:
        """Load or fetch from cache a Hugging-Face tokenizer."""
        key = _make_key("tokenizer", id)
        if not force_reload and self.is_loaded(key):
            model_info = self.get_model(key)
            return model_info.model, model_info.metadata

        try:
            trust_remote_code = (
                trust_remote_code if trust_remote_code is not None else TRUST_REMOTE_CODE
            )
            tok = AutoTokenizer.from_pretrained(
                tokenizer_name_or_path, trust_remote_code=trust_remote_code
            )

            self._store(
                key,
                ModelInfo(
                    model=tok,
                    model_type=ModelType.TOKENIZER,
                    config={"tokenizer_name": tokenizer_name_or_path},
                    device=target_output_device or torch.device("cpu"),
                    dtype=torch.float32,
                    metadata={"source": "huggingface", "trust_remote_code": trust_remote_code},
                ),
            )
            logger.info("Loaded tokenizer %s", key)
            return tok, self.get_model(key).metadata

        except Exception:
            logger.exception("Failed to load tokenizer %s", id)
            return None


    def load_shunt_adapter(
            self,
            adapter_id: str,
            config: Dict[str, Any],
            path: Optional[str] = None,
            repo_id: Optional[str] = None,
            filename: Optional[str] = None,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            force_reload: bool = False
    ) -> Optional[nn.Module]:
        """
        Load a shunt adapter from local path or HuggingFace.

        Args:
            adapter_id: Unique identifier for the adapter
            config: Configuration dictionary for the adapter
            path: Local path to the adapter file
            repo_id: HuggingFace repository ID
            filename: Filename in the HuggingFace repository
            device: Target device
            dtype: Target dtype
            force_reload: Force reload even if cached

        Returns:
            Loaded adapter model or None if failed
        """
        if not force_reload and self.is_loaded(adapter_id):
            logger.info(f"Using cached adapter: {adapter_id}")
            return self._maybe_convert_dtype(adapter_id, dtype, device)
        try:
            # Import here to avoid circular imports
            from .dual_stream_adapter_model import ConditionModulationShuntAdapter

            # Determine file location
            file_path = self._resolve_file_path(path, repo_id, filename)
            if not file_path:
                raise FileNotFoundError(f"Could not find adapter file for {adapter_id}")
            # Initialize adapter
            # if the filename ends with t5-vit-l-14-dual_shunt_booru_13_000_000.safetensors we set attention heads to 4, else we set to 12
            logger.info(f"Loading adapter {adapter_id} from {file_path}")
            adapter = ConditionModulationShuntAdapter(config=config)
            logger.info(f"Initialized adapter {adapter_id} with config: {config}")
            # Load weights
            state_dict = load_file(file_path)
            logger.info(f"Loaded state_dict for adapter {adapter_id} from {file_path}")
            adapter.load_state_dict(state_dict, strict=False)
            logger.info(f"Adapter {adapter_id} state_dict loaded successfully")

            # Move to device and dtype
            device = device or self.device
            dtype = dtype or torch.float32
            logger.info(f"Moving adapter {adapter_id} to device: {device}, dtype: {dtype}")
            adapter = adapter.to(device=device, dtype=dtype)
            logger.info(f"Adapter {adapter_id} moved to device and dtype successfully")

            # Cache the model
            self.models[adapter_id] = ModelInfo(
                model=adapter,
                model_type=ModelType.SHUNT_ADAPTER,
                config=config,
                device=device,
                dtype=dtype,
                metadata={"file_path": str(file_path)}
            )
            logger.info(f"Adapter {adapter_id} cached successfully")

            logger.info(f"Successfully loaded adapter: {adapter_id}")
            return adapter

        except Exception as e:
            logger.error(f"Failed to load adapter {adapter_id} from {path or repo_id}/{filename}: {e}")
            logger.debug(f"Traceback: {e.__traceback__}")
            return None

    def load_encoder_model(self,
                           model_type: str, # use this to see if it's compatible with the current model manager
                           model_id: str,
                           model_name_or_path: str,
                           device: Optional[torch.device] = None,
                           dtype: Optional[torch.dtype] = None,
                           force_reload: bool = False,
                           trust_remote_code: Optional[bool] = None,  # Overrides the global TRUST_REMOTE_CODE setting.
                           config: Optional[Dict[str, Any]] = None  # Additional configuration for the model
    ) -> Optional[nn.Module]:
        """
        Load an encoder model (e.g., BERT, T5) and return it.

        Args:
            model_type: Type of the model (e.g., "bert", "t5")
            model_id: Unique identifier for the model
            model_name_or_path: Model name or path
            device: Target device
            dtype: Target dtype
            force_reload: Force reload even if cached

        Returns:
            Loaded model or None if failed
        """
        if model_type == "bert":
            return self.load_bert_model(model_id, model_name_or_path, device, dtype, force_reload, trust_remote_code)
        elif model_type == "nomic_bert":
            # Nomic BERT is a specific variant of BERT, so we can use the same loading function
            return self.load_bert_model(model_id, model_name_or_path, device, dtype, force_reload, trust_remote_code)
        elif "t5" in model_type:
            return self.load_t5_model(model_id, model_name_or_path, device, dtype, force_reload, trust_remote_code, config)
        else:
            logger.error(f"Unsupported model type: {model_type}")
            return None

    def load_bert_model(
            self,
            model_id: str,
            model_name_or_path: str,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            force_reload: bool = False,
            trust_remote_code: Optional[bool] = None  # Overrides the global TRUST_REMOTE_CODE setting.
    ) -> Optional[Tuple[nn.Module, Any]]:

        """
        Load a BERT model and tokenizer.

        Returns:
            Tuple of (model, tokenizer) or None if failed
        """
        if not force_reload and self.is_loaded(model_id):
            logger.info(f"Using cached BERT model: {model_id}")
            model_info = self.get_model(model_id)
            return model_info.model, model_info.metadata.get("tokenizer")

        try:
            device = device or self.device
            dtype = dtype or torch.float32

            config = AutoConfig.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code if trust_remote_code is not None else TRUST_REMOTE_CODE  # Use the global flag for remote code execution
            )

            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                config=config,
                use_special_tokens=True,  # Ensure special tokens are used
                trust_remote_code=trust_remote_code if trust_remote_code is not None else TRUST_REMOTE_CODE  # Use the global flag for remote code execution
            )
            model = AutoModel.from_pretrained(
                model_name_or_path,
                config=config,
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code if trust_remote_code is not None else TRUST_REMOTE_CODE  # Use the global flag for remote code execution
            ).to(device)

            # Cache the model

            self._store(_make_key("bert", model_id), ModelInfo(
                model=model,
                model_type=ModelType.BERT_MODEL,
                config={"model_name": model_name_or_path},
                device=device,
                dtype=dtype,
                metadata={"tokenizer": tokenizer},
                trust_remote_code=trust_remote_code if trust_remote_code is not None else TRUST_REMOTE_CODE
            ))

            logger.info(f"Successfully loaded BERT model: {model_id}")
            return model, tokenizer

        except Exception as e:
            logger.error(f"Failed to load BERT model {model_id}: {e}")
            return None

    def load_t5_model(
            self,
            model_id: str,
            model_name_or_path: str,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            force_reload: bool = False,
            override_remote_code: Optional[bool] = None, # Overrides the global TRUST_REMOTE_CODE setting.
            config: Optional[Dict[str, Any]] = None  # Additional configuration for the model
    ) -> Optional[Tuple[nn.Module, Any]]:
        """
        Load a T5 model and tokenizer.

        Returns:
            Tuple of (model, tokenizer) or None if failed
        """
        if not force_reload and self.is_loaded(model_id):
            logger.info(f"Using cached T5 model: {model_id}")
            model_info = self.get_model(model_id)
            return model_info.model, model_info.metadata.get("tokenizer")

        try:
            device = device or self.device
            dtype = dtype or torch.float32
            trust_remote_code = override_remote_code if override_remote_code is not None else TRUST_REMOTE_CODE
            # Load tokenizer and model
            if config.get("type", "t5") == "t5":
                tokenizer = AutoTokenizer.from_pretrained(
                    "google/flan-t5-base",
                    trust_remote_code=trust_remote_code  # Use the global flag for remote code execution
                )
            elif config.get("type", "t5") == "t5_unchained":
                tokenizer = T5TokenizerFast.from_pretrained(
                    "AbstractPhil/t5xxl-unchained",
                    trust_remote_code=trust_remote_code  # Use the global flag for remote code execution
                )
            else:
                tokenizer = T5TokenizerFast.from_pretrained(
                    "google/flan-t5-base",
                    trust_remote_code=trust_remote_code  # Use the global flag for remote code execution
                )

            if config.get("type", "t5") == "t5":
                logger.info(f"Loading T5ForConditionalGeneration model from {model_name_or_path}")
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name_or_path,
                    torch_dtype=dtype,
                    trust_remote_code=trust_remote_code  # Use the global flag for remote code execution
                ).to(device)
            elif config.get("type", "t5") == "t5_encoder_with_projection":
                # Load T5EncoderModel with projection layer
                logger.info(f"Loading T5EncoderWithProjection model from {model_name_or_path}")
                model = T5EncoderWithProjection.from_pretrained(
                    model_name_or_path,
                    torch_dtype=dtype,
                    trust_remote_code=trust_remote_code  # Use the global flag for remote code execution
                ).to(device)

            else:
                # Load standard T5 model
                logger.info(f"Loading T5EncoderModel from {model_name_or_path}")
                model = AutoModel.from_pretrained(
                    model_name_or_path,
                    torch_dtype=dtype,
                    trust_remote_code=trust_remote_code  # Use the global flag for remote code execution
                ).to(device)

            # Cache the model
            self._store(_make_key("t5", model_id), ModelInfo(
                model=model,
                model_type=ModelType.T5_MODEL,
                config={"model_name": model_name_or_path},
                device=device,
                dtype=dtype,
                metadata={"tokenizer": tokenizer}
            ))

            logger.info(f"Successfully loaded T5 model: {model_id}")
            return model, tokenizer

        except Exception as e:
            logger.error(f"Failed to load T5 model {model_id}: {e}")
            return None

    def unload_model(self, model_id: str) -> bool:
        """
        Unload a model to free memory.

        Returns:
            True if successfully unloaded, False otherwise
        """
        if model_id in self.models:
            try:
                # Move to CPU first to free GPU memory
                model_info = self.models[model_id]
                model_info.model.cpu()

                # Delete the model
                del self.models[model_id]

                # Force garbage collection
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                logger.info(f"Successfully unloaded model: {model_id}")
                return True

            except Exception as e:
                logger.error(f"Failed to unload model {model_id}: {e}")
                return False
        else:
            logger.warning(f"Model {model_id} not found in cache")
            return False

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all loaded models with their information"""
        return {
            model_id: {
                "type": info.model_type.value,
                "device": str(info.device),
                "dtype": str(info.dtype),
                "config": info.config
            }
            for model_id, info in self.models.items()
        }

    def clear_all(self):
        """Clear all loaded models"""
        model_ids = list(self.models.keys())
        for model_id in model_ids:
            self.unload_model(model_id)
        logger.info("All models cleared from memory")

    def _resolve_file_path(
            self,
            local_path: Optional[str],
            repo_id: Optional[str],
            filename: Optional[str]
    ) -> Optional[Path]:
        """Resolve file path from local or HuggingFace"""
        # Try local path first
        if local_path and os.path.exists(local_path):
            return Path(local_path)

        # Try HuggingFace
        if repo_id and filename:
            try:
                from huggingface_hub import hf_hub_download

                file_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir=str(self.cache_dir),
                    repo_type="model"
                )
                return Path(file_path)

            except Exception as e:
                logger.error(f"Failed to download from HuggingFace: {e}")

        return None

    def _maybe_convert_dtype(
            self,
            model_id: str,
            target_dtype: Optional[torch.dtype],
            target_device: Optional[torch.device]
    ) -> Optional[nn.Module]:
        """Convert model dtype/device if needed"""
        model_info = self.get_model(model_id)
        if not model_info:
            return None

        model = model_info.model
        changed = False

        # Check dtype conversion
        if target_dtype and model_info.dtype != target_dtype:
            try:
                model = model.to(dtype=target_dtype)
                model_info.dtype = target_dtype
                changed = True
                logger.info(f"Converted {model_id} to dtype: {target_dtype}")
            except Exception as e:
                logger.error(f"Failed to convert dtype for {model_id}: {e}")

        # Check device conversion
        if target_device and model_info.device != target_device:
            try:
                model = model.to(device=target_device)
                model_info.device = target_device
                changed = True
                logger.info(f"Moved {model_id} to device: {target_device}")
            except Exception as e:
                logger.error(f"Failed to move {model_id} to device: {e}")

        if changed:
            model_info.model = model

        return model


    def __del__(self):
        """Cleanup on deletion"""
        self.clear_all()


# Global instance (singleton pattern)
_global_model_manager: Optional[ModelManager] = None


def get_model_manager(cache_dir: Optional[str] = None) -> ModelManager:
    """Get or create the global model manager instance"""
    global _global_model_manager

    if _global_model_manager is None:
        _global_model_manager = ModelManager(cache_dir=cache_dir)

    return _global_model_manager



@dataclass
class ClipPipelineConfig:
    # houses the information required for this CLIP model down the pipeline, can exist or not.
    clip_type: str
    model_path: str
    dtype: torch.dtype
    tokenizer: Optional[str] = None
    source: Optional[str] = None
