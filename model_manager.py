from typing import Dict, Optional, Any, Union, Tuple
import os
import torch
import torch.nn as nn
import logging
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Enum for different model types"""
    SHUNT_ADAPTER = "shunt_adapter"
    T5_MODEL = "t5_model"
    BERT_MODEL = "bert_model"
    GENERIC = "generic"


@dataclass
class ModelInfo:
    """Container for model information"""
    model: nn.Module
    model_type: ModelType
    config: Dict[str, Any]
    device: torch.device
    dtype: torch.dtype
    metadata: Dict[str, Any] = None


class ModelManager:
    """
    Centralized model manager for loading, caching, and managing various model types.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.models: Dict[str, ModelInfo] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = self._setup_cache_dir(cache_dir)

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

    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get a loaded model by ID"""
        return self.models.get(model_id)

    def is_loaded(self, model_id: str) -> bool:
        """Check if a model is loaded"""
        return model_id in self.models

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
            from .dual_stream_adapter_model import TwoStreamShuntAdapter

            # Determine file location
            file_path = self._resolve_file_path(path, repo_id, filename)
            if not file_path:
                raise FileNotFoundError(f"Could not find adapter file for {adapter_id}")

            # Initialize adapter
            adapter = TwoStreamShuntAdapter(config=config)

            # Load weights
            state_dict = load_file(file_path)
            adapter.load_state_dict(state_dict, strict=True)

            # Move to device and dtype
            device = device or self.device
            dtype = dtype or torch.float32
            adapter = adapter.to(device=device, dtype=dtype)

            # Cache the model
            self.models[adapter_id] = ModelInfo(
                model=adapter,
                model_type=ModelType.SHUNT_ADAPTER,
                config=config,
                device=device,
                dtype=dtype,
                metadata={"file_path": str(file_path)}
            )

            logger.info(f"Successfully loaded adapter: {adapter_id}")
            return adapter

        except Exception as e:
            logger.error(f"Failed to load adapter {adapter_id}: {e}")
            return None

    def load_t5_model(
            self,
            model_id: str,
            model_name_or_path: str,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            force_reload: bool = False
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

            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name_or_path,
                torch_dtype=dtype
            ).to(device)

            # Cache the model
            self.models[model_id] = ModelInfo(
                model=model,
                model_type=ModelType.T5_MODEL,
                config={"model_name": model_name_or_path},
                device=device,
                dtype=dtype,
                metadata={"tokenizer": tokenizer}
            )

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
