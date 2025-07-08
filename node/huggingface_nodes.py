import logging

from comfy.sd import CLIP
from ..model.configs import ShuntUtil

logger = logging.getLogger(__name__)

import hashlib
from ..model.model_manager import get_model_manager
from ..model.configs import ENCODER_CONFIGS, ShuntData, EncoderData

class SetHuggingfaceToken:
    """
    Sets the Hugging Face token for accessing private models.
    This is useful for loading models that require authentication.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "token": ("STRING", {
                    "default": get_model_manager().get_huggingface_key(),
                    "tooltip": "Hugging Face token for private model access."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("token",)
    FUNCTION = "set_token"
    CATEGORY = "adapter/testing"
    OUTPUT_NODE = True

    def set_token(self, token):
        """Set the Hugging Face token."""

        if token:
            get_model_manager().set_huggingface_key(token)
        return (token,)

class SetHuggingfaceCacheDirectory:
    """
    Sets the Hugging Face cache directory for model storage.
    This is useful for managing where models are downloaded and stored.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cache_directory": ("STRING", {
                    "default": get_model_manager().get_huggingface_cache_directory(),
                    "tooltip": "Directory to store Hugging Face models."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("cache_directory",)
    FUNCTION = "set_cache_directory"
    CATEGORY = "adapter/testing"
    OUTPUT_NODE = True

    def set_cache_directory(self, cache_directory):
        """Set the Hugging Face cache directory."""
        if cache_directory:
            get_model_manager().set_huggingface_cache_directory(cache_directory)
        return (cache_directory,)
