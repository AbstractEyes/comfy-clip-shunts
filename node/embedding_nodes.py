import os
import logging
from pathlib import Path
from typing import Optional
import folder_paths

import torch
from comfy.sd import CLIP
from ..model.configs import ShuntUtil

logger = logging.getLogger(__name__)

import hashlib
from ..model.model_manager import get_model_manager
from ..model.configs import ENCODER_CONFIGS, ShuntData, EncoderData
from ..utils.conditioning_shifter import ConditioningShifter

from ..sampler.formulas.folding import FoldingKernels
from ..sampler.formulas.schedules import SchedulerModes
from ..text_encoders.embedding_manager import get_bank, EmbeddingManager


class ASimpleEmbeddingShaperNode:
    """
    A node to reshape embeddings for the ABS Shunt Adapters.
    This node allows you to reshape embeddings based on the provided configuration.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", ),
                "prompt": ("STRING", {"default": "", "multiline": False}),
                "learn": ("BOOLEAN", {"default": False}),
                "directory": ("STRING", {"default": "", "multiline": False}),
                "force_recache": ("BOOLEAN", {"default": False}),

            },
        }

    RETURN_TYPES = ("CONDITIONING", )
    RETURN_NAMES = ("conditioning", )
    FUNCTION = "reshape_embedding"
    CATEGORY = "utils/embedding"

    def reshape_embedding(self, conditioning: list,
                          prompt: str,
                          learn: Optional[bool] = False,
                          directory: str = "",
                          force_recache: Optional[bool] = False) -> tuple:
        """
        Reshapes the provided conditioning embeddings based on the prompt and learn flag.
        If a directory is provided, it will save the reshaped embeddings to that directory.
        """
        if not isinstance(conditioning, list):
            raise ValueError("Conditioning must be a list of embeddings.")

        # Get the embedding bank
        bank = get_bank(directory)
        if not bank:
            raise ValueError("Embedding bank could not be retrieved.")

        # Reshape the embeddings
        reshaped_conditioning = bank.process_embeddings(conditioning, prompt, learn, force_recache)

        return (reshaped_conditioning, )



class ASaveEmbeddingFromConditioning:
    """
    A class to save and load embeddings for the ABS Shunt Adapters.
    This class handles the saving and loading of embeddings to and from the embedding folder.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", ),
                "activator": ("STRING", {"default": "", "multiline": False}),
                "sequence": ("STRING", {"default": "", "multiline": False}),
                "name": ("STRING", {"default": "", "multiline": False}),
                "subfolder": ("STRING", {"default": "cached_embeddings", "multiline": False}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", )
    RETURN_NAMES = ("conditioning", )
    FUNCTION = "save_embedding"
    CATEGORY = "utils/context_window"

    def save_embedding(self,
                       conditioning: list,
                       activator: str,
                       prompt: str,
                       learn: Optional[bool] = False,
                       ) -> tuple:
        ...
        # embedding_directory = folder_paths.get_directory_by_type("embeddings")
        # if not os.path.exists(Path(embedding_directory + "/" + subfolder)):
        #     os.makedirs(embedding_directory)
        # # Create a unique filename based on the activator, sequence, and name
        # bank = get_bank(embedding_directory)
        # bank.save_embedding(activator, sequence, name, conditioning, folder="")




