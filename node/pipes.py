from dataclasses import dataclass
from typing import Optional

import comfy
import torch
import torch.nn as nn



class EncodingType:
    """
    Enum-like class to represent different encoding types.
    """

    VISION = "vision"
    TEXT = "text"
    HYBRID = "hybrid"
    AUDIO = "audio"
    SIMILARITY = "similarity"
    SHUNTED = "shunted"


class GeneratedType:
    """
    Enum-like class to represent different generated types.
    """

    IMAGE = "image"
    TEXT = "text"
    AUDIO = "audio"
    VIDEO = "video"
    SIMILARITY = "similarity"
    SHUNTED = "shunted"


class EncoderPipe:

    def __init__(self):
        self.encoders = {}
        self.config = {}

    def add_encoder(self, encoder, config):
        """
        Adds an encoder to the pipe.
        :param encoder: The encoder to add.
        :param config: Configuration for the encoder.
        """
        # use existing config if it exists; clone and merge with new config
        if encoder.name in self.config:
            existing_config = self.config[encoder.name]
            config = {**existing_config, **config}
        self.encoders[encoder.name] = config

    def clone(self):
        """
        Clones the current EncoderPipe instance.
        :return: A new EncoderPipe instance with the same encoders and config.
        """
        new_pipe = EncoderPipe()
        new_pipe.encoders = self.encoders.copy()
        new_pipe.config = self.config.copy()
        return new_pipe


@dataclass
class EncodedNode:
    # represents a node for the ConditionPipe system
    embedding: torch.Tensor                 # embedding tensor for the node
    config: Optional[dict] = None           # additional configuration for the node
    loggings: dict = None                   # additional logging information for the node

    def log(self, key: str, value: str):
        """
        Logs a key-value pair to the node's logging dictionary.
        :param key: The key to log.
        :param value: The value to log.
        """
        if self.loggings is None:
            self.loggings = {}
        self.loggings[key] = value


class ModelConditioningExpectation:
    """
    Represents a model's expected conditioning models and pooled models.
    This determines how the outputs of a model should be expected for converting to a ComfyUI consumable format.

    This requires at least one expected conditioner and most likely one expected pooled model.
    Omitting a pooled model is acceptable, but will probably fail for most models.
    """

    def __init__(self,
                 model_name: str,
                 expected_conditioners: list = [], # the expected conditioning models
                 expected_pooled: list = [], # the expected pooled models
                 config: Optional[dict] = None):
        self.model_name = model_name
        self.expected_conditioners = expected_conditioners
        self.expected_pooled = expected_pooled
        self.config = config

    # models require certain conditioning sizes and a pooled size
    # these should be streamlined so the user doesn't end up needing to count everything
    # however, the expert may want to substitute things for other things
    # that's why we provide so many projection options with other nodes


    def create(self,
             node: any, #ConditionNode,
             format: str = "comfyui"):
        """
        Creates a ComfyUI consumable representation of the daisy chained nodes.
        """
        #todo: create the translation to ComfyUI consumable format after the surrounding structure is ready
        return node





class ConditionPipe:
    ...


