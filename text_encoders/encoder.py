"""
    Encoder module for ComfyUI, providing a flexible interface for various text, vision, and multimodal encoders.
    Author: AbstractPhil

    The primary workhorse is NeuralIO, which is a dataclass that houses the expectations for neural models to be used in a pipeline.
    If this paradigm of neural model handling requires additional functionality, it can be easily extended with
    simple torch lambda functions or additional methods.

    I built this as a flexible interface for various text, vision, and multimodal encoders.
    This is built as a solution to the messy and inconsistent handling of encoders in ComfyUI,
    this inconsistency leads to issues with model loading, configuration, and usage across different nodes.
    The entire spectrum of encoders is affected by this sloppy handling, including CLIP, T5, LLAMA, and other text-based.

    This is a centralized and unified interface for handling encoders, allowing for easier management in the
    torch-based ComfyUI environment. It's intended to provide a consistent way to load, configure, and use encoders
    without the need for multiple different rewritten classes and methods - essentially gating the entire encoder system
    from the average user who isn't allowed to touch the internals of ComfyUI in a distributed environment.

    The encoder module provides a series of convenient classes and methods to handle various encoders without altering
    core ComfyUI functionality or workflows.

    The entire structure is built around the concept of fixing problems.

    So if this module is not working as intended, or causes more problems than it solves, it will be deprecated.
"""
import comfy
import torch
import torch.nn as nn
import uuid

from comfy.model_management import intermediate_device
from dataclasses import dataclass
from typing import Optional

from comfy import supported_models

from ..abs_sd.sd import CLIP # we will be using a modified CLIP pipeline for the encoder, so we import it here


@dataclass
class NeuralIO(nn.Module):
    # houses the target expectations for a transformers-based neural model to be used in a pipeline
    identifier: str = ""               # required target identifier for a tokenizer, e.g. "clip-vit-large-patch14"
    types: str = ""                    # required target type for a tokenizer, e.g. "clip", "text", etc
    config: dict = ()                  # optional configuration for the tokenizer, e.g. {"clip": {"vision_tower": "clip-vit-large-patch14"}}
    input_expectations: {} = None      # optional list of expected tokenizers, e.g. ["clip-vit-large-patch14", "t5-xxl"]
    output_expectations: {} = None     # optional list of expected outputs, e.g. ["clip", "text", shape=(1, 768), "clip-vit-large-patch14", "t5-xxl"]


@dataclass
class InterpolationWrapper:
    # houses the interpolation information for the encoder
    can_project_upward: bool = False       # whether the encoder can be interpolated to a larger size
    can_project_downward: bool = False     # whether the encoder can be interpolated to a smaller size
    target_size: int = -1                  # -1 tries to guess, any other is the target size of the output tensor pool

@dataclass
class EncoderWrapper(nn.Module):
    identifier: str = ""                               # unique identifier for the Encoder model, e.g. "clip-vit-large-patch14"
    expectations: dict[str, NeuralIO] = ()             # Our IO container - housing the expectations for the encoder model, e.g. {"Encoder": EncoderIO(...)}
    encoders: dict[str, NeuralIO] = ()                 # AT LEAST ONE is required to function
    tokenizers: dict[str, NeuralIO] = ()               # tokenizer or tokenizers, if any, used by the Encoder model
    state_dict: Optional[dict] = None                  # symbolic link to the Encoder model, must be confirmed before use
    config: Optional[dict] = None                      # configuration of the Encoder model, must be confirmed before

    patcher: Optional[object] = None                   # patcher, if any, used by the Encoder model; needed for loras
    device: str | torch.device = "cpu"                 # the device on which the Encoder model is loaded, e.g. "cuda:0" or "cpu"
    metadata: dict = ()                                # metadata about the Encoder model


class EncoderConditioner(NeuralIO):
    # houses representations to the currently pipelined Encoder models
    intermediate_device = intermediate_device()
    device = intermediate_device

    def __init__(self,
                 target_size: int = 768,
                 ):
        self.id = uuid.uuid4()
        self.order = order
        # all the contained encoders to encode data with
        self.container: list = []
        # the expected size of the output tensor pool from all encoders
        self.target_size = target_size


    def append(self, encoder: EncoderWrapper):
        # append an encoder to the container
        if not isinstance(encoder, EncoderWrapper):
            raise TypeError("Encoder must be an instance of EncoderWrapper")
        self.container.append(encoder)
        return

    def order(self, encoderTarget: EncoderContainerTarget):
        # set the order so it can be used to encode data in the correct order.
        self.target = encoderTarget
        return


    def encode(self, data, config):
        # instruct the container to encode the data in the correct order using it's encoders.
        pass # TODO: implement the multi-tiered encoding system

    def grab_slices(self):
        pass # grabs the sliced output from the encoders in the container

