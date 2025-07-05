"""
    Encoder Factory for ComfyUI
    Author: AbstractPhil

    This module provides a factory for creating hookable encoder structures in ComfyUI.
    Everything encoder-related is hooked piece by piece layer by layer, allowing for flexible and dynamic encoder management.

    Attaching callbacks is as simple as attaching a function to the `callback_hooks` dictionary of the `NeuralIO` class.
    [Callback, Callback] -> [Condition, Callback]
    Each condition is a bool-centric function that returns True or False based on the input and is used to trigger callbacks.
    Each callback is likely a necessary and core function dedicated to the encoder's functionality,
    whether it be a tokenizer, or a layer, or a complete model.

"""

import os
from dataclasses import dataclass

import torch
from torch import nn

class EncoderFactory(nn.Module):
    pass
    ##def __init__(self, device="cpu", dtype=None, model_options={}):
    ##    super().__init__()
    ##    self.device = device
    ##    self.dtype = dtype or torch.float32
    ##    self.model_options = model_options
##
    ##def create_encoder(self, identifier: str,
    ##                   expectations: dict,
    ##                   use_repo_config: bool = Tru