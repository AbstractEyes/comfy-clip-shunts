import comfy
import torch
from torch import nn
import logging

logger = logging.getLogger(__name__)



class SamplerCore(nn.Module):
    """
    Base class for sampler core functionality.
    This class provides the basic structure for implementing different sampling strategies.
    """

    def __init__(self, encoders, conditioners, sampler_config, model_config):
        super().__init__()
        self.model_config = model_config