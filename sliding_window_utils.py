import torch
"""
    Sliding Window Utilities
    Author: AbstractPhil
    Description: Utility functions for sliding window operations on tensors and sequences.
    # License: MIT License

    These mimic the timestep functionality of interpolated conditional adapters based on masking layer outputs.

    Produces X amount of sliding windows from a given tensor or sequence
    with a specified step size and window length.
    Minimum and maximum lengths can be specified to control the output.

    This is designed to work NATIVELY with ComfyUI's tensor and sequence handling,
    which means this can be reused with many different types of ComfyUI-based timestep systems and sequences.
"""


class ShuntStackConfig:
    # capitalize

    MODEL_TYPE = "shunt_adapter"
    MODEL_NAME = "abs_shunt_adapter"
    LOCAL_PATH = "custom_nodes/comfy-abs-shunt-adapters/shunt_adapter_model.pt"
    CONTEXT_WINDOW = True
    USE_CONTEXT_WINDOW = True
    CONTEXT_WINDOW_SIZE = 512
    SLIDING_WINDOW_SIZE = 256
    SLIDING_WINDOW_STRIDE = 128
    MAX_LENGTH = 1024
    FOLDING = "sliding_window"
    PADDING = "max_length"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ShuntStackBuilder:
    # Builds the representative conditioning stack for the Shunt adapters
    pass

