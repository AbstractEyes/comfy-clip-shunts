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
import torch
from .formulas.modes import (
    FoldingPaddingTypes,
    FoldingTypes,
    FoldingPoolingTypes
)


class ShuntStackConfig:
    # capitalize
    CONTEXT_WINDOW = True
    OVERRIDE_CONTEXT_WINDOW = True
    FOLD_STEPS = 4  # Number of folds to apply to the context window, cannot exceed number of windows
    PADDING_MODE = "max_length"  # Padding mode for sequences
    PADDING_FILL_MODE = FoldingPaddingTypes.NONE # Here we determine if we fill the dead space with interpolated values or leave them masked.
    CONTEXT_WINDOW_SIZE = 512
    SLIDING_WINDOW_SIZE = 77
    SLIDING_WINDOW_STRIDE = 128
    MAX_LENGTH = 1024
    FOLDING = "sliding_window"
    PADDING = "max_length"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SlidingWindowBuilder:
    # calculate how many sliding window strides must occur to reach the context window size.
    # afterwords, calculate how many context windows must be created to reach the maximum length.
    # interpolate the overlapping windows to create a full context window that saturates all tokens.

    @staticmethod
    def build_sliding_windows(tensor: torch.Tensor, config: ShuntStackConfig) -> list:
        """
        Builds sliding windows from the input tensor based on the configuration.

        Returns:
            list: A list of sliding window tensors.
        """
        windows = []
        stride = config.SLIDING_WINDOW_STRIDE
        window_size = config.SLIDING_WINDOW_SIZE
        max_length = config.MAX_LENGTH

        # Clamp sizes
        context_window_size = min(config.CONTEXT_WINDOW_SIZE, max_length)
        sliding_window_size = min(window_size, max_length)
        sliding_window_stride = min(stride, max_length)

        # Get total context slice from tensor
        main_window = tensor[:, :context_window_size]  # For global analysis or fallback

        # Calculate number of sliding windows
        num_windows = (context_window_size - sliding_window_size) // sliding_window_stride + 1

        for i in range(num_windows):
            start = i * sliding_window_stride
            end = start + sliding_window_size

            # Prevent out-of-bounds
            if end > tensor.size(1):
                break

            window = tensor[:, start:end]
            windows.append(window)

        return windows
