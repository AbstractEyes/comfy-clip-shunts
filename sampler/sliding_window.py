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
from dataclasses import dataclass

import torch
from .formulas.modes import (
    FoldingPaddingTypes,
    FoldingPoolingTypes
)


@dataclass
class ShuntStackConfig:
    # capitalize
    context_window: str = ""
    override_context_window: bool = True
    fold_steps: int = 4  # Number of folds to apply to the context window, cannot exceed number of windows
    padding_mode: str = "max_length"  # Padding mode for sequences
    padding_fill_mode: any = FoldingPaddingTypes.NONE # Here we determine if we fill the dead space with interpolated values or leave them masked.
    context_window_size: int = 512
    sliding_window_size: int = 77
    sliding_window_stride: int = 128
    max_length: int = 1024
    max_windows: int = 8  # Maximum number of sliding windows to create
    folding: str = "sliding_window"
    padding: str = "max_length"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


    def __copy__(self):
        return ShuntStackConfig(
            context_window=self.context_window,
            override_context_window=self.override_context_window,
            fold_steps=self.fold_steps,
            padding_mode=self.padding_mode,
            padding_fill_mode=self.padding_fill_mode,
            context_window_size=self.context_window_size,
            sliding_window_size=self.sliding_window_size,
            sliding_window_stride=self.sliding_window_stride,
            max_length=self.max_length,
            max_windows=self.max_windows,
            folding=self.folding,
            padding=self.padding,
            device=self.device
        )


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
        stride = config.sliding_window_stride
        window_size = config.sliding_window_size
        max_length = config.max_length

        # Clamp sizes
        context_window_size = min(config.context_window_size, max_length)
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
