"""
    Encode Anything Encoder Classes

    This module provides a series of classes and utils for encoders that handle encoding text inputs into a more complex series of variations.

    It supports diffusers, transformers, and other encoder types, allowing for flexible integration into various pipelines.
    Self encapsulating and can be used to encode text inputs into embeddings or other representations.
    Intercepts, replaces, and interpolates pre-tokenization, unprepared embeddings, sequences, and other data types.
    Handles top-k, top-20k, and top-50k tokenization schemes, as well as various encoding modes.

    This util is primarily static functions and does not require instantiation, which makes it ideal for reuse.

"""


import torch

from typing import Optional, Any
from ..sampler.formulas.schedules import SchedulerModes
from ..sampler.formulas.folding import FoldingKernels
from ..sampler.formulas.padding import FoldingPaddingTypes

class ScheduledEncoderConfig:
    """
    Configuration for Scheduled Encoder.
    Contains settings for the encoder's operation, including model type, context window size,
    and other parameters that control how the encoder processes inputs.
    """
    CONTEXT_WINDOW = ""
    OVERRIDE_CONTEXT_WINDOW = False
    SCHEDULER_MODE = SchedulerModes.TAU  # Default scheduler mode which is tau
    FOLDING_MODE = FoldingKernels.fold  # Default folding mode
    PADDING_FILL_MODE = FoldingPaddingTypes.NONE  # Determines how padding is filled in sequences
    PADDING_MODE = "max_length"  # Padding mode for sequences
    CONTEXT_WINDOW_SIZE = 450
    SLIDING_WINDOW_SIZE = 225
    SLIDING_WINDOW_STRIDE = 77
    MAX_LENGTH = 512
    FOLDING = "sliding_window"
    PADDING = "max_length"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class AbsEncoderProcessor:
    # contains a representation of the encoder's baseline functionality
    # will be used to;
    # 1. take in plain text, images, latents, encodings, or other data types and encode them using a simple formula.
    # 2. intercept tokens directly and sort them into a more easy to rationalize format.
    # 3. handles multiple forms of preprocessing, processing, and postprocessing for the data.
    # 4. can prepare the encodings to a form directly usable by samplers downstream for ComfyUI.

    @staticmethod
    def encode_text(text: str,
                    tokenizer: Any,
                    model: Any,
                    input_shape: Optional[tuple] = None,
                    target_shape: Optional[tuple] = None) -> torch.Tensor:
        """
        Encodes a text input into a tensor representation using the provided tokenizer and model.

        Args:
            text (str): The input text to encode.
            tokenizer (Any): The tokenizer to use for encoding the text.
            model (Any): The model to use for encoding the text.
            input_shape (tuple): The shape of the input tensor. If None, uses the shape from the tokenizer.
            target_shape (Optional[tuple]): The desired shape of the output tensor. If None, uses input_shape.

        Returns:
            torch.Tensor: The encoded tensor representation of the input text.
        """
        tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        inputs = tokens['input_ids'].to(model.device)

        with torch.no_grad():
            outputs = model(inputs)

        if target_shape is not None:
            outputs = outputs.view(*target_shape)

        return outputs


