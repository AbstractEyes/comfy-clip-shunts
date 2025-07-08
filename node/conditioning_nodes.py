"""
    Conditioning Nodes
    Author: AbstractPhil

    This houses a uniquely designed, flexible, and highly potent conditioning node setup for ComfyUI.
    These are designed to be used with everything from simple text encoders to complex multimodal models.
    The conditioning system does not require any specific model to be used, and only requires encoded tensors.

    Core Conditioner:
        Replaces the current conditioning pipeline entirely with a new one.

    Conditioning Configuration:
        Includes a complex series of conditioning configurations that allow the encoders to be targeted and configured.
        Each encoder can be assigned a unique identifier, and each identifier can have it's own unqiue configuration.
        These include features like:
            * full scheduling and formula-capable internal components
            * advanced tokenization and encoder management capabilities
            * the standard simple and advanced conditioning nodes
            * soft and hard attention masks for conditioning shaping including formula access
            * dtype conversion and management for the conditioning tensors at runtime
            * shunt conditioning shaping using a multitude of experimental and advanced paper-driven techniques
            * runtime token classification for multi-shot capable conditioning using additional encoders
"""
import os
import comfy


class PromptConditioningNode:
    """
    A node to handle prompt conditioning in ComfyUI.
    This node is designed to be flexible and can be used with various text encoders.
    It allows for advanced conditioning configurations and management.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "encoder": ("ENCODER", ),
            },
        }

    RETURN_TYPES = ("CONDITIONING", )
    RETURN_NAMES = ("conditioning", )
    FUNCTION = "condition_prompt"
    CATEGORY = "conditioning/prompt"

    def condition_prompt(self, prompt, encoder):
        """
        Conditions the provided prompt using the specified encoder.
        """
        conditioning = comfy.conditioning.Conditioning(prompt=prompt, encoder=encoder)
        return (conditioning, )