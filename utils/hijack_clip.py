import comfy
import torch
from dataclasses import dataclass


from comfy.sd import CLIP


TargetFunctions: dict = {
    "clip_text_encode": "abs_clip_text_encode",  # Placeholder for text encoding function
    "clip_encode_": None,  # Placeholder for image encoding function
    "clip_vision_encode": None,  # Placeholder for vision encoding function
}

@dataclass
class HijackClipConfig:
    """
    Configuration for hijacking CLIP outputs.
    This can be extended with more parameters as needed.
    """
    use_custom_hooks: bool = True
    custom_hook_function: callable = None

class Abs_HijackClip:
    """
    We snap our custom hooks to replace the default CLIP model outputs with a functional hook.
    """

    @staticmethod
    def attach(clip_pipe: CLIP, config: dict = None) -> CLIP:
        """ Determine which clips are here, and then snap our hooks to them as hijacked outputs. """
        # clone first
        clip_pipe = clip_pipe.clone() # this prevents modifying the original clip_pipe for safety

        # If no config provided, we just snap the default hooks.




        return clip_pipe