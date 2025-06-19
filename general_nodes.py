import comfy
import comfy.utils
import importlib

from .configs import ShuntUtil
import logging

class ABS_ReplaceClip:
    """
    A node to replace the current CLIP model with a new one.
    This is useful for switching between different CLIP models in a pipeline.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", ),
            },
        }

    RETURN_TYPES = ("CLIP", )
    RETURN_NAMES = ("clip", )
    FUNCTION = "replace_clip"
    CATEGORY = "General/CLIP"

    def replace_clip(self, clip):
        """
        Replaces the current CLIP model with the provided one.
        """
        return (clip, )


class ABS_SimpleTextNode:
    """
    A simple text node that can be used to display information or instructions.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": ""}),
                "multiline": ("BOOLEAN", {"default": True, "label": "Multiline Text"}),
            },
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("multiline_text",)
    FUNCTION = "send_text"
    CATEGORY = "General/Display"

    def send_text(self, text, multiline):
        """
        Outputs the provided text as a multiline string if specified.
        """
        if multiline:
            return (text, )
        else:
            return (text.replace("\n", " "), )

class ABS_DebugNode:
    # takes in anything and outputs it as a debug message
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "*": ("ANY", {"default": ""}),
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "debug"
    CATEGORY = "General/Debug"
    OUTPUT_NODE = True
    def debug(self, *args):
        """
        Outputs the provided arguments as a debug message.
        """
        message = "Debug: " + ", ".join(str(arg) for arg in args)
        logging.info(message)
        return ()