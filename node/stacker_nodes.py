import comfy
import torch
import logging

logger = logging.getLogger(__name__)


class StackEncoderPipelinesNode:
    """
        A node to stack encoders for multi-use conditioning.
        This node allows you to stack multiple encoders into a single conditioning stack.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "encoder_1": ("ENCODER_PIPELINE", {}),
                "encoder_2": ("ENCODER_PIPELINE", {}),
            },
            "optional": {
                "encoder_3": ("ENCODER_PIPELINE", {}),
                "encoder_4": ("ENCODER_PIPELINE", {}),
                "encoder_5": ("ENCODER_PIPELINE", {}),
            },
        }
    RETURN_TYPES = ("ENCODER_PIPELINE",)
    RETURN_NAMES = ("encoders",)
    FUNCTION = "stack_encoders"
    CATEGORY = "utils/conditioning"

    def stack_encoders(self, encoder_1: list, encoder_2: list, encoder_3=None, encoder_4=None, encoder_5=None):
        """
        Stacks up to 5 encoder conditioning objects into a single encoder conditioning stack.
        This allows for complex conditioning setups to be managed easily.
        """
        encoders = []
        encoders.extend(encoder_1)
        encoders.extend(encoder_2)
        if encoder_3 is not None:
            encoders.extend(encoder_3)
        if encoder_4 is not None:
            encoders.extend(encoder_4)
        if encoder_5 is not None:
            encoders.extend(encoder_5)

        return (encoders,)


class StackClipPipelines:
    """
    Similar to stacking encoders, this prepares lists of clip_pipelines for use in a pipeline.
    This can be quite strange depending how the pipeline is set up, but it allows for
    complex conditioning setups to be managed in a more convenient way.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_pipeline_1": ("CLIP_PIPELINE", {}),
                "clip_pipeline_2": ("CLIP_PIPELINE", {}),
            },
            "optional": {
                "clip_pipeline_3": ("CLIP_PIPELINE", {}),
                "clip_pipeline_4": ("CLIP_PIPELINE", {}),
                "clip_pipeline_5": ("CLIP_PIPELINE", {}),
            },
        }
    RETURN_TYPES = ("CLIP_PIPELINE",)
    RETURN_NAMES = ("clip_pipelines",)
    FUNCTION = "stack_clip_pipelines"
    CATEGORY = "utils/conditioning"

    def stack_clip_pipelines(self, clip_pipeline_1: list, clip_pipeline_2: list, clip_pipeline_3=None, clip_pipeline_4=None, clip_pipeline_5=None):
        """
        Stacks up to 5 clip pipelines into a single clip pipeline stack.
        This allows for complex conditioning setups to be managed easily.
        """
        clip_pipelines = []
        clip_pipelines.extend(clip_pipeline_1)
        clip_pipelines.extend(clip_pipeline_2)
        if clip_pipeline_3 is not None:
            clip_pipelines.extend(clip_pipeline_3)
        if clip_pipeline_4 is not None:
            clip_pipelines.extend(clip_pipeline_4)
        if clip_pipeline_5 is not None:
            clip_pipelines.extend(clip_pipeline_5)

        return (clip_pipelines,)


class StackConditioning:
    """
    A node to stack multiple conditioning objects into a single conditioning stack.
    This allows for complex conditioning setups to be managed easily.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning_1": ("CONDITIONING", {}),
                "conditioning_2": ("CONDITIONING", {}),
            },
            "optional": {
                "conditioning_3": ("CONDITIONING", {}),
                "conditioning_4": ("CONDITIONING", {}),
                "conditioning_5": ("CONDITIONING", {}),
            },
        }
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditionings",)
    FUNCTION = "stack_conditioning"
    CATEGORY = "utils/conditioning"

    def stack_conditioning(self, conditioning_1: list, conditioning_2: list, conditioning_3=None, conditioning_4=None, conditioning_5=None):
        """
        Stacks up to 5 conditioning objects into a single conditioning stack.
        This allows for complex conditioning setups to be managed easily.
        """
        conditionings = []
        conditionings.extend(conditioning_1)
        conditionings.extend(conditioning_2)
        if conditioning_3 is not None:
            conditionings.extend(conditioning_3)
        if conditioning_4 is not None:
            conditionings.extend(conditioning_4)
        if conditioning_5 is not None:
            conditionings.extend(conditioning_5)

        return (conditionings,)

