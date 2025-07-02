import comfy
import torch
import uuid

from comfy.model_management import intermediate_device
from dataclasses import dataclass
from typing import Optional

from .sd import CLIP # using our modified clip

@dataclass
class EncoderContainerTarget:
    UNORDERED = "unordered" # doesn't matter which encoder model is used, shove it in the square hole.
    SD1 = "sd1"             # clip_l
    SD2 = "sd2"             # clip_l, clip_g?
    SDXL = "sdxl"           # clip_l, clip_g
    SD3 = "sd3"             # clip_l, clip_g, t5xxl
    SD35M = "sd35m"         # clip_l, clip_g, t5xxl
    SD35L = "sd35l"         # clip_l, clip_g, t5xxl
    FLUX = "flux"           # clip_l, t5xxl
    HIDREAM = "hidream"     # clip_l, clip_g, t5xxl, llama
    PIXART = "pixart"       # ??
    HUNYUAN = "hunyuan"     # clip_l, t5xxl? llama?
    WAN = "wan"             # clip_l, llama
    UNKNOWN = "unknown"     # anything ordered


@dataclass
class EncoderTypes:
    WHATEVER = "whatever"       # unknown encoder type, doesn't matter which encoder if it fits the size
    CLIP = "clip"               # Standard CLIP encoder, e.g. SDXL, SD3, etc
    TEXT = "text"               # Text encoder, e.g. T5XXL, LLAMA3, etc


@dataclass
class EncoderWrapper:
    identifier: str = ""                        # unique identifier for the CLIP model, e.g. "clip-vit-large-patch14"
    type: EncoderTypes = EncoderTypes.WHATEVER   # the type of the encoder, e.g. CLIP, TEXT, etc
    can_project_upward: bool = False            # whether the encoder can be interpolated to a larger size
    can_project_downward: bool = False          # whether the encoder can be interpolated to a smaller size
    encoder: Optional[CLIP | dict] = None       # ONE is required to function
    state_dict: Optional[dict] = None           # houses the symbolic link to the CLIP model, must be confirmed before use
    config: Optional[dict] = None               # houses the configuration of the CLIP model, must be confirmed before
    tokenizer: Optional[object] = None          # houses the tokenizer, if any, used by the CLIP model
    tokenizer_config: Optional[dict] = None     # houses the configuration of the tokenizer, if any
    patcher: Optional[object] = None            # houses the patcher, if any, used by the CLIP model; needed for loras
    device: Optional[str] = None                # the device on which the CLIP model is loaded, e.g. "cuda:0" or "cpu"
    metadata: Optional[dict] = None             # houses metadata about the CLIP model

class EncoderContainer:
    # houses representations to the currently pipelined Encoder models
    intermediate_device = intermediate_device()
    device = intermediate_device

    def __init__(self,
                 target_size: int = 768,
                 order: EncoderContainerTarget = EncoderContainerTarget.UNORDERED):
        self.id = uuid.uuid4()
        self.order = order
        # all the contained encoders to encode data with
        self.container: list = []
        # the expected size of the output tensor pool from all encoders
        self.target_size = target_size


    def append(self, encoder: EncoderWrapper):
        # append an encoder to the container
        if not isinstance(encoder, EncoderWrapper):
            raise TypeError("Encoder must be an instance of EncoderWrapper")
        self.container.append(encoder)
        return

    def order(self, encoderTarget: EncoderContainerTarget):
        # set the order so it can be used to encode data in the correct order.
        self.target = encoderTarget
        return


    def encode(self, data, config):
        # instruct the container to encode the data in the correct order using it's encoders.
        pass # TODO: implement the multi-tiered encoding system

    def grab_slices(self):
        pass # grabs the sliced output from the encoders in the container



def make_encoder_wrapper(encoder: CLIP, identifier: str = "", type: EncoderTypes = EncoderTypes.CLIP,
                     can_project_upward: bool = False, can_project_downward: bool = False,
                     state_dict: Optional[dict] = None, config: Optional[dict] = None,
                     tokenizer: Optional[object] = None, tokenizer_config: Optional[dict] = None,
                     patcher: Optional[object] = None, device: Optional[str] = None,
                     metadata: Optional[dict] = None) -> EncoderWrapper:
    return EncoderWrapper(identifier=identifier, type=type, can_project_upward=can_project_upward,
                          can_project_downward=can_project_downward, encoder=encoder, state_dict=state_dict,
                          config=config, tokenizer=tokenizer, tokenizer_config=tokenizer_config,
                          patcher=patcher, device=device, metadata=metadata)