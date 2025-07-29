import numpy as np
from dataclasses import dataclass
from typing import Optional, Union

#from custom_nodes.comfyui_controlnet_aux.src.custom_midas_repo.midas.blocks import Interpolate


#@dataclass
#class ConditioningSchedulerTypes:
#    TAU = "tau"
#    COS = "cos"
#    WAVE = "wave"
#    PULSE = "pulse"
#    SHOCKWAVE = "shockwave"
#    CASCADE = "cascade"
#    TOP_K = "top_k"
#    TOP_20K = "top_20k"
#    TOP_50K = "top_50k"
#    SINE = "sine"
#    COSINE = "cosine"
#    NONE = "none"  # Default mode
#
#CONDITIONING_SCHEDULERS = []
#for item in ConditioningSchedulerTypes.__dict__.items():
#    CONDITIONING_SCHEDULERS.append(item[1]) if not item[0].startswith('__') and not callable(item[1]) else None

#@dataclass
#class FoldingTypes:
#    """
#        "rigid", "zeus", "helios", "surge", "surge-fold", "fold", "interpolate",
#        "collapse", "zipper", "concat-flatten", "cascade", "ripple"
#    """
#    SURGE_FOLD = "surge-fold"
#    SLERP = "slerp"  # Spherical linear interpolation
#    SLIP = "slip"  # Entropic phase slip, requires delta in context
#    RIGID = "rigid"
#    FOLD = "fold"
#    ZIPPER = "zipper"
#    RIPPLE = "ripple"
#    SURGE = "surge"
#    COLLAPSE = "collapse"
#    CONCAT_FLATTEN = "concat-flatten"
#    ZEUS = "zeus"
#    HELIOS = "helios"
#    CASCADE = "cascade"
#    INTERPOLATE = "interpolate"
#

#FOLDING_MODES = []
#for item in FoldingTypes.__dict__.items():
#    FOLDING_MODES.append(item[1]) if not item[0].startswith('__') and not callable(item[1]) else None

@dataclass
class FoldingPaddingTypes:
    # how we replace natural padded tokens in the folding interpolation
    BELL_MOLD = "bell_mold"  # Bell mold, where we use a bell-shaped curve to interpolate between padded and full embeddings

    INTERPOLATE = "interpolate"  # Interpolate between masked and full embeddings with folding strategies
    MASK_EDGES = "mask_edges"  # Mask edges of padded tokens, leaving them as is
    MASK_TOP_K = "mask_top_k"  # folding only the top K tokens in the order
    MASK_BOTTOM_K = "mask_bottom_k"  # folding only the bottom K tokens in the order
    REPLACE = "replace"  # Rigidly replace padded tokens with folded embeddings
    GAPPED = "gapped"  # Use a gapped approach, where we leave gaps in the output for padded tokens for spacing
    SPARSE = "sparse"  # Use a sparse approach, where we only fill in non-padded tokens and leave others empty
    BLEND = "blend"  # Blend padded tokens with folded embeddings, but keep base where mask < 0.5
    BLEND2 = "blend2"  # Blend padded tokens with folded embeddings, but keep base where mask < 0.5
    SHUFFLE = "shuffle"  # Shuffle padded tokens with folded embeddings

    NONE = "none"  # No padding replacement, leave padded tokens as is

    @staticmethod
    def to_list() -> list[str]:
        """
        Returns a list of all available padding types.
        """
        return [
            "bell_mold", "shuffle", "interpolate", "mask_edges", "mask_top_k", "mask_bottom_k",
            "blend", "blend2", "replace", "gapped", "sparse", "none",
        ]


@dataclass
class FoldingPoolingTypes:
    # these are applied to the entire pooled set of embeddings if multiple embeddings are provided
    # they are passed by [[start, end],...], giving a start and end index for each embedding to omit or pool
    TRIANGULAR = "triangular_overlap"  # Triangular pooling, where we pool embeddings in a triangular fashion
    BILINEAR = "bilinear"  # Bilinear pooling, where we pool embeddings in a bilinear fashion
    NEAREST = "nearest"  # Nearest pooling, where we pool embeddings in a nearest neighbor fashion
    SIMILARITY_TREE = "similarity_tree" #
    SIMILARITY_O = "similarity_o"  # Similarity pooling, where we pool embeddings orderly based on similarity
    SIMILARITY_X = "similarity_x"  # Similarity pooling, where we pool embeddings in a cross similarity fashion
    SIMILARITY_MASK = "similarity_mask"  # Similarity pooling, where we pool embeddings based on mask similarity
    CONV2 = "conv2"  # Projected conv2 pooling, where we pool embeddings in a convolutional fashion
    CONV3 = "conv3"  # Projected conv3 pooling, where we pool embeddings in a 3D convolutional fashion
    CONV4 = "conv4"  # Projected conv4 pooling, where we pool embeddings in a 4D convolutional fashion
    FLOOD = "flood"  # Flood pooling, where we pool embeddings in a flooding fashion
    SLERP = "slerp"  # Spherical linear interpolation pooling
    AVERAGE = "average"  # Average pooling across embeddings
    MAX = "max"  # Max pooling across embeddings
    SUM = "sum"  # Sum pooling across embeddings
    NONE = "none"  # No pooling, return embeddings as is, default behavior for preliminary testing


    @staticmethod
    def to_list() -> list[str]:
        """
        Returns a list of all available pooling types.
        """
        return [
            "similarity_tree",
            "triangular_overlap", "bilinear", "nearest", "similarity_o", "similarity_x",
            "similarity_mask", "conv2", "conv3", "conv4", "flood", "slerp",
            "average", "max", "sum", "none"
        ]