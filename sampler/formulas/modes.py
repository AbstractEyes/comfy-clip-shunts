import numpy as np
from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class ConditioningSchedulerTypes:
    TAU = "tau"
    COS = "cos"
    WAVE = "wave"
    PULSE = "pulse"
    SHOCKWAVE = "shockwave"
    CASCADE = "cascade"
    TOP_K = "top_k"
    TOP_20K = "top_20k"
    TOP_50K = "top_50k"
    SINE = "sine"
    COSINE = "cosine"
    NONE = "none"  # Default mode

CONDITIONING_SCHEDULERS = []
for item in ConditioningSchedulerTypes.__dict__.items():
    CONDITIONING_SCHEDULERS.append(item[1]) if not item[0].startswith('__') and not callable(item[1]) else None

@dataclass
class FoldingTypes:
    """
        "rigid", "zeus", "helios", "surge", "surge-fold", "fold", "interpolate",
        "collapse", "zipper", "concat-flatten", "cascade", "ripple"
    """
    SURGE_FOLD = "surge-fold"
    SLERP = "slerp"  # Spherical linear interpolation
    SLIP = "slip"  # Entropic phase slip, requires delta in context
    RIGID = "rigid"
    FOLD = "fold"
    ZIPPER = "zipper"
    RIPPLE = "ripple"
    SURGE = "surge"
    COLLAPSE = "collapse"
    CONCAT_FLATTEN = "concat-flatten"
    ZEUS = "zeus"
    HELIOS = "helios"
    CASCADE = "cascade"
    INTERPOLATE = "interpolate"

FOLDING_MODES = []
for item in FoldingTypes.__dict__.items():
    FOLDING_MODES.append(item[1]) if not item[0].startswith('__') and not callable(item[1]) else None

@dataclass
class FoldingPaddingTypes:
    # how we replace natural padded tokens in the folding interpolation
    INTERPOLATE = "interpolate"  # Interpolate between masked and full embeddings with folding strategies
    REPLACE = "replace"  # Rigidly replace padded tokens with folded embeddings
    GAPPED = "gapped"  # Use a gapped approach, where we leave gaps in the output for padded tokens for spacing
    SPARSE = "sparse"  # Use a sparse approach, where we only fill in non-padded tokens and leave others empty
    NONE = "none"  # No padding replacement, leave padded tokens as is

FOLDING_PADDING_TYPES = []
for item in FoldingPaddingTypes.__dict__.items():
    FOLDING_PADDING_TYPES.append(item[1]) if not item[0].startswith('__') and not callable(item[1]) else None


@dataclass
class FoldingPoolingTypes:
    # these are applied to the entire pooled set of embeddings if multiple embeddings are provided
    # they are passed by [[start, end],...], giving a start and end index for each embedding to omit or pool
    TRIANGULAR = "triangular_overlap"  # Triangular pooling, where we pool embeddings in a triangular fashion
    SLERP = "slerp"  # Spherical linear interpolation pooling
    AVERAGE = "average"  # Average pooling across embeddings
    MAX = "max"  # Max pooling across embeddings
    SUM = "sum"  # Sum pooling across embeddings
    NONE = "none"  # No pooling, return embeddings as is, default behavior for preliminary testing

FOLDING_POOLING_TYPES = []
for item in FoldingPoolingTypes.__dict__.items():
    FOLDING_POOLING_TYPES.append(item[1]) if not item[0].startswith('__') and not callable(item[1]) else None