import numpy as np
from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class ConditioningSchedulerTypes:
    TAU = "tau"
    TOP_K = "top_k"
    TOP_20K = "top_20k"
    TOP_50K = "top_50k"
    COS = "cos"
    SINE = "sine"
    COSINE = "cosine"
    WAVE = "wave"
    PULSE = "pulse"
    SHOCKWAVE = "shockwave"
    CASCADE = "cascade"
    NONE = "none"  # Default mode


@dataclass
class FoldingTypes:
    """
        "rigid", "zeus", "helios", "surge", "surge-fold", "fold", "interpolate",
        "collapse", "zipper", "concat-flatten", "cascade", "ripple"
    """
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
    SURGE_FOLD = "surge-fold"

@dataclass
class FoldingPaddingTypes:
    # how we replace natural padded tokens in the folding interpolation
    NONE = "none"  # No padding replacement, leave padded tokens as is
    REPLACE = "replace"  # Rigidly replace padded tokens with folded embeddings
    INTERPOLATE = "interpolate"  # Interpolate between masked and full embeddings with folding strategies
    GAPPED = "gapped"  # Use a gapped approach, where we leave gaps in the output for padded tokens for spacing
    SPARSE = "sparse"  # Use a sparse approach, where we only fill in non-padded tokens and leave others empty

@dataclass
class FoldingPoolingTypes:
    # these are applied to the entire pooled set of embeddings if multiple embeddings are provided
    # they are passed by [[start, end],...], giving a start and end index for each embedding to omit or pool
    NONE = "none"  # No pooling, return embeddings as is, default behavior for preliminary testing
    AVERAGE = "average"  # Average pooling across embeddings
    MAX = "max"  # Max pooling across embeddings
    SUM = "sum"  # Sum pooling across embeddings
