import os
import comfy
import logging
import torch

# This is the ram-safe vram-safe version of the conditioning cache.
# This is akin to throwing the kitchen sink at the problem, covering everything simultaneously.
# In the same light, it will not be optimized for speed, but rather for memory efficiency to prevent OOM errors.

from typing import List, Optional, Dict, Any

class ConditioningCache:

    def __init__(self):
        self.cache: BinaryTree = BinaryTree()

    def clear(self):
        """Clear the entire cache."""
        self.cache.clear()
        self.cond_cache.clear()
        self.cond_meta.clear()
        self.cond_bank.clear()
        self.cond_bank_meta.clear()
        self.cond_bank_bit_matrix = None
        self.cond_bank_token2col.clear()