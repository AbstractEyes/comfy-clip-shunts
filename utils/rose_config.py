# ----------------------------------------------------------------------
# ROSE Configuration Registry
# --------------------------------------------------------------------------
# Author: AbstractPhil
# Date: 07/31/2025
# license: Apache-2.0
#
# rose_config.py
# ----------------------------------------------------------------------
# Configuration registry for all ROSE-related modules.
# Each configuration object is paired to a corresponding component (e.g., Rose4D or RoseTensorBank)
# and controls computation behavior, resource management, and projection logic.

from dataclasses import dataclass, field
from typing import List, Optional
import torch

@dataclass
class RoseConfig:
    # --- Dimensional Role Control ---
    dimensions: int = 4                                  # Number of semantic roles (typically x, need, relation, purpose)
    role_mode: str = "standard"                          # Options: standard, inverse, scatter, singular (affects permutation logic)

    # --- Weighting ---
    weight_triads: float = 1.0                           # Direct cosine(x, need/relation/purpose) mean alignment
    weight_condensed: float = 1.0                        # Alignment to pairwise/condensed vector blends
    weight_entropy: float = 0.5                          # Scaled entropy as a signal complexity factor
    weight_magnitude: float = 0.25                       # Scaled vector magnitude for strength of signal
    weight_external_field: float = 1.0                   # Strength multiplier for external 5D stimulation

    # --- Tensor Handling ---
    clone_inputs: bool = True                            # Clone inputs to avoid in-place mutation
    clone_outputs: bool = True                           # Clone outputs for safe isolation
    return_full: bool = False                            # Return full score breakdown dict if True
    attempt_any: bool = False                            # Attempt partial evaluation when missing inputs (planned for RoseAny)
    decouple_cache: bool = True                          # Detach tensors after compute (e.g., for cleanup or offload)

    # --- ROPE Spiral Logic ---
    rope_spiral: bool = False                            # Enable positional encoding-based potential shaping (future)
    rope_potential_mode: str = "harmonic_std"            # Mode for interpreting spiral modulation (placeholder)
    rope_phase_offsets: Optional[List[int]] = field(default_factory=list)  # Phase offsets for advanced ROPE alignment

    # --- ROSE Spiral Logic ---
    rose_spiral: bool = False                            # Activate rose spiral embedding behavior (e.g., symbolic spiral routing)
    spiral_potential_mode: str = "harmonic_std"          # Logic for spiral generation or interpretation (future reserved)
    spiral_phase_offsets: Optional[torch.Tensor] = None  # Precomputed tensor spiral offsets for symbolic curves

    # --- Bias & Modulation Control ---
    anchor_bias_mode: Optional[str] = None               # Bias alignment from anchor vector (e.g., mean_bias, purpose_bias)
    bias_vector: Optional[torch.Tensor] = None           # Injected vector for intentional modulation (optional)
    noise_amplification: float = 0.0                     # Add controlled noise to x before projection (for robustness or spread)

    # --- Output Shaping ---
    use_normalized_delta: bool = False                   # Project x by normalized direction to condensed mean
    residual_output: bool = False                        # Return x, delta, and other diagnostics instead of rose only
    projection_mode: str = "alignment"                   # Determines transformation type: alignment, projection, folding, or skip
    rose_threshold: float = 0.25                         # Suppress output if rose < threshold (planned gate mechanism)

    # --- Seeded Behavior ---
    seed: Optional[int] = None                           # Set for deterministic noise generation or stimulation

    # --- Logging ---
    verbose: bool = False                                # Enable debug logs
    tag: Optional[str] = None                            # Optional named tag for tracking


@dataclass
class RoseTensorBankConfig:
    # --- Allocation Preference ---
    offload_order: List[str] = field(default_factory=lambda: ["cuda", "cpu", "disk"])  # Priority cascade
    force_cpu: bool = False                              # Forces CPU usage regardless of availability

    # --- Cache Directory ---
    cache_directory: Optional[str] = None                # Override disk cache directory
    clear_cache_on_exit: bool = True                     # Whether to delete on object destruction

    # --- Resource Limits ---
    max_vram_utilization: float = 0.8                    # Maximum VRAM threshold for allocation
    max_ram_utilization: float = 0.8                     # Same for RAM

    # --- Estimation ---
    enable_size_estimation: bool = True                  # Enable internal size estimation logic
    estimate_margin: float = 0.2                         # Estimation range padding
    track_allocation_deltas: bool = False                # Reserved for incremental tracking

    # --- Disk Handling ---
    allow_disk_offload: bool = True                      # Enable fallback to disk if VRAM/RAM constrained
    compress_disk_tensors: bool = False                  # Use zip-safe save/load (not yet active)

    # --- Logging ---
    verbose: bool = False                                # Enable debug output from tensor bank
    tag: Optional[str] = None                            # Tag name for cache sessions or experiments
