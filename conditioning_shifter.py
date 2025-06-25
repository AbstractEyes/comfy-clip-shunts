import comfy
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AdapterOutput:
    """Container for adapter outputs"""
    anchor: torch.Tensor
    delta: torch.Tensor
    gate: torch.Tensor
    log_sigma: torch.Tensor
    tau: torch.Tensor
    g_pred: torch.Tensor
    attention_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    
    @property
    def device(self):
        return self.anchor.device
    
    @property
    def shape(self):
        return self.anchor.shape


@dataclass
class ShiftConfig:
    """Configuration for a shift operation"""
    strength: float = 1.0
    delta_mean: float = 0.0
    delta_scale: float = 1.0
    sigma_scale: float = 0.0
    gate_probability: float = 1.0
    gate_threshold: float = 0.1
    noise_injection: float = 0.0
    use_anchor: bool = True


class ConditioningShifter:
    """Static utility for semantic conditioning transformations"""
    
    @staticmethod
    def apply_adapter_output(
        clip_slice: torch.Tensor,
        adapter_output: AdapterOutput,
        config: ShiftConfig
    ) -> torch.Tensor:
        """
        Apply a single adapter output to a clip slice.
        This replaces the hardcoded logic in ShuntConditioning.
        """
        # Extract components
        anchor = adapter_output.anchor
        delta = adapter_output.delta
        gate = adapter_output.gate
        log_sigma = adapter_output.log_sigma
        
        # Scale and offset delta
        delta = delta * config.delta_scale + config.delta_mean
        
        # Process gate with threshold
        gate_scaled = gate * config.gate_probability
        gate_mask = (gate_scaled > config.gate_threshold).float()
        gate_masked = gate_scaled * gate_mask
        
        # Resize if needed
        if delta.shape[1] != clip_slice.shape[1]:
            delta = F.interpolate(
                delta.transpose(1, 2),
                size=clip_slice.size(1),
                mode="nearest"
            ).transpose(1, 2)
            
            gate_masked = F.interpolate(
                gate_masked.transpose(1, 2),
                size=clip_slice.size(1),
                mode="nearest"
            ).transpose(1, 2)
            
            if anchor.shape[1] != clip_slice.shape[1]:
                anchor = F.interpolate(
                    anchor.transpose(1, 2),
                    size=clip_slice.size(1),
                    mode="nearest"
                ).transpose(1, 2)
        
        # Apply strength
        delta_final = delta * config.strength
        
        # Apply modification based on mode
        if config.use_anchor:
            # Blend between original and modified anchor
            clip_modified = clip_slice * (1 - gate_masked) + (anchor + delta_final) * gate_masked
        else:
            # Simple additive modification
            clip_modified = clip_slice + (delta_final * gate_masked)
        
        # Apply noise if requested
        if config.sigma_scale > 0 and config.noise_injection > 0:
            sigma = torch.exp(log_sigma * config.sigma_scale)
            clip_modified += torch.randn_like(clip_modified) * sigma * config.noise_injection
        elif config.noise_injection > 0:
            clip_modified += torch.randn_like(clip_modified) * config.noise_injection
        
        return clip_modified
    
    @staticmethod
    def pool_adapter_outputs(
        adapter_outputs: List[Tuple[AdapterOutput, float]],
        method: str = "weighted_average"
    ) -> AdapterOutput:
        """
        Pool multiple adapter outputs into a single output.
        
        Args:
            adapter_outputs: List of (AdapterOutput, weight) tuples
            method: Pooling method - "weighted_average", "max", "rms"
        
        Returns:
            Pooled AdapterOutput
        """
        if not adapter_outputs:
            raise ValueError("No adapter outputs to pool")
        
        if len(adapter_outputs) == 1:
            return adapter_outputs[0][0]
        
        # Calculate total weight
        total_weight = sum(weight for _, weight in adapter_outputs)
        if total_weight == 0:
            total_weight = len(adapter_outputs)
        
        # Initialize pooled tensors with zeros
        first_output = adapter_outputs[0][0]
        pooled = {
            'anchor': torch.zeros_like(first_output.anchor),
            'delta': torch.zeros_like(first_output.delta),
            'gate': torch.zeros_like(first_output.gate),
            'log_sigma': torch.zeros_like(first_output.log_sigma),
            'tau': torch.zeros_like(first_output.tau) if first_output.tau is not None else None,
            'g_pred': torch.zeros_like(first_output.g_pred) if first_output.g_pred is not None else None,
        }
        
        if method == "weighted_average":
            for output, weight in adapter_outputs:
                norm_weight = weight / total_weight
                pooled['anchor'] += output.anchor * norm_weight
                pooled['delta'] += output.delta * norm_weight
                pooled['gate'] += output.gate * norm_weight
                pooled['log_sigma'] += output.log_sigma * norm_weight
                if output.tau is not None and pooled['tau'] is not None:
                    pooled['tau'] += output.tau * norm_weight
                if output.g_pred is not None and pooled['g_pred'] is not None:
                    pooled['g_pred'] += output.g_pred * norm_weight
                    
        elif method == "max":
            # Take maximum activation
            for output, _ in adapter_outputs:
                pooled['anchor'] = torch.maximum(pooled['anchor'], output.anchor)
                pooled['delta'] = torch.maximum(pooled['delta'], output.delta)
                pooled['gate'] = torch.maximum(pooled['gate'], output.gate)
                pooled['log_sigma'] = torch.maximum(pooled['log_sigma'], output.log_sigma)
                
        elif method == "rms":
            # Root mean square pooling
            for output, weight in adapter_outputs:
                norm_weight = weight / total_weight
                pooled['anchor'] += (output.anchor ** 2) * norm_weight
                pooled['delta'] += (output.delta ** 2) * norm_weight
                pooled['gate'] += (output.gate ** 2) * norm_weight
                pooled['log_sigma'] += (output.log_sigma ** 2) * norm_weight
            
            pooled['anchor'] = torch.sqrt(pooled['anchor'])
            pooled['delta'] = torch.sqrt(pooled['delta'])
            pooled['gate'] = torch.sqrt(pooled['gate'])
            pooled['log_sigma'] = torch.sqrt(pooled['log_sigma'])
        
        else:
            raise ValueError(f"Unknown pooling method: {method}")
        
        return AdapterOutput(
            anchor=pooled['anchor'],
            delta=pooled['delta'],
            gate=pooled['gate'],
            log_sigma=pooled['log_sigma'],
            tau=pooled['tau'],
            g_pred=pooled['g_pred']
        )
    
    @staticmethod
    def shift_conditioning_tensor(
        conditioning_tensor: torch.Tensor,
        adapter_outputs_by_type: Dict[str, List[Tuple[AdapterOutput, float]]],
        config: ShiftConfig,
        pool_method: str = "weighted_average"
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply all adapter outputs to a conditioning tensor.
        
        Args:
            conditioning_tensor: The full conditioning tensor [batch, seq, dims]
            adapter_outputs_by_type: Dict mapping 'clip_l' or 'clip_g' to list of (output, weight)
            config: Shift configuration
            pool_method: How to pool multiple adapters of same type
            
        Returns:
            Modified conditioning tensor and statistics
        """
        modified_tensor = conditioning_tensor.clone()
        stats = {}
        
        for adapter_type, outputs in adapter_outputs_by_type.items():
            if not outputs:
                continue
                
            # Determine slice range
            if adapter_type == 'clip_l':
                slice_start, slice_end = 0, 768
            elif adapter_type == 'clip_g':
                total_dim = conditioning_tensor.size(-1)
                slice_start = 768
                slice_end = min(2048, total_dim)
            else:
                logger.warning(f"Unknown adapter type: {adapter_type}")
                continue
            
            # Extract slice
            clip_slice = modified_tensor[:, :, slice_start:slice_end]
            
            # Pool outputs if multiple
            if len(outputs) > 1:
                pooled_output = ConditioningShifter.pool_adapter_outputs(outputs, pool_method)
                logger.info(f"Pooled {len(outputs)} {adapter_type} adapters using {pool_method}")
            else:
                pooled_output = outputs[0][0]
            
            # Apply the pooled output
            clip_modified = ConditioningShifter.apply_adapter_output(
                clip_slice,
                pooled_output,
                config
            )
            
            # Update tensor
            modified_tensor[:, :, slice_start:slice_end] = clip_modified.type_as(conditioning_tensor)
            
            # Collect stats
            with torch.no_grad():
                gate_mean = pooled_output.gate.mean().item()
                delta_magnitude = pooled_output.delta.abs().mean().item()
                
            stats[adapter_type] = {
                'gate_mean': gate_mean,
                'delta_magnitude': delta_magnitude,
                'slice_range': (slice_start, slice_end),
                'num_adapters': len(outputs)
            }
        
        return modified_tensor, stats
    
    @staticmethod
    def create_adapter_output(
        adapter_model: torch.nn.Module,
        encoder_embeddings: torch.Tensor,
        clip_embeddings: torch.Tensor,
        guidance_scale: float = 10.0
    ) -> AdapterOutput:
        """
        Run adapter forward pass and package outputs.
        
        Args:
            adapter_model: The adapter model
            encoder_embeddings: T5/BERT embeddings
            clip_embeddings: CLIP embeddings slice
            guidance_scale: Guidance scale for generation
            
        Returns:
            Packaged AdapterOutput
        """
        gen_config = {"max_guidance": guidance_scale}
        
        with torch.no_grad():
            outputs = adapter_model(
                encoder_embeddings.float(),
                clip_embeddings.float(),
                config=gen_config
            )
        
        if isinstance(outputs, tuple) and len(outputs) == 8:
            anchor, delta, log_sigma, attn_c2m, attn_m2c, tau, g_pred, gate = outputs
        else:
            raise ValueError(f"Unexpected adapter output format: {type(outputs)}")
        
        return AdapterOutput(
            anchor=anchor,
            delta=delta,
            gate=gate,
            log_sigma=log_sigma,
            tau=tau,
            g_pred=g_pred,
            attention_weights=(attn_c2m, attn_m2c)
        )
    
    # High-level semantic operations (to be implemented)
    
    @staticmethod
    def shift_towards_concepts(
        conditioning: torch.Tensor,
        adapter_outputs: Dict[str, List[Tuple[AdapterOutput, float]]],
        strength: float = 1.0
    ) -> torch.Tensor:
        """Shift conditioning toward sparse conceptual representation"""
        config = ShiftConfig(
            strength=strength,
            use_anchor=False,  # Pure delta application
            gate_threshold=0.15  # Higher threshold for concept focus
        )
        modified, _ = ConditioningShifter.shift_conditioning_tensor(
            conditioning, adapter_outputs, config
        )
        return modified
    
    @staticmethod
    def shift_towards_style(
        conditioning: torch.Tensor,
        adapter_outputs: Dict[str, List[Tuple[AdapterOutput, float]]],
        strength: float = 1.0
    ) -> torch.Tensor:
        """Preserve linguistic/stylistic elements"""
        config = ShiftConfig(
            strength=-strength,  # Negative strength to reduce concepts
            use_anchor=True,  # Use anchor for stability
            gate_threshold=0.05  # Lower threshold to affect more tokens
        )
        modified, _ = ConditioningShifter.shift_conditioning_tensor(
            conditioning, adapter_outputs, config
        )
        return modified