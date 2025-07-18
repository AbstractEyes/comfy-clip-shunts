"""
Minimal runnable bootstrap for the new encoder stack.
 - creates a DummyTokenizer / DummyModel so `.encode()` never crashes
 - registers a tiny HuggingFace loader (cpu-only for now)
 - wires the TextEncoder subclass into EncoderRegistry
"""

import torch
import torch.nn as nn
from typing import List, Union, Any, Dict
from dataclasses import dataclass, field
from .automodel import (
    EncoderMetadata, ITokenizer, BaseEncoder, EncoderRegistry
)

import logging
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# 1.  Dummy components for smoke-tests
# ──────────────────────────────────────────────────────────

class DummyTokenizer(ITokenizer):
    """Dummy tokenizer that returns tensor based on string length"""

    def tokenize(self, inputs: Union[str, List[str]]) -> torch.Tensor:
        # Handle both single string and list of strings
        texts = inputs if isinstance(inputs, list) else [inputs]
        # Return tensor with length of each text
        return torch.tensor([[len(text)] for text in texts], dtype=torch.long)

    def detokenize(self, tokens: Any) -> Union[str, List[str]]:
        # Return dummy strings for each token
        batch_size = tokens.shape[0] if hasattr(tokens, 'shape') else 1
        result = ["<dummy_text>" for _ in range(batch_size)]
        return result[0] if batch_size == 1 else result


class DummyModel(nn.Module):
    """Dummy model that projects input to specified dimension"""

    def __init__(self, out_dim: int = 4):
        super().__init__()
        self.out_dim = out_dim
        self.proj = nn.Linear(1, out_dim, bias=False)
        # Initialize with some non-zero weights for testing
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is float and has right shape
        if x.dim() == 1:
            x = x.unsqueeze(1)
        return self.proj(x.float())


# ──────────────────────────────────────────────────────────
# 2.  HuggingFace loader (cpu-only for now)
# ──────────────────────────────────────────────────────────

def load_hf_encoder(repo_id: str, device: str = "cpu") -> tuple:
    """Load a HuggingFace model and tokenizer

    Args:
        repo_id: HuggingFace repository ID (e.g., "openai/clip-vit-base-patch32")
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        from transformers import AutoModel, AutoTokenizer

        logger.info(f"Loading HuggingFace model: {repo_id}")
        model = AutoModel.from_pretrained(
            repo_id,
            device_map={"": device},
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)

        return model, tokenizer
    except ImportError:
        logger.error("transformers library not installed")
        raise
    except Exception as e:
        logger.error(f"Failed to load model {repo_id}: {e}")
        raise


# ──────────────────────────────────────────────────────────
# 3.  Concrete TextEncoder implementation
# ──────────────────────────────────────────────────────────

class TextEncoder(BaseEncoder):
    """Text encoder that can use dummy or real HuggingFace models"""

    def __init__(self,
                 name: str,
                 dummy: bool = True,
                 repo_id: str = "",
                 device: str = "cpu",
                 **kwargs):
        # Create metadata
        metadata = EncoderMetadata(
            identifier=name,
            encoder_type="text",
            version="1.0",
            capabilities=["text_encoding"],
            config={
                "dummy": dummy,
                "repo_id": repo_id,
                **kwargs
            }
        )

        # Initialize base encoder
        super().__init__(metadata, device=device)

        # Set up model and tokenizer
        if dummy:
            logger.info(f"Creating dummy encoder: {name}")
            self.set_model(DummyModel())
            self.set_tokenizer(DummyTokenizer())
        else:
            if not repo_id:
                raise ValueError("repo_id required when dummy=False")
            logger.info(f"Creating HuggingFace encoder: {name} from {repo_id}")
            model, tokenizer = load_hf_encoder(repo_id, device)
            self.set_model(model)
            # Wrap HF tokenizer to match our interface
            self.set_tokenizer(HFTokenizerWrapper(tokenizer))

        # Register output reshaping hook
        self._setup_output_hook()

    def _setup_output_hook(self):
        """Setup hook to ensure output is properly shaped"""

        @self.hooks.register_hook("reshape_output")
        def reshape_output(output):
            # Handle different output types from models
            if isinstance(output, torch.Tensor):
                # Ensure 2D output [batch, features]
                if output.dim() == 3:
                    # Pool sequence dimension (e.g., for transformers)
                    output = output.mean(dim=1)
                elif output.dim() == 1:
                    # Add batch dimension
                    output = output.unsqueeze(0)
                return output
            elif hasattr(output, 'last_hidden_state'):
                # Handle transformers output
                hidden = output.last_hidden_state
                return hidden.mean(dim=1)  # Mean pooling
            elif hasattr(output, 'pooler_output'):
                # Use pooler output if available
                return output.pooler_output
            else:
                logger.warning(f"Unknown output type: {type(output)}")
                return output

        # Add to post-forward stage
        self.hooks.add_hook("post_forward", "reshape_output", priority=10)


class HFTokenizerWrapper(ITokenizer):
    """Wrapper to adapt HuggingFace tokenizer to our interface"""

    def __init__(self, hf_tokenizer):
        self.tokenizer = hf_tokenizer

    def tokenize(self, inputs: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        # Use HF tokenizer
        return self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

    def detokenize(self, tokens: Any) -> Union[str, List[str]]:
        # Decode tokens back to text
        if isinstance(tokens, dict):
            tokens = tokens.get('input_ids', tokens)

        decoded = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        return decoded[0] if len(decoded) == 1 else decoded


# ──────────────────────────────────────────────────────────
# 4.  Register encoder type
# ──────────────────────────────────────────────────────────

# Register so EncoderRegistry can create by string
EncoderRegistry.register("text", TextEncoder)


# ──────────────────────────────────────────────────────────
# 5.  Bootstrap test function
# ──────────────────────────────────────────────────────────

def boot():
    """Run smoke test of the encoder system"""
    try:
        logger.info("Starting encoder bootstrap test...")

        # Test 1: Create dummy encoder
        logger.info("Test 1: Creating dummy encoder")
        dummy_enc = EncoderRegistry.create("text", "dummy-unit", dummy=True)

        # Test single string
        logger.info("Test 2: Encoding single string")
        out_single = dummy_enc.encode("hello comfy")
        logger.info(f"Single encoding shape: {out_single.shape} | Sample values: {out_single[0][:4].tolist()}")

        # Test batch of strings
        logger.info("Test 3: Encoding batch of strings")
        out_batch = dummy_enc.encode(["hello", "comfy", "ui"])
        logger.info(f"Batch encoding shape: {out_batch.shape}")

        # Test memory tracking
        logger.info("Test 4: Memory tracking")
        total_size = dummy_enc.memory.get_total_size()
        loaded_size = dummy_enc.memory.get_device_usage(dummy_enc.device)
        logger.info(f"Total model size: {total_size} bytes, Loaded size: {loaded_size} bytes")

        # Test hooks
        logger.info("Test 5: Hook system")
        active_hooks = dummy_enc.hooks.list_stages()
        logger.info(f"Available hook stages: {len(active_hooks)}")

        # Optional: Test real model (commented out to avoid dependencies)
        # logger.info("Test 6: Real HuggingFace model")
        # real_enc = EncoderRegistry.create(
        #     "text",
        #     "clip-text",
        #     dummy=False,
        #     repo_id="openai/clip-vit-base-patch32"
        # )
        # out_real = real_enc.encode("a photo of a cat")
        # logger.info(f"Real encoding shape: {out_real.shape}")

        logger.info("✓ All tests passed! Encoder system is working correctly.")

        # Return encoder for further testing if needed
        return dummy_enc

    except Exception as e:
        logger.error(f"Bootstrap test failed: {e}", exc_info=True)
        raise


# Allow running as standalone script
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    boot()