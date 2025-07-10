"""
    ABS Shunt Suite - Advanced Bridging System Adapters
    Author: AbstractPhil

    Description: A suite of advanced shunt adapters for ComfyUI, designed to enhance and bridge the capabilities of CLIP
    with various other forms of embeddings; primarily BERT, T5, LLAMA, and other text-based models.

    A shunt is a form of cross-analytical adapter trained to bridge similar symbolic representations between entirely
    different models that produce different types of embeddings with different logical structures.

    These cross-analytical adapters form similarity bridges between different models, allowing for the transfer of knowledge
    that would otherwise be lost in translation.

    This suite includes nodes for loading, conditioning, and visualizing shunt adapters, as well as utilities for managing
    shunt models and their configurations.

    License: Apache License 2.0
"""
import comfy
import logging
logger = logging.getLogger(__name__)

from .node.encoder_nodes import (
    # Importing all necessary nodes for the ABS Shunt Suite
    SimpleEncoderLoader,
    EncoderLoader,
    T5LoaderTest,
    EncoderSamplerConfig,
    # Clip-based swap and handling nodes
    EncoderSampler
)


from .node.shunt_nodes import (
    ShuntConditioning,
    ShuntConditioningAdvanced,
    StackShuntAdapters,
    LoadAdapterShunt,
    LoadShuntSimple,
    ListLoadedShuntModels,
    UnloadShuntModels,
    MergeShunts,
    ShuntScheduler,
    VisualizeShuntEffect,
    SimpleShuntSetup,
    EasyShunt,
    QuickShuntPreview,
    ShuntStrengthTest,
    SuperiorConditioningPreview,
)

from .node.clip_nodes import (
    #ClipTokenizerSwap, no longer needed
    AbsClipSplitter,
    ACLIPLoader,
    ADualCLIPLoader,
    ATripleCLIPLoader,
    AQuadrupleCLIPLoader,
)

from .node.general_nodes import (
    ABS_PromptNode,
    ABS_ConcatPrompts,
    ABS_DebugNode
)

from .node.huggingface_nodes import (
    # Importing HuggingFace nodes for additional functionality
    SetHuggingfaceToken,
    SetHuggingfaceCacheDirectory
)

from .node.model_sampling_nodes import (
    AModelSamplingDiscrete
)

from .node.diffusion_nodes import (
    ALoadCheckpointSimple,  # Node for loading a simple diffusion-based checkpoint model
)

NODE_CLASS_MAPPINGS = {

    # Checkpoint Nodes
    "ALoadCheckpointSimple": ALoadCheckpointSimple,  # Node for loading a simple diffusion-based checkpoint model

    # Sampler nodes
    "EncoderSamplerConfig": EncoderSamplerConfig,  # Configuration node for sampling encoders
    "EncoderSampler": EncoderSampler,  # Sampler node for encoders
    "AModelSamplingDiscrete": AModelSamplingDiscrete,  # Discrete sampling node for model outputs

    # Shunt adapter loading and management nodes
    "SimpleEncoderLoader": SimpleEncoderLoader, # simplified loader for shunt adapters
    "EncoderLoader": EncoderLoader, # advanced loader for shunt adapters with many more options
    "LoadShuntSimple": LoadShuntSimple, # Loads a simple adapter shunt model to translate embeddings
    "LoadAdapterShunt": LoadAdapterShunt, # Loads a complex adapter model with advanced options

    # Multi-shunt management nodes
    "StackShuntAdapters": StackShuntAdapters,
    "MergeShunts": MergeShunts,
    "ShuntScheduler": ShuntScheduler,
    "UnloadShuntModels": UnloadShuntModels,


    # Pass through nodes for shunt conditioning
    "ShuntConditioning": ShuntConditioning,
    "ShuntConditioningAdvanced": ShuntConditioningAdvanced,

    # Convenience nodes set up with default configurations for quick shunt usage
    # # Primarily intended for quick setup and testing of shunt adapters
    "SimpleShuntSetup": SimpleShuntSetup,
    "EasyShunt": EasyShunt,

    # Visualization and testing nodes
    "QuickShuntPreview": QuickShuntPreview,
    "ShuntStrengthTest": ShuntStrengthTest,
    "SuperiorConditioningPreview": SuperiorConditioningPreview,
    "ListLoadedShuntModels": ListLoadedShuntModels,
    "VisualizeShuntEffect": VisualizeShuntEffect,

    # General utility nodes
    "Prompt": ABS_PromptNode,
    "ConcatPrompts": ABS_ConcatPrompts,
    "ADebugNode": ABS_DebugNode,  # A debug node for testing and debugging purposes

    # Clip-based nodes
    "ACLIPLoader": ACLIPLoader,               # Loads a dual CLIP model (clip-l, clip-g)
    "ADualCLIPLoader": ADualCLIPLoader,       # Loads a dual CLIP model (clip-l, clip-g)
    "ATripleCLIPLoader": ATripleCLIPLoader,   # Loads a triple CLIP model (clip-l, clip-g, t5)
    "AQuadrupleCLIPLoader": AQuadrupleCLIPLoader, # Loads a quadruple CLIP model (clip-l, clip-g, t5, llama)
    #"ClipTokenizerSwap": ClipTokenizerSwap,     # added v0.4.0
    "AbsClipSplitter": AbsClipSplitter,         # added v0.4.0

    # HuggingFace nodes for additional functionality
    "SetHuggingfaceToken": SetHuggingfaceToken,  # Sets the Hugging Face token for private model access
    "SetHuggingfaceCacheDirectory": SetHuggingfaceCacheDirectory,  # Sets the Hugging Face cache directory for model storage

    # Deprecated nodes
    "T5LoaderTest": T5LoaderTest,  # deprecated, use EncoderLoader instead
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Checkpoint Nodes
    "ALoadCheckpointSimple": "📦 Load Simple Checkpoint",  # Node for loading a simple diffusion-based checkpoint model

    # Sampler nodes
    "EncoderSamplerConfig": "🎛️ Encoder Sampler Config",  # Configuration node for sampling encoders
    "EncoderSampler": "🎲 Encoder Sampler",  # Sampler node for encoders
    "AModelSamplingDiscrete": "🎲 Discrete Model Sampling",  # Discrete sampling node for model outputs

    # Shunt adapter loading and management nodes
    "SimpleEncoderLoader": "🔍 Simple Encoder Loader",
    "EncoderLoader": "📦 Encoder Loader",
    "LoadShuntSimple": "🔄 Load Shunt Adapter Simple",
    "LoadAdapterShunt": "⚡ Load Shunt Adapter",

    # Multi-shunt management nodes
    "StackShuntAdapters": "📚 Stack Shunt Adapters",
    "MergeShunts": "🔀 Merge Shunt Adapters",
    "ShuntScheduler": "📊 Shunt Scheduler",
    "UnloadShuntModels": "🗑️ Unload Shunt Models",

    # Conditioning and passthrough nodes
    "ShuntConditioning": "🔌 Shunt Conditioning",
    "ShuntConditioningAdvanced": "🎛️ Shunt Conditioning Advanced",

    # Convenience nodes for quick shunt setup
    "SimpleShuntSetup": "🔧 Simple Shunt Setup",
    "EasyShunt": "🚀 Easy Shunt",

    # Preview nodes
    "QuickShuntPreview": "👁️ Quick Shunt Preview",
    "ShuntStrengthTest":  "🧪 Shunt Strength Test",
    "SuperiorConditioningPreview": "🌟 Superior Conditioning Preview",
    "ListLoadedShuntModels":  "📋 List Loaded Shunt Models",
    "VisualizeShuntEffect": "📈 Visualize Shunt Effect",

    # General utility nodes
    "Prompt": "📝 Simple Prompt Node",
    "ConcatPrompts": "🔗 Concatenate Prompts",
    "ADebugNode": "🐞 Debug Node",  # A debug node for testing and debugging purposes

    # Clip-based nodes
    "AQuadrupleCLIPLoader": "📦 Quadruple CLIP Loader",  # Loads a quadruple CLIP model (clip-l, clip-g, t5, llama)
    "ACLIPLoader": "📦 A-CLIP Loader",  # Loads a dual CLIP model (clip-l, clip-g)
    "ADualCLIPLoader": "📦 Dual CLIP Loader",  # Loads a dual CLIP model (clip-l, clip-g)
    "ATripleCLIPLoader": "📦 Triple CLIP Loader",  # Loads a triple CLIP model (clip-l, clip-g, t5)
    "ClipTokenizerSwap": "🔄 Clip Tokenizer Swap",
    "AbsClipSplitter": "🔗 Abs Clip Splitter",

    # HuggingFace nodes for additional functionality
    "SetHuggingfaceToken": "🔑 Set Hugging Face Token",  # Sets the Hugging Face token for private model access
    "SetHuggingfaceCacheDirectory": "📂 Set Hugging Face Cache Directory",  # Sets the Hugging Face cache directory for model storage

    # deprecated nodes
    "T5LoaderTest": "🚀 T5 Encoder Loader",

}


# ASCII art banner
logger.info("""
╔══════════════════════════════════════════╗
║        🚀 ABS SHUNT SUITE 🚀            ║
║    Dev Advanced Bridging System Adapters ║
║         ⚡ Version 0.7.0 ⚡                ║
╚══════════════════════════════════════════╝
""")


try:
    import torch
except ImportError:
    logger.error("❌ PyTorch is not installed. Please install it to use ABS Shunt Suite.")
    raise
try:
    import psutil
except ImportError:
    logger.error("❌ psutil is not installed. Please install it to use ABS Shunt Suite.")
    raise
try:
    import platform
except ImportError:
    logger.error("❌ platform is not installed. Please install it to use ABS Shunt Suite.")
    raise

def print_system_summary():
    bar = "=" * 60
    logger.info(f"\n{bar}")
    logger.info(" ABS SYSTEM DIAGNOSTICS".center(60))
    logger.info(f"{bar}")

    # CPU
    cpu_count = psutil.cpu_count(logical=True)
    cpu_name = platform.processor() or "Unknown CPU"
    logger.info(f" CPU         : {cpu_name} ({cpu_count} cores)")

    # RAM
    ram_gib = psutil.virtual_memory().total / 1024 ** 3
    logger.info(f" System RAM  : {ram_gib:.2f} GiB")


    # GPU
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        logger.info(f" CUDA Devices: {count} detected\n")
        ct = 0
        for i in range(count):
            ct += 1
            logger.info(f" debug count: {count}")
            props = torch.cuda.get_device_properties(i)
            total_vram = props.total_memory / 1024 ** 3
            free_vram, _ = torch.cuda.mem_get_info(i)

            logger.info(f" [GPU {i}] {props.name}")
            logger.info(f"     Total VRAM : {total_vram:.2f} GiB")
            logger.info(f"     Free  VRAM : {free_vram / 1024 ** 3:.2f} GiB")

    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        logger.info(" MPS Device  : Apple Metal Performance Shaders")
        logger.info("              (VRAM metrics not available)")

    else:
        logger.info(" Accelerator : CPU-only (no CUDA or MPS available)")

    logger.info(f"{bar}\n")


print_system_summary()

logger.info("Loading ABS Shunt Suite...")
logger.info("✅ Shunt adapters initialized")
logger.info("⚡ Ready to bridge a multitude of embeddings")

__all__ = [NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS]
