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
from .node.nodes import (
    # Importing all necessary nodes for the ABS Shunt Suite
    SimpleEncoderLoader,
    EncoderLoader,
    T5LoaderTest,
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
    # Clip-based swap and handling nodes
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
)

NODE_CLASS_MAPPINGS = {

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

    # Clip-based nodes
    "ACLIPLoader": ACLIPLoader,               # Loads a dual CLIP model (clip-l, clip-g)
    "ADualCLIPLoader": ADualCLIPLoader,       # Loads a dual CLIP model (clip-l, clip-g)
    "ATripleCLIPLoader": ATripleCLIPLoader,   # Loads a triple CLIP model (clip-l, clip-g, t5)
    "AQuadrupleCLIPLoader": AQuadrupleCLIPLoader, # Loads a quadruple CLIP model (clip-l, clip-g, t5, llama)
    #"ClipTokenizerSwap": ClipTokenizerSwap,     # added v0.4.0
    "AbsClipSplitter": AbsClipSplitter,         # added v0.4.0

    # Deprecated nodes
    "T5LoaderTest": T5LoaderTest,  # deprecated, use EncoderLoader instead
}

NODE_DISPLAY_NAME_MAPPINGS = {
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

    # Clip-based nodes
    "AQuadrupleCLIPLoader": "📦 Quadruple CLIP Loader",  # Loads a quadruple CLIP model (clip-l, clip-g, t5, llama)
    "ACLIPLoader": "📦 A-CLIP Loader",  # Loads a dual CLIP model (clip-l, clip-g)
    "ADualCLIPLoader": "📦 Dual CLIP Loader",  # Loads a dual CLIP model (clip-l, clip-g)
    "ATripleCLIPLoader": "📦 Triple CLIP Loader",  # Loads a triple CLIP model (clip-l, clip-g, t5)
    "ClipTokenizerSwap": "🔄 Clip Tokenizer Swap",
    "AbsClipSplitter": "🔗 Abs Clip Splitter",

    # deprecated nodes
    "T5LoaderTest": "🚀 T5 Encoder Loader",

}


# ASCII art banner
print("""
╔══════════════════════════════════════════╗
║        🚀 ABS SHUNT SUITE 🚀            ║
║    Dev Advanced Bridging System Adapters ║
║         ⚡ Version 0.5.2 ⚡                ║
╚══════════════════════════════════════════╝
""")

print("Loading ABS Shunt Suite...")
print("✅ Shunt adapters initialized")
print("⚡ Ready to bridge a multitude of embeddings")

__all__ = [NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS]
