#from .general_nodes import *
from .subject import csv_handler
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
    SuperiorConditioningPreview
)

from .node.general_nodes import (
    ABS_PromptNode,
    ABS_ConcatPrompts,
)

NODE_CLASS_MAPPINGS = {
    "SimpleEncoderLoader": SimpleEncoderLoader,
    "EncoderLoader": EncoderLoader,
    "T5LoaderTest": T5LoaderTest,
    "LoadAdapterShunt": LoadAdapterShunt,
    "LoadShuntSimple": LoadShuntSimple,
    "ShuntConditioning": ShuntConditioning,
    "ShuntConditioningAdvanced": ShuntConditioningAdvanced,
    "StackShuntAdapters": StackShuntAdapters,
    "ListLoadedShuntModels": ListLoadedShuntModels,
    "UnloadShuntModels": UnloadShuntModels,
    "MergeShunts": MergeShunts,
    "ShuntScheduler": ShuntScheduler,
    "VisualizeShuntEffect": VisualizeShuntEffect,

    "SimpleShuntSetup": SimpleShuntSetup,
    "EasyShunt": EasyShunt,
    "QuickShuntPreview": QuickShuntPreview,
    "ShuntStrengthTest": ShuntStrengthTest,
    "SuperiorConditioningPreview": SuperiorConditioningPreview,
    "Prompt": ABS_PromptNode,
    "ConcatPrompts": ABS_ConcatPrompts,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleEncoderLoader": "🔍 Simple Encoder Loader",
    "EncoderLoader": "📦 Encoder Loader",
    "T5LoaderTest": "🚀 T5 Encoder Loader",
    "LoadAdapterShunt": "⚡ Load Shunt Adapter",
    "LoadShuntSimple": "🔄 Load Shunt Adapter Simple",
    "ShuntConditioning": "🔌 Shunt Conditioning",
    "ShuntConditioningAdvanced": "🎛️ Shunt Conditioning Advanced",
    "StackShuntAdapters": "📚 Stack Shunt Adapters",
    "ListLoadedShuntModels": "📋 List Loaded Shunts",
    "UnloadShuntModels": "🗑️ Unload Shunt Models",
    "MergeShunts": "🔀 Merge Shunt Adapters",
    "ShuntScheduler": "📊 Shunt Scheduler",
    "VisualizeShuntEffect": "📈 Visualize Shunt Effect",

    "SimpleShuntSetup": "✨ Simple Shunt Setup",
    "EasyShunt": "🎯 Easy Shunt",
    "QuickShuntPreview": "👁️ Quick Shunt Preview",
    "ShuntStrengthTest": "🧪 Shunt Strength Test",
    "SuperiorConditioningPreview": "🌟 Superior Conditioning Preview",

    "Prompt": "📝 Simple Prompt Node",
    "ConcatPrompts": "🔗 Concatenate Prompts",

}


# ASCII art banner
print("""
╔══════════════════════════════════════════╗
║        🚀 ABS SHUNT SUITE 🚀            ║
║    Dev Advanced Bridging System Adapters ║
║         ⚡ Version 0.3.0 ⚡                ║
╚══════════════════════════════════════════╝
""")

print("Loading ABS Shunt Suite...")
print("✅ Shunt adapters initialized")
print("⚡ Ready to bridge T5 → CLIP embeddings")

__all__ = [NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS]
