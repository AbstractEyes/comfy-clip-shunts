from .nodes import (
    T5LoaderTest,
    ShuntConditioning,
    ShuntConditioningAdvanced,
    StackShuntAdapters,
    LoadAdapterShunt,
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

NODE_CLASS_MAPPINGS = {
    "T5LoaderTest": T5LoaderTest,
    "LoadAdapterShunt": LoadAdapterShunt,
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
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "T5LoaderTest": "🚀 T5 Encoder Loader",
    "LoadAdapterShunt": "⚡ Load Shunt Adapter",
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
}

# ASCII art banner
print("""
╔══════════════════════════════════════════╗
║        🚀 ABS SHUNT SUITE 🚀             ║
║    Advanced Bridging System Adapters     ║
║         ⚡ Version 0.1.0 ⚡               ║
╚══════════════════════════════════════════╝
""")

print("Loading ABS Shunt Suite...")
print("✅ Shunt adapters initialized")
print("⚡ Ready to bridge T5 → CLIP embeddings")

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
