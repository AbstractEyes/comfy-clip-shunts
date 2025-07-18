import torch
import logging
from model_core.framework.encoder_orchestrator import EncoderOrchestrator
from model_core.encoders import register_all_encoders

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def inspect_vram(orch: EncoderOrchestrator):
    print("\n=== VRAM USAGE REPORT ===")
    for name in orch.list_encoders():
        encoder = orch.get_encoder(name)
        print(f"\n[{name.upper()}]")
        encoder.print_memory_report()

    print("\n[SUMMARY]")
    for name in orch.list_encoders():
        encoder = orch.get_encoder(name)
        mem = encoder.get_memory_report()
        print(f"{name:8}: {mem['total_size_mb']:.2f} MB, Layers: {mem['layer_count']}")

if __name__ == "__main__":
    orch = EncoderOrchestrator()
    register_all_encoders(orch, device="cpu")
    inspect_vram(orch)
