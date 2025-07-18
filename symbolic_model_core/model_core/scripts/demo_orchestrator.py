from model_core.framework.encoder_orchestrator import EncoderOrchestrator
from model_core.encoders.registry import register_all_encoders

orch = EncoderOrchestrator()
register_all_encoders(orch, device="cpu")

prompt = "A glowing artifact resting atop a cracked altar."
outputs = orch.encode(prompt)

for name, tensor in outputs.items():
    print(f"[{name}] output shape: {tensor.shape}")
    encoder = orch.get_encoder(name)
    encoder.print_memory_report()
