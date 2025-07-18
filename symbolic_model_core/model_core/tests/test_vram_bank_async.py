import asyncio
import torch
from model_core.implementation.vram_bank import VramBank
from model_core.implementation.dummy import DummyModel, DummyTokenizer
from model_core.framework.automodel import EncoderMetadata, BaseEncoder


async def simulate_encoder(name, out_dim, size, bank: VramBank, device="cpu"):
    encoder = BaseEncoder(
        metadata=EncoderMetadata(identifier=name, encoder_type="text", capabilities=[]),
        device=device
    )
    encoder.set_model(DummyModel(out_dim=out_dim))
    encoder.set_tokenizer(DummyTokenizer())

    # Simulate layer memory sizes and queue for each
    for i, (lname, module) in enumerate(encoder.model.named_modules()):
        if len(list(module.parameters(recurse=False))) == 0:
            continue

        # Fake size for each layer for demonstration
        layer_size = size if i == 0 else size // 2
        await bank.request_layer(
            encoder_name=name,
            layer_name=f"{name}.layer{i}",
            layer_size=layer_size,
            device=device,
            callback=lambda mod=module: mod.to(device)
        )
    return encoder

async def run_vram_test():
    bank = VramBank(max_vram_mb=40)  # 40MB VRAM cap for test

    encoders = [
        simulate_encoder("small-a", out_dim=2, size=10_000_000, bank=bank),
        simulate_encoder("small-b", out_dim=2, size=5_000_000, bank=bank),
        simulate_encoder("medium-c", out_dim=2, size=20_000_000, bank=bank),
    ]

    await asyncio.gather(*encoders)

    await asyncio.sleep(1)  # let queue cycle
    bank.print_report()


#asyncio.run(run_vram_test())