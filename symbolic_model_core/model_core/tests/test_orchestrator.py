from model_core.framework.automodel import EncoderMetadata
from model_core.framework.encoder_orchestrator import EncoderOrchestrator
from model_core.implementation.dummy import DummyModel, DummyTokenizer
from model_core.framework.automodel import BaseEncoder

def test_orchestrator_single_encoder():
    orchestrator = EncoderOrchestrator()
    encoder = BaseEncoder(
        metadata=EncoderMetadata(identifier="dummy", encoder_type="text", capabilities=[]),
        device="cpu"
    )
    encoder.set_model(DummyModel(out_dim=4))
    encoder.set_tokenizer(DummyTokenizer())

    orchestrator.add_encoder("dummy", encoder)
    output = orchestrator.encode("symbolic")
    assert "dummy" in output
    assert output["dummy"].shape[1] == 4