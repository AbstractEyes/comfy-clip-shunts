import torch
from model_core.implementation.dummy import DummyModel, DummyTokenizer
from model_core.framework.automodel import EncoderMetadata, BaseEncoder

def test_dummy_encode_single():
    encoder = BaseEncoder(
        metadata=EncoderMetadata(
            identifier="dummy",
            encoder_type="text",
            capabilities=["test"]
        ),
        device="cpu"
    )
    encoder.set_model(DummyModel(out_dim=4))
    encoder.set_tokenizer(DummyTokenizer())

    output = encoder.encode("hello test")
    assert isinstance(output, torch.Tensor)
    assert output.shape[1] == 4

def test_dummy_encode_batch():
    encoder = BaseEncoder(
        metadata=EncoderMetadata(
            identifier="dummy-batch",
            encoder_type="text",
            capabilities=["test"]
        ),
        device="cpu"
    )
    encoder.set_model(DummyModel(out_dim=8))
    encoder.set_tokenizer(DummyTokenizer())

    batch_output = encoder.encode(["hello", "world"])
    assert batch_output.shape == (2, 8)