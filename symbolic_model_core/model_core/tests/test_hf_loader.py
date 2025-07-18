import pytest
from model_core.implementation.hf_loader import load_hf_encoder, HFTokenizerWrapper

def test_load_hf_encoder_cpu():
    model, tokenizer = load_hf_encoder("sshleifer/tiny-distilbert-base-cased", device="cpu")
    assert model is not None
    assert tokenizer is not None

    wrapped = HFTokenizerWrapper(tokenizer)
    result = wrapped.tokenize("test")
    assert "input_ids" in result