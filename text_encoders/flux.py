from comfy import (sd1_clip)
from comfy.text_encoders import hunyuan_video
from comfy.text_encoders import sd3_clip
from comfy import sd1_clip
from comfy import sdxl_clip
import comfy.model_management
from transformers import T5TokenizerFast
import torch
import os

import logging

logger = logging.getLogger(__name__)


class T5XXLModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, attention_mask=False, model_options={}):
        logger.info("Initializing T5XXLModel with options: {}".format(model_options))
        if model_options.get("unchained_t5", False):
            logger.info("Using unchained T5XXL text encoder")
            textmodel_json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "t5_config_unchained_xxl.json")
        else:
            logger.info("Using chained T5XXL text encoder")
            textmodel_json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "t5_config_xxl.json")
        t5xxl_scaled_fp8 = model_options.get("t5xxl_scaled_fp8", None)
        if t5xxl_scaled_fp8 is not None:
            model_options = model_options.copy()
            model_options["scaled_fp8"] = t5xxl_scaled_fp8

        model_options = {**model_options, "model_name": "t5xxl"}
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config=textmodel_json_config, dtype=dtype, special_tokens={"end": 1, "pad": 0}, model_class=comfy.text_encoders.t5.T5, enable_attention_masks=attention_mask, return_attention_masks=attention_mask, model_options=model_options)



class FluxTokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        self.clip_l = sd1_clip.SDTokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)
        self.t5xxl = sd3_clip.T5XXLTokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)

    def tokenize_with_weights(self, text:str, return_word_ids=False, **kwargs):
        out = {}
        out["l"] = self.clip_l.tokenize_with_weights(text, return_word_ids, **kwargs)
        out["t5xxl"] = self.t5xxl.tokenize_with_weights(text, return_word_ids, **kwargs)
        return out

    def untokenize(self, token_weight_pair):
        return self.clip_l.untokenize(token_weight_pair)

    def state_dict(self):
        return {}


class FluxClipModel(torch.nn.Module):
    def __init__(self, dtype_t5=None, device="cpu", dtype=None, model_options={}):
        super().__init__()
        dtype_t5 = comfy.model_management.pick_weight_dtype(dtype_t5, dtype, device)
        logger.info("Initializing FluxClipModel with options: {}".format(model_options))
        logger.info("First clip_l")
        self.clip_l = sd1_clip.SDClipModel(device=device, dtype=dtype, return_projected_pooled=False, model_options=model_options)
        logger.info("Then t5xxl")
        self.t5xxl = T5XXLModel(device=device, dtype=dtype_t5, model_options=model_options)
        self.dtypes = set([dtype, dtype_t5])

    def set_clip_options(self, options):
        self.clip_l.set_clip_options(options)
        self.t5xxl.set_clip_options(options)

    def reset_clip_options(self):
        self.clip_l.reset_clip_options()
        self.t5xxl.reset_clip_options()

    def encode_token_weights(self, token_weight_pairs):
        token_weight_pairs_l = token_weight_pairs["l"]
        token_weight_pairs_t5 = token_weight_pairs["t5xxl"]

        t5_out, t5_pooled = self.t5xxl.encode_token_weights(token_weight_pairs_t5)
        l_out, l_pooled = self.clip_l.encode_token_weights(token_weight_pairs_l)
        return t5_out, l_pooled

    def load_sd(self, sd):
        if "text_model.encoder.layers.1.mlp.fc1.weight" in sd:
            return self.clip_l.load_sd(sd)
        else:
            return self.t5xxl.load_sd(sd)

def flux_clip(dtype_t5=None, t5xxl_scaled_fp8=None, unchained_t5=False):
    class FluxClipModel_(FluxClipModel):
        def __init__(self, device="cpu", dtype=None, model_options={}, unchained_t5=unchained_t5):
            if t5xxl_scaled_fp8 is not None and "t5xxl_scaled_fp8" not in model_options:
                model_options = model_options.copy()
                model_options["t5xxl_scaled_fp8"] = t5xxl_scaled_fp8
            if unchained_t5:
                model_options = model_options.copy()
                model_options["unchained_t5"] = True
            else:
                model_options = model_options.copy()
                model_options["unchained_t5"] = False
            super().__init__(dtype_t5=dtype_t5, device=device, dtype=dtype, model_options=model_options)
    return FluxClipModel_
