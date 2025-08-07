"""
    Direct port from ComfyUI of the HiDream Text Encoder Model
"""
import os

from comfy.text_encoders import hunyuan_video
from comfy.text_encoders import sd3_clip
from comfy import sd1_clip
from comfy import sdxl_clip
import comfy.model_management
import torch
import logging


logger = logging.getLogger(__name__)

class T5XXLModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, attention_mask=False, model_options={}):
        if model_options.get("unchained_t5", False):
            textmodel_json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "t5_config_unchained_xxl.json")
        else:
            textmodel_json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "t5_config_xxl.json")
        t5xxl_scaled_fp8 = model_options.get("t5xxl_scaled_fp8", None)
        if t5xxl_scaled_fp8 is not None:
            model_options = model_options.copy()
            model_options["scaled_fp8"] = t5xxl_scaled_fp8

        model_options = {**model_options, "model_name": "t5xxl"}
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config=textmodel_json_config, dtype=dtype, special_tokens={"end": 1, "pad": 0}, model_class=comfy.text_encoders.t5.T5, enable_attention_masks=attention_mask, return_attention_masks=attention_mask, model_options=model_options)



class HiDreamTokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        self.clip_l = sd1_clip.SDTokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)
        self.clip_g = sdxl_clip.SDXLClipGTokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)
        self.t5xxl = sd3_clip.T5XXLTokenizer(embedding_directory=embedding_directory, min_length=128, max_length=128, tokenizer_data=tokenizer_data)
        self.llama = hunyuan_video.LLAMA3Tokenizer(embedding_directory=embedding_directory, min_length=128, pad_token=128009, tokenizer_data=tokenizer_data)

    def tokenize_with_weights(self, text:str, return_word_ids=False, **kwargs):
        out = {}
        out["g"] = self.clip_g.tokenize_with_weights(text, return_word_ids, **kwargs)
        out["l"] = self.clip_l.tokenize_with_weights(text, return_word_ids, **kwargs)
        t5xxl = self.t5xxl.tokenize_with_weights(text, return_word_ids, **kwargs)
        out["t5xxl"] = [t5xxl[0]]  # Use only first 128 tokens
        out["llama"] = self.llama.tokenize_with_weights(text, return_word_ids, **kwargs)
        return out

    def untokenize(self, token_weight_pair):
        return self.clip_g.untokenize(token_weight_pair)

    def state_dict(self):
        return {}



class HiDreamTEModel(torch.nn.Module):
    def __init__(self, clip_l=True, clip_g=True, t5=True, llama=True, dtype_t5=None, dtype_llama=None, device="cpu", dtype=None, model_options={}):
        super().__init__()
        self.dtypes = set()
        if clip_l:
            logger.debug("Using SD1 Clip-L text encoder from hidream TEModel")
            self.clip_l = sd1_clip.SDClipModel(device=device, dtype=dtype, return_projected_pooled=True, model_options=model_options)
            self.dtypes.add(dtype)
        else:
            self.clip_l = None

        if clip_g:
            logger.debug("Using SDXL Clip-G text encoder from hidream TEModel")
            self.clip_g = sdxl_clip.SDXLClipG(device=device, dtype=dtype, model_options=model_options)
            self.dtypes.add(dtype)
        else:
            self.clip_g = None

        if t5:
            logger.info("Using T5XXL text encoder from hidream TEModel")
            dtype_t5 = comfy.model_management.pick_weight_dtype(dtype_t5, dtype, device)
            # peek shared.weight layer to determine if we need unchained model
            self.t5xxl = T5XXLModel(device=device, dtype=dtype_t5, model_options=model_options, attention_mask=True)
            self.dtypes.add(dtype_t5)
        else:
            self.t5xxl = None


        if llama:
            logger.debug("Using LLAMA3 text encoder from hidream TEModel")
            dtype_llama = comfy.model_management.pick_weight_dtype(dtype_llama, dtype, device)
            if "vocab_size" not in model_options:
                model_options["vocab_size"] = 128256
            self.llama = hunyuan_video.LLAMAModel(device=device, dtype=dtype_llama, model_options=model_options, layer="all", layer_idx=None, special_tokens={"start": 128000, "pad": 128009})
            self.dtypes.add(dtype_llama)
        else:
            self.llama = None

        logger.info("Created HiDream text encoder with: clip_l {}, clip_g {}, t5xxl {}:{}, llama {}:{}".format(clip_l, clip_g, t5, dtype_t5, llama, dtype_llama))

    def set_clip_options(self, options):
        if self.clip_l is not None:
            self.clip_l.set_clip_options(options)
        if self.clip_g is not None:
            self.clip_g.set_clip_options(options)
        if self.t5xxl is not None:
            self.t5xxl.set_clip_options(options)
        if self.llama is not None:
            self.llama.set_clip_options(options)

    def reset_clip_options(self):
        if self.clip_l is not None:
            self.clip_l.reset_clip_options()
        if self.clip_g is not None:
            self.clip_g.reset_clip_options()
        if self.t5xxl is not None:
            self.t5xxl.reset_clip_options()
        if self.llama is not None:
            self.llama.reset_clip_options()

    def encode_token_weights(self, token_weight_pairs):
        token_weight_pairs_l = token_weight_pairs["l"]
        token_weight_pairs_g = token_weight_pairs["g"]
        token_weight_pairs_t5 = token_weight_pairs["t5xxl"]
        token_weight_pairs_llama = token_weight_pairs["llama"]
        lg_out = None
        pooled = None
        extra = {}

        if len(token_weight_pairs_g) > 0 or len(token_weight_pairs_l) > 0:
            if self.clip_l is not None:
                lg_out, l_pooled = self.clip_l.encode_token_weights(token_weight_pairs_l)
            else:
                l_pooled = torch.zeros((1, 768), device=comfy.model_management.intermediate_device())

            if self.clip_g is not None:
                g_out, g_pooled = self.clip_g.encode_token_weights(token_weight_pairs_g)
            else:
                g_pooled = torch.zeros((1, 1280), device=comfy.model_management.intermediate_device())

            pooled = torch.cat((l_pooled, g_pooled), dim=-1)

        if self.t5xxl is not None:
            t5_output = self.t5xxl.encode_token_weights(token_weight_pairs_t5)
            t5_out, t5_pooled = t5_output[:2]
        else:
            t5_out = None

        if self.llama is not None:
            ll_output = self.llama.encode_token_weights(token_weight_pairs_llama)
            ll_out, ll_pooled = ll_output[:2]
            ll_out = ll_out[:, 1:]
        else:
            ll_out = None

        if t5_out is None:
            t5_out = torch.zeros((1, 128, 4096), device=comfy.model_management.intermediate_device())

        if ll_out is None:
            ll_out = torch.zeros((1, 32, 1, 4096), device=comfy.model_management.intermediate_device())

        if pooled is None:
            pooled = torch.zeros((1, 768 + 1280), device=comfy.model_management.intermediate_device())

        extra["conditioning_llama3"] = ll_out
        return t5_out, pooled, extra

    def load_sd(self, sd):
        if "text_model.encoder.layers.30.mlp.fc1.weight" in sd:
            return self.clip_g.load_sd(sd)
        elif "text_model.encoder.layers.1.mlp.fc1.weight" in sd:
            return self.clip_l.load_sd(sd)
        elif "encoder.block.23.layer.1.DenseReluDense.wi_1.weight" in sd:
            return self.t5xxl.load_sd(sd)
        else:
            return self.llama.load_sd(sd)


def hidream_clip(clip_l=True, clip_g=True, t5=True, llama=True, dtype_t5=None, dtype_llama=None, t5xxl_scaled_fp8=None, llama_scaled_fp8=None, unchained_t5=False, distilled_t5=False):
    class HiDreamTEModel_(HiDreamTEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if t5 and unchained_t5:
                model_options = model_options.copy()
                model_options["unchained_t5"] = True
            if t5 and distilled_t5:
                model_options = model_options.copy()
                model_options["distilled_t5"] = True
            if t5xxl_scaled_fp8 is not None and "t5xxl_scaled_fp8" not in model_options:
                model_options = model_options.copy()
                model_options["t5xxl_scaled_fp8"] = t5xxl_scaled_fp8
            if llama_scaled_fp8 is not None and "llama_scaled_fp8" not in model_options:
                model_options = model_options.copy()
                model_options["llama_scaled_fp8"] = llama_scaled_fp8
            super().__init__(clip_l=clip_l, clip_g=clip_g, t5=t5, llama=llama, dtype_t5=dtype_t5, dtype_llama=dtype_llama, device=device, dtype=dtype, model_options=model_options)
    return HiDreamTEModel_
