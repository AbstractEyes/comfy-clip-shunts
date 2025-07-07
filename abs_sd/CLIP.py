"""
    CLIP.py - A class for handling CLIP text encoder models in ComfyUI
    Originally created by: ComfyUI Team
    Modified by: AbstractPhil

    This is a refactored and fully capable variation of the CLIP pipeline object
    This variation is designed to be used with the ABS Shunt Suite and is backwards compatible
    It houses the same behavioral responses as the original CLIP pipeline object,
    However it is designed to be more flexible and extensible for future use cases.

    This is fully integrated with the HOOK structure defined in the EncoderWrapper using the NeuralIO concept.

    This does not complexify, this simplifies and distributes behavior across multiple multiple capabilities.
    This enables the additional model inclusion of new capabilities without destroying the original functionality,
    cutting the developer time to a fraction of what it would be otherwise.

    This will allow for inline integration of many new models in rapid succession,
    with minimal changes to the existing codebase.
"""
from enum import Enum

import torch

import comfy
import logging

from ..text_encoders.old import sd1_clip
from ..text_encoders.old import hidream
from ..text_encoders.old import sdxl_clip
from ..text_encoders.old import flux
from ..text_encoders.old import sa_t5
from ..text_encoders.old import sd2_clip
from ..text_encoders.old import sd3_clip
from ..text_encoders.old import spiece_tokenizer
from ..text_encoders.old import omnigen2
from ..text_encoders.old import cosmos
from ..text_encoders.old import t5
from ..text_encoders.old import ace
from ..text_encoders.old import ace_text_cleaners
from ..text_encoders.old import aura_t5
from ..text_encoders.old import bert
from ..text_encoders.old import genmo
from ..text_encoders.old import hunyuan_video
from ..text_encoders.old import hydit
from ..text_encoders.old import llama
from ..text_encoders.old import long_clipl
from ..text_encoders.old import lt
from ..text_encoders.old import lumina2
from ..text_encoders.old import pixart_t5
from ..text_encoders.old import sa_t5
from ..text_encoders.old import wan



from comfy import model_management
from comfy.utils import ProgressBar

logger = logging.getLogger(__name__)


class CLIP:
    def __init__(self, target=None, embedding_directory=None, no_init=False, tokenizer_data={}, parameters=0, model_options={}):
        if no_init:
            return
        params = target.params.copy()
        clip = target.clip
        tokenizer = target.tokenizer

        load_device = model_options.get("load_device", model_management.text_encoder_device())
        offload_device = model_options.get("offload_device", model_management.text_encoder_offload_device())
        dtype = model_options.get("dtype", None)
        if dtype is None:
            dtype = model_management.text_encoder_dtype(load_device)

        params['dtype'] = dtype
        params['device'] = model_options.get("initial_device", model_management.text_encoder_initial_device(load_device, offload_device, parameters * model_management.dtype_size(dtype)))
        params['model_options'] = model_options

        self.cond_stage_model = clip(**(params))

        for dt in self.cond_stage_model.dtypes:
            if not model_management.supports_cast(load_device, dt):
                load_device = offload_device
                if params['device'] != offload_device:
                    self.cond_stage_model.to(offload_device)
                    logging.warning("Had to shift TE back.")

        self.tokenizer = tokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)
        self.patcher = comfy.model_patcher.ModelPatcher(self.cond_stage_model, load_device=load_device, offload_device=offload_device)
        self.patcher.hook_mode = comfy.hooks.EnumHookMode.MinVram
        self.patcher.is_clip = True
        self.apply_hooks_to_conds = None
        if params['device'] == load_device:
            model_management.load_models_gpu([self.patcher], force_full_load=True)
        self.layer_idx = None
        self.use_clip_schedule = False
        logging.info("CLIP/text encoder model load device: {}, offload device: {}, current: {}, dtype: {}".format(load_device, offload_device, params['device'], dtype))
        self.tokenizer_options = {}

    def clone(self):
        n = CLIP(no_init=True)
        n.patcher = self.patcher.clone()
        n.cond_stage_model = self.cond_stage_model
        n.tokenizer = self.tokenizer
        n.layer_idx = self.layer_idx
        n.tokenizer_options = self.tokenizer_options.copy()
        n.use_clip_schedule = self.use_clip_schedule
        n.apply_hooks_to_conds = self.apply_hooks_to_conds
        return n

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        return self.patcher.add_patches(patches, strength_patch, strength_model)

    def set_tokenizer_option(self, option_name, value):
        self.tokenizer_options[option_name] = value

    def clip_layer(self, layer_idx):
        self.layer_idx = layer_idx

    def tokenize(self, text, return_word_ids=False, **kwargs):
        tokenizer_options = kwargs.get("tokenizer_options", {})
        if len(self.tokenizer_options) > 0:
            tokenizer_options = {**self.tokenizer_options, **tokenizer_options}
        if len(tokenizer_options) > 0:
            kwargs["tokenizer_options"] = tokenizer_options
        return self.tokenizer.tokenize_with_weights(text, return_word_ids, **kwargs)

    def add_hooks_to_dict(self, pooled_dict: dict[str]):
        if self.apply_hooks_to_conds:
            pooled_dict["hooks"] = self.apply_hooks_to_conds
        return pooled_dict

    def encode_from_tokens_scheduled(self, tokens, unprojected=False, add_dict: dict[str]={}, show_pbar=True):
        all_cond_pooled: list[tuple[torch.Tensor, dict[str]]] = []
        all_hooks = self.patcher.forced_hooks
        if all_hooks is None or not self.use_clip_schedule:
            # if no hooks or shouldn't use clip schedule, do unscheduled encode_from_tokens and perform add_dict
            return_pooled = "unprojected" if unprojected else True
            pooled_dict = self.encode_from_tokens(tokens, return_pooled=return_pooled, return_dict=True)
            cond = pooled_dict.pop("cond")
            # add/update any keys with the provided add_dict
            pooled_dict.update(add_dict)
            all_cond_pooled.append([cond, pooled_dict])
        else:
            scheduled_keyframes = all_hooks.get_hooks_for_clip_schedule()

            self.cond_stage_model.reset_clip_options()
            if self.layer_idx is not None:
                self.cond_stage_model.set_clip_options({"layer": self.layer_idx})
            if unprojected:
                self.cond_stage_model.set_clip_options({"projected_pooled": False})

            self.load_model()
            all_hooks.reset()
            self.patcher.patch_hooks(None)
            if show_pbar:
                pbar = ProgressBar(len(scheduled_keyframes))

            for scheduled_opts in scheduled_keyframes:
                t_range = scheduled_opts[0]
                # don't bother encoding any conds outside of start_percent and end_percent bounds
                if "start_percent" in add_dict:
                    if t_range[1] < add_dict["start_percent"]:
                        continue
                if "end_percent" in add_dict:
                    if t_range[0] > add_dict["end_percent"]:
                        continue
                hooks_keyframes = scheduled_opts[1]
                for hook, keyframe in hooks_keyframes:
                    hook.hook_keyframe._current_keyframe = keyframe
                # apply appropriate hooks with values that match new hook_keyframe
                self.patcher.patch_hooks(all_hooks)
                # perform encoding as normal
                o = self.cond_stage_model.encode_token_weights(tokens)
                cond, pooled = o[:2]
                pooled_dict = {"pooled_output": pooled}
                # add clip_start_percent and clip_end_percent in pooled
                pooled_dict["clip_start_percent"] = t_range[0]
                pooled_dict["clip_end_percent"] = t_range[1]
                # add/update any keys with the provided add_dict
                pooled_dict.update(add_dict)
                # add hooks stored on clip
                self.add_hooks_to_dict(pooled_dict)
                all_cond_pooled.append([cond, pooled_dict])
                if show_pbar:
                    pbar.update(1)
                model_management.throw_exception_if_processing_interrupted()
            all_hooks.reset()
        return all_cond_pooled

    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        self.cond_stage_model.reset_clip_options()

        if self.layer_idx is not None:
            self.cond_stage_model.set_clip_options({"layer": self.layer_idx})

        if return_pooled == "unprojected":
            self.cond_stage_model.set_clip_options({"projected_pooled": False})

        self.load_model()
        o = self.cond_stage_model.encode_token_weights(tokens)
        cond, pooled = o[:2]
        if return_dict:
            out = {"cond": cond, "pooled_output": pooled}
            if len(o) > 2:
                for k in o[2]:
                    out[k] = o[2][k]
            self.add_hooks_to_dict(out)
            return out

        if return_pooled:
            return cond, pooled
        return cond

    def encode(self, text):
        tokens = self.tokenize(text)
        return self.encode_from_tokens(tokens)

    def load_sd(self, sd, full_model=False):
        if full_model:
            return self.cond_stage_model.load_state_dict(sd, strict=False)
        else:
            return self.cond_stage_model.load_sd(sd)

    def get_sd(self):
        sd_clip = self.cond_stage_model.state_dict()
        sd_tokenizer = self.tokenizer.state_dict()
        for k in sd_tokenizer:
            sd_clip[k] = sd_tokenizer[k]
        return sd_clip

    def load_model(self):
        model_management.load_model_gpu(self.patcher)
        return self.patcher

    def get_key_patches(self):
        return self.patcher.get_key_patches()


class CLIPType(Enum):
    STABLE_DIFFUSION = 1
    STABLE_CASCADE = 2
    SD3 = 3
    STABLE_AUDIO = 4
    HUNYUAN_DIT = 5
    FLUX = 6
    MOCHI = 7
    LTXV = 8
    HUNYUAN_VIDEO = 9
    PIXART = 10
    COSMOS = 11
    LUMINA2 = 12
    WAN = 13
    HIDREAM = 14
    CHROMA = 15
    ACE = 16
    OMNIGEN2 = 17


def load_clip(ckpt_paths, embedding_directory=None, clip_type=CLIPType.STABLE_DIFFUSION, model_options={}):
    clip_data = []
    for p in ckpt_paths:
        clip_data.append(comfy.utils.load_torch_file(p, safe_load=True))
    return load_text_encoder_state_dicts(clip_data, embedding_directory=embedding_directory, clip_type=clip_type, model_options=model_options)


class TEModel(Enum):
    CLIP_L = 1
    CLIP_H = 2
    CLIP_G = 3
    T5_XXL = 4
    T5_XL = 5
    T5_BASE = 6
    LLAMA3_8 = 7
    T5_XXL_OLD = 8
    GEMMA_2_2B = 9
    QWEN25_3B = 10

def detect_te_model(sd):
    if "text_model.encoder.layers.30.mlp.fc1.weight" in sd:
        return TEModel.CLIP_G
    if "text_model.encoder.layers.22.mlp.fc1.weight" in sd:
        return TEModel.CLIP_H
    if "text_model.encoder.layers.0.mlp.fc1.weight" in sd:
        return TEModel.CLIP_L
    if "encoder.block.23.layer.1.DenseReluDense.wi_1.weight" in sd:
        weight = sd["encoder.block.23.layer.1.DenseReluDense.wi_1.weight"]
        if weight.shape[-1] == 4096:
            return TEModel.T5_XXL
        elif weight.shape[-1] == 2048:
            return TEModel.T5_XL
    if 'encoder.block.23.layer.1.DenseReluDense.wi.weight' in sd:
        return TEModel.T5_XXL_OLD
    if "encoder.block.0.layer.0.SelfAttention.k.weight" in sd:
        return TEModel.T5_BASE
    if 'model.layers.0.post_feedforward_layernorm.weight' in sd:
        return TEModel.GEMMA_2_2B
    if 'model.layers.0.self_attn.k_proj.bias' in sd:
        return TEModel.QWEN25_3B
    if "model.layers.0.post_attention_layernorm.weight" in sd:
        return TEModel.LLAMA3_8
    return None

def t5_xxl_detect(state_dict, prefix=""):
    out = {}
    t5_key = "{}encoder.final_layer_norm.weight".format(prefix)
    unchained_key = "shared.weight"
    if t5_key in state_dict:
        out["dtype_t5"] = state_dict[t5_key].dtype
        if unchained_key in state_dict:
            # check shape, if (69328, 4096) it is an unchained model
            if state_dict[unchained_key].shape == (69328, 4096):
                out["unchained_t5"] = True
            else:
                # we are chained
                out["unchained_t5"] = False

    scaled_fp8_key = "{}scaled_fp8".format(prefix)
    if scaled_fp8_key in state_dict:
        out["t5xxl_scaled_fp8"] = state_dict[scaled_fp8_key].dtype

    return out

def t5xxl_detect(clip_data):
    weight_name = "encoder.block.23.layer.1.DenseReluDense.wi_1.weight"
    weight_name_old = "encoder.block.23.layer.1.DenseReluDense.wi.weight"


    for sd in clip_data:
        if weight_name in sd or weight_name_old in sd:
            return t5_xxl_detect(sd)

    return {}


def llama_detect(clip_data):
    weight_name = "model.layers.0.self_attn.k_proj.weight"

    for sd in clip_data:
        if weight_name in sd:
            return hunyuan_video.llama_detect(sd)

    return {}


def load_text_encoder_state_dicts(state_dicts=[], embedding_directory=None, clip_type=CLIPType.STABLE_DIFFUSION, model_options={}):
    clip_data = state_dicts

    class EmptyClass:
        pass

    for i in range(len(clip_data)):
        if "transformer.resblocks.0.ln_1.weight" in clip_data[i]:
            clip_data[i] = comfy.utils.clip_text_transformers_convert(clip_data[i], "", "")
        else:
            if "text_projection" in clip_data[i]:
                clip_data[i]["text_projection.weight"] = clip_data[i]["text_projection"].transpose(0, 1) #old models saved with the CLIPSave node

    tokenizer_data = {}
    clip_target = EmptyClass()
    clip_target.params = {}
    if len(clip_data) == 1:
        te_model = detect_te_model(clip_data[0])
        if te_model == TEModel.CLIP_G:
            if clip_type == CLIPType.STABLE_CASCADE:
                clip_target.clip = sdxl_clip.StableCascadeClipModel
                clip_target.tokenizer = sdxl_clip.StableCascadeTokenizer
            elif clip_type == CLIPType.SD3:
                clip_target.clip = sd3_clip.sd3_clip(clip_l=False, clip_g=True, t5=False)
                clip_target.tokenizer = sd3_clip.SD3Tokenizer
            elif clip_type == CLIPType.HIDREAM:
                logger.info("Loading HiDream CLIP CLIPtype.HIDREAM: model_options={}".format(model_options))
                clip_target.clip = hidream.hidream_clip(clip_l=False, clip_g=True, t5=False, llama=False, dtype_t5=None, dtype_llama=None, t5xxl_scaled_fp8=None, llama_scaled_fp8=None)
                clip_target.tokenizer = hidream.HiDreamTokenizer
            else:
                clip_target.clip = sdxl_clip.SDXLRefinerClipModel
                clip_target.tokenizer = sdxl_clip.SDXLTokenizer
        elif te_model == TEModel.CLIP_H:
            clip_target.clip = sd2_clip.SD2ClipModel
            clip_target.tokenizer = sd2_clip.SD2Tokenizer
        elif te_model == TEModel.T5_XXL:
            if clip_type == CLIPType.SD3:
                clip_target.clip = sd3_clip.sd3_clip(clip_l=False, clip_g=False, t5=True, **t5xxl_detect(clip_data))
                clip_target.tokenizer = sd3_clip.SD3Tokenizer
            elif clip_type == CLIPType.LTXV:
                clip_target.clip = lt.ltxv_te(**t5xxl_detect(clip_data))
                clip_target.tokenizer = lt.LTXVT5Tokenizer
            elif clip_type == CLIPType.PIXART or clip_type == CLIPType.CHROMA:
                clip_target.clip = pixart_t5.pixart_te(**t5xxl_detect(clip_data))
                clip_target.tokenizer = pixart_t5.PixArtTokenizer
            elif clip_type == CLIPType.WAN:
                clip_target.clip = wan.te(**t5xxl_detect(clip_data))
                clip_target.tokenizer = wan.WanT5Tokenizer
                tokenizer_data["spiece_model"] = clip_data[0].get("spiece_model", None)
            elif clip_type == CLIPType.HIDREAM:
                logger.info("Loading HiDream CLIP: model_options={}".format(model_options))
                clip_target.clip = hidream.hidream_clip(**t5xxl_detect(clip_data),
                                                                        clip_l=False, clip_g=False, t5=True, llama=False, dtype_llama=None, llama_scaled_fp8=None)
                clip_target.tokenizer = hidream.HiDreamTokenizer
            else: #CLIPType.MOCHI
                clip_target.clip = genmo.mochi_te(**t5xxl_detect(clip_data))
                clip_target.tokenizer = genmo.MochiT5Tokenizer
        elif te_model == TEModel.T5_XXL_OLD:
            clip_target.clip = cosmos.te(**t5xxl_detect(clip_data))
            clip_target.tokenizer = cosmos.CosmosT5Tokenizer
        elif te_model == TEModel.T5_XL:
            clip_target.clip = aura_t5.AuraT5Model
            clip_target.tokenizer = aura_t5.AuraT5Tokenizer
        elif te_model == TEModel.T5_BASE:
            if clip_type == CLIPType.ACE or "spiece_model" in clip_data[0]:
                clip_target.clip = ace.AceT5Model
                clip_target.tokenizer = ace.AceT5Tokenizer
                tokenizer_data["spiece_model"] = clip_data[0].get("spiece_model", None)
            else:
                clip_target.clip = sa_t5.SAT5Model
                clip_target.tokenizer = sa_t5.SAT5Tokenizer
        elif te_model == TEModel.GEMMA_2_2B:
            clip_target.clip = lumina2.te(**llama_detect(clip_data))
            clip_target.tokenizer = lumina2.LuminaTokenizer
            tokenizer_data["spiece_model"] = clip_data[0].get("spiece_model", None)
        elif te_model == TEModel.LLAMA3_8:
            logger.info("Loading HiDream CLIP - TEModel.LLAMA3_8: model_options={}".format(model_options))
            clip_target.clip = hidream.hidream_clip(**llama_detect(clip_data),
                                                                        clip_l=False, clip_g=False, t5=False, llama=True, dtype_t5=None, t5xxl_scaled_fp8=None)
            clip_target.tokenizer = hidream.HiDreamTokenizer
        elif te_model == TEModel.QWEN25_3B:
            clip_target.clip = omnigen2.te(**llama_detect(clip_data))
            clip_target.tokenizer = omnigen2.Omnigen2Tokenizer
        else:
            # clip_l
            if clip_type == CLIPType.SD3:
                clip_target.clip = sd3_clip.sd3_clip(clip_l=True, clip_g=False, t5=False)
                clip_target.tokenizer = sd3_clip.SD3Tokenizer
            elif clip_type == CLIPType.HIDREAM:
                logger.info("Loading HiDream CLIP CLIPtype.HIDREAM 2: model_options={}".format(model_options))
                clip_target.clip = hidream.hidream_clip(clip_l=True, clip_g=False, t5=False, llama=False, dtype_t5=None, dtype_llama=None, t5xxl_scaled_fp8=None, llama_scaled_fp8=None)
                clip_target.tokenizer = hidream.HiDreamTokenizer
            else:
                clip_target.clip = sd1_clip.SD1ClipModel
                clip_target.tokenizer = sd1_clip.SD1Tokenizer
    elif len(clip_data) == 2:
        if clip_type == CLIPType.SD3:
            te_models = [detect_te_model(clip_data[0]), detect_te_model(clip_data[1])]
            clip_target.clip = sd3_clip.sd3_clip(clip_l=TEModel.CLIP_L in te_models, clip_g=TEModel.CLIP_G in te_models, t5=TEModel.T5_XXL in te_models, **t5xxl_detect(clip_data))
            clip_target.tokenizer = sd3_clip.SD3Tokenizer
        elif clip_type == CLIPType.HUNYUAN_DIT:
            clip_target.clip = hydit.HyditModel
            clip_target.tokenizer = hydit.HyditTokenizer
        elif clip_type == CLIPType.FLUX:
            logger.info("Loading Flux CLIP CLIPtype.FLUX: model_options={}".format(model_options))
            clip_target.clip = flux.flux_clip(**t5xxl_detect(clip_data))
            clip_target.tokenizer = flux.FluxTokenizer
        elif clip_type == CLIPType.HUNYUAN_VIDEO:
            clip_target.clip = hunyuan_video.hunyuan_video_clip(**llama_detect(clip_data))
            clip_target.tokenizer = hunyuan_video.HunyuanVideoTokenizer
        elif clip_type == CLIPType.HIDREAM:
            # Detect
            logger.info("Loading HiDream CLIP CLIPtype.HIDREAM 3: model_options={}".format(model_options))
            hidream_dualclip_classes = []
            for hidream_te in clip_data:
                te_model = detect_te_model(hidream_te)
                hidream_dualclip_classes.append(te_model)

            clip_l = TEModel.CLIP_L in hidream_dualclip_classes
            clip_g = TEModel.CLIP_G in hidream_dualclip_classes
            t5 = TEModel.T5_XXL in hidream_dualclip_classes
            llama = TEModel.LLAMA3_8 in hidream_dualclip_classes

            # Initialize t5xxl_detect and llama_detect kwargs if needed
            t5_kwargs = t5xxl_detect(clip_data) if t5 else {}
            llama_kwargs = llama_detect(clip_data) if llama else {}
            logger.info("Model Options: model_options: {}, t5_kwargs: {}, llama_kwargs: {}".format(model_options, t5_kwargs, llama_kwargs))
            clip_target.clip = hidream.hidream_clip(clip_l=clip_l, clip_g=clip_g, t5=t5, llama=llama, **t5_kwargs, **llama_kwargs)
            clip_target.tokenizer = hidream.HiDreamTokenizer
        else:
            clip_target.clip = sdxl_clip.SDXLClipModel
            clip_target.tokenizer = sdxl_clip.SDXLTokenizer
    elif len(clip_data) == 3:
        clip_target.clip = sd3_clip.sd3_clip(**t5xxl_detect(clip_data))
        clip_target.tokenizer = sd3_clip.SD3Tokenizer
    elif len(clip_data) == 4:
        logger.info("Loading HiDream CLIP CLIPtype.HIDREAM 4, most likely: model_options={}".format(model_options))
        clip_target.clip = hidream.hidream_clip(**t5xxl_detect(clip_data), **llama_detect(clip_data))
        clip_target.tokenizer = hidream.HiDreamTokenizer

    parameters = 0
    for c in clip_data:
        parameters += comfy.utils.calculate_parameters(c)
        tokenizer_data, model_options = long_clipl.model_options_long_clip(c, tokenizer_data, model_options)

    clip = CLIP(clip_target, embedding_directory=embedding_directory, parameters=parameters, tokenizer_data=tokenizer_data, model_options=model_options)
    for c in clip_data:
        m, u = clip.load_sd(c)
        if len(m) > 0:
            logging.warning("clip missing: {}".format(m))

        if len(u) > 0:
            logging.debug("clip unexpected: {}".format(u))
    return clip