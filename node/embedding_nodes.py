# abs_nodes/embedding_nodes.py
# ============================================================
import json, logging, uuid, re, torch
from pathlib import Path
from typing import List, Tuple, Optional

from folder_paths import get_folder_paths            # Comfy helper

from ..embedding.embedding_manager import get_bank, EmbeddingManager

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# helpers – path & bank
def _default_embed_dir() -> str:
    return str(Path(get_folder_paths("embeddings")[0]) / "cached_embeddings")

def _bank(path: str = "", force=False) -> EmbeddingManager:
    return get_bank(path or _default_embed_dir(), force_reload=force)

# token parsing helpers
def _parse_token_field(raw: str) -> List[int]:
    """'101, 202 303' → [101,202,303]."""
    return [int(v) for v in re.split(r"[,\s]+", raw.strip()) if v]

def _extract_tokens_from_cond(cond: list) -> List[int]:
    """Look for token-IDs embedded in conditioning extras."""
    if len(cond) < 2:
        return []
    key, obj = cond[1]
    if isinstance(obj, torch.Tensor) and key.lower().startswith("token"):
        return obj.flatten().int().cpu().tolist()
    if isinstance(obj, dict) and "token_ids" in obj:
        return list(map(int, obj["token_ids"]))
    return []

# -------------------------------------------------------------------------
# 1 ▸ SAVE NODE
class ABS_SaveEmbedding:
    """
    Save first tensor in CONDITIONING into the ABS bank.
    Token-IDs are auto-extracted or can be supplied manually.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "trigger": (["BEATRIX", "ZANA", "WILDCARD"], {"default": "BEATRIX"} ),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "meta_json": ("STRING", {"default": "{}", "multiline": True}),
                "force": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    CATEGORY     = "utils/embedding"
    FUNCTION     = "save"
    OUTPUT_NODE = True

    # ------------------------------------------------------------------
    def save(self, conditioning, trigger, prompt, meta_json, force):
        if not conditioning:
            raise ValueError("conditioning list is empty")

        bank = _bank()
        if not trigger:
            trigger = f"auto_{uuid.uuid4().hex[:8]}"
        if not force:
            for md5, meta in bank.meta.items():
                if meta.get("prompt_trigger") == trigger:
                    logger.info(f"[ABS] trigger '{trigger}' exists – skip")
                    return (conditioning,)

        md5 = bank.save_conditioning_pack(
            conditioning=conditioning,
            trigger=trigger,
            prompt_text=prompt,
            folder=bank.path,
            meta_extra=json.loads(meta_json or "{}")
        )
        bank.build_prompt_matrix()
        logger.info(f"[ABS] saved bundle {md5[:8]} as '{trigger}' | prompt='{prompt[:64]}…'")
        return (conditioning,)


# -------------------------------------------------------------------------
# dropdown helper
def _bundle_lists():
    b = _bank()
    if not b.meta:
        return (["<no bundles>"], [""])
    vis, true = [], []
    for md5, meta in b.meta.items():
        vis.append(meta.get("prompt_trigger") or md5[:8])
        true.append(md5)
    return vis, true

# -------------------------------------------------------------------------
# 2 ▸ LOAD NODE
class ABS_LoadEmbedding:
    """Load a stored bundle tensor as CONDITIONING."""

    @classmethod
    def INPUT_TYPES(cls):
        vis, _ = _bundle_lists()
        if not vis:
            vis = ["<no bundles>"]
            default = "<no bundles>"
        else:
            default = vis[0]
        return {
            "required": {
                "bundle_id": (vis, {"default": default}),
                "tensor_key": ("STRING", {"default": "conditioning"}),
                "device": ("STRING", {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    CATEGORY     = "utils/embedding"
    FUNCTION     = "load"

    # ------------------------------------------------------------------
    def _md5(self, ident: str) -> str:
        for md5, meta in _bank().meta.items():
            if md5.startswith(ident) or meta.get("prompt_trigger") == ident:
                return md5
        raise KeyError(f"bundle '{ident}' not found")

    def load(self, bundle_id, tensor_key, device):
        bank = _bank()
        md5  = self._md5(bundle_id)
        tensor = bank.get_tensor(md5, tensor_key)

        device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        return ([(tensor_key, tensor.to(device))],)


# -------------------------------------------------------------------------
# 3 ▸ SHAPER NODE
class ABS_ShaperEmbedding:
    """
    learn=True  → save incoming tensor (+ tokens) under trigger
    learn=False → prepend saved embedding (trigger / md5) to pipe
    """

    @classmethod
    def INPUT_TYPES(cls):
        vis, _ = _bundle_lists()
        if not vis:
            vis = ["<no bundles>"]
            default = "<no bundles>"
        else:
            default = vis[0]
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "bundle_id": (vis, {"default": default}),
                "learn":   ("BOOLEAN", {"default": False}),
                "force":   ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    CATEGORY     = "utils/embedding"
    FUNCTION     = "apply"
    OUTPUT_NODE = True


    # ------------------------------------------------------------------
    def apply(self, conditioning, trigger, learn, force):
        if not conditioning:
            raise ValueError("conditioning list empty")

        if learn:
            ABS_SaveEmbedding().save(conditioning, trigger, "", "{}", force)
            return (conditioning,)

        if not trigger:
            raise ValueError("trigger required when learn=False")

        #todo rewrite it's broken
        return #todo broken


from comfy.utils import ProgressBar
from ..embedding.embedding_manager import get_bank
from ..text_encoders.symbolic_logic_manager import SymbolicLogicManager

BEATRIX_CATEGORIES = [
    "<subject>","<subject1>","<subject2>",
    "<pose>","<emotion>","<surface>",
    "<lighting>","<material>","<accessory>",
    "<footwear>", "<upper_body_clothing>","<hair_style>",
    "<hair_length>","<headwear>","<texture>",
    "<pattern>","<grid>","<zone>",
    "<offset>","<object_left>","<object_right>",
    "<relation>","<intent>","<style>",
    "<fabric>","<jewelry>"
]
class SymbolicPromptRouter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "encoder_pipe": ("ENCODER_PIPE", {}),
                "pad_first": ("INT", {"default": 0, "min": 0, "max": 16}),
                "slice_length": ("INT", {"default": 77, "min": 4, "max": 8192}),
                "max_length": ("INT", {"default": 2048, "min": 64, "max": 8192}),
                "each_mode": (["rigid", "push", "end", "inject"], {"default": "rigid"}),
                "top_k": ("INT", {"default": 5, "min": 1, "max": 20}),
            }
        }


    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("symbolic_matches",)
    FUNCTION = "run"
    CATEGORY = "symbolic/logic"

    OUTPUT_NODE = True

    def run(self, prompt, encoder_pipe, pad_first, slice_length, max_length, each_mode, top_k):

        encoder = encoder_pipe[0]
        if encoder is None:
            raise ValueError("No encoder pipeline provided")
        else:
            if "bert" not in encoder.get("config", {}).get("model_type", "").lower():
                logger.warning(f"[SymbolicPromptRouter] Encoder pipeline type '{encoder}' is not 'bert'.")
                logger.warning(f"[SymbolicPromptRouter] Config type: {encoder.get('config', {})} {encoder.get('config', {}).get('model_type', '')}")
                raise ValueError("Encoder pipeline must be of type 'symbolic_logic'")
            model = encoder.get("model", None)
            tokenizer = encoder.get("tokenizer", None)

        pbar = ProgressBar(total=52)

        logic = SymbolicLogicManager(
            base_prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            pad_first=pad_first,
            slice_length=slice_length,
            max_length=max_length,
            each_mode=each_mode,
            pbar=pbar
        )
        logger.info(f"[SymbolicPromptRouter] Running symbolic logic with prompt: {prompt}")

        matches = logic.extract_alpha_similarities(
            embedding_manager=get_bank(),
            top_k=top_k
        )
        """
            candidates.append({
                "md5": f"{special_token}_injection",
                "score": float(score.item()),
                "trigger": special_token,
                "prompt_text": self.tokenizer.decode(modified_ids[0], skip_special_tokens=False),
                "pooled": pooled.squeeze(0).cpu().tolist() if use_pooled else None
            })
        """
        for match in matches:
            logger.info(f"[SymbolicPromptRouter] Found match: {match.get('trigger', '<unknown>')} "
                        f"with score: {match.get('score', 0.0)} ")
                        #f"and prompt: {match.get('prompt_text', '<no prompt>')[:64]}…")

        return (matches,)


import random
from ..text_encoders.symbolic_caption_data import SymbolicCaptionGenerator

class BertPromptSimilarityFlood:
    """
    Dynamically flood a prompt with symbolic captions generated from SymbolicCaptionGenerator.
    """

    @classmethod
    def INPUT_TYPES(cls):
        categories: list[str] = ["all"] + BEATRIX_CATEGORIES
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "count": ("INT", {"default": 50, "min": 1, "max": 5000}),
                "token_limit": ("INT", {"default": 512, "min": 16, "max": 4096}),
                "mode": (["append", "replace", "push"], {"default": "append"}),
                "shuffle": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 42}),
                "join_with": ("STRING", {"default": ".,"}),
                "category": (categories, {"default": "<subject>"}),
            }
        }


    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("flooded_prompt",)
    FUNCTION = "flood"
    CATEGORY = "symbolic/logic"

    def __init__(self):
        self.generator = SymbolicCaptionGenerator()

    def flood(self, prompt, count, token_limit, mode, shuffle, seed, join_with, category=None):
        random.seed(seed)
        prompts = []

        category = category if category and category != "all" else None
        for _ in range(count):
            sample = self.generator.generate_training_sample(num_captions=1, primary_category=category)
            prompts.append(sample.full_caption.strip() + join_with)

        if shuffle:
            random.shuffle(prompts)

        if mode == "replace":
            all_text = prompts
        elif mode == "push":
            all_text = prompts + ([prompt.strip()] if prompt.strip() else [])
        else:  # append
            all_text = ([prompt.strip()] if prompt.strip() else []) + prompts


        final_text = " ".join(all_text)
        tokens = final_text.split()

        if len(tokens) > token_limit:
            tokens = tokens[:token_limit]

        return (" ".join(tokens),)
