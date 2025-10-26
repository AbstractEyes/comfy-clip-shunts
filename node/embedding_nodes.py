# abs_nodes/embedding_nodes.py
# ============================================================
import json
import logging
import uuid
from pathlib import Path
from typing import List

import torch
from folder_paths import get_folder_paths
from server import PromptServer

from ..embedding.embedding_manager import get_bank, EmbeddingManager
from ..utils.conditioning_helper import ConditioningHelper, UsefulConditioning

from comfy.utils import ProgressBar
from ..text_encoders.symbolic_logic_manager import SymbolicLogicManager
from ..text_encoders.symbolic_caption_data import SymbolicCaptionGenerator
import random

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Constants / categories
CATEGORY_EMBED = "utils/embedding"
CATEGORY_SYM   = "symbolic/logic"
EVENT_REFRESH  = "abs.bundle.refresh"


# -------------------------------------------------------------------------
# helpers – path & bank
def _default_embed_dir() -> str:
    """Absolute path to the bank subdir under ComfyUI embeddings."""
    return str(Path(get_folder_paths("embeddings")[0]) / "cached_embeddings")


def _bank(*, path: str = "", force: bool = False) -> EmbeddingManager:
    """
    Retrieve the singleton manager. We do not downcast dtypes here — use whatever
    the manager was constructed with (you can change via get_bank(...) elsewhere).
    """
    if path:
        # switch subdir if caller asks; manager persists this until changed again
        subdir = Path(path).name
        b = get_bank(force_reload=force, subdir=subdir)
        b.load_dir(path)  # ensure absolute path (Windows-safe)
        return b
    # default bank
    return get_bank(force_reload=force)


def _bundle_labels() -> List[str]:
    """
    Display labels for dropdowns. We show triggers if present; else 8-char id.
    Resolution during load uses manager._resolve_ident so labels can be either.
    """
    b = _bank()
    ids = b.list_bundles()
    if not ids:
        return ["<no bundles>"]
    labels = []
    for bid in ids:
        info = b.info(bid)
        label = info.get("prompt_trigger") or bid[:8]
        labels.append(label)
    return labels


# -------------------------------------------------------------------------
# 1 ▸ SAVE NODE
class ABS_SaveEmbedding:
    """
    Save a conditioning bundle into the ABS bank.
    Input accepts either a plain list[[tensor, meta], ...] or UsefulConditioning.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "trigger": ("STRING", {"default": "BEATRIX"}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "meta_json": ("STRING", {"default": "{}", "multiline": True}),
                "force": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "node_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    CATEGORY = CATEGORY_EMBED
    FUNCTION = "save"
    OUTPUT_NODE = True

    def save(self, conditioning, trigger, prompt, meta_json, force, node_id=None):
        # Normalize to the format the manager expects (your helper enforces shape)
        try:
            uc = conditioning if isinstance(conditioning, UsefulConditioning) \
                 else ConditioningHelper.convert_conditioning(conditioning)
        except Exception as e:
            raise ValueError(f"[ABS_SaveEmbedding] invalid conditioning: {e}")

        if len(uc) == 0:
            raise ValueError("conditioning is empty")

        bank = _bank(force=force)

        try:
            extra_meta = json.loads(meta_json or "{}")
        except json.JSONDecodeError:
            logger.warning("[ABS_SaveEmbedding] invalid meta_json; using {}")
            extra_meta = {}

        bundle_id = bank.save_bundle(
            trigger=trigger or f"auto_{uuid.uuid4().hex[:8]}",
            conditioning=uc.to_list(),
            prompt_text=prompt or "",
            folder=bank.paths.dir,
            meta_extra=extra_meta,
        )

        # Nudge the UI to refresh dropdown options immediately
        try:
            PromptServer.instance.send_sync(EVENT_REFRESH, {
                "bundle_id": bundle_id,
                "trigger": trigger,
                "node": node_id or "",
            })
        except Exception:
            pass

        # Pass-through (as conditioning). Keep the same object the graph already carries.
        return (conditioning,)


# -------------------------------------------------------------------------
# 2 ▸ LOAD NODE
class ABS_LoadEmbedding:
    """Load a saved bundle into a UsefulConditioning-compatible list."""

    @classmethod
    def INPUT_TYPES(cls):
        labels = _bundle_labels()
        return {
            "required": {
                "bundle_id": (labels, {"default": labels[0]}),
                "device": (["auto", "cpu", "cuda"], {"default": "auto"}),
                "dtype": (["keep", "float16", "float32"], {"default": "keep"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    CATEGORY = CATEGORY_EMBED
    FUNCTION = "load"

    def _resolve_dtype(self, pref: str):
        if pref == "float16":
            return torch.float16
        if pref == "float32":
            return torch.float32
        return None  # keep saved dtype

    def load(self, bundle_id, device, dtype):
        if bundle_id == "<no bundles>":
            raise ValueError("No bundles available to load")

        bank = _bank()

        # Resolve device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        to_dtype = self._resolve_dtype(dtype)

        # Manager returns list[[tensor, meta], ...] already in correct shape
        cond_list = bank.load_by_id(bundle_id, to_device=device, to_dtype=to_dtype)
        return (cond_list,)


# -------------------------------------------------------------------------
# 3 ▸ SHAPER NODE
class ABS_ShaperEmbedding:
    """
    learn=True  → save incoming conditioning under 'bundle_id' (used as trigger)
    learn=False → load and merge saved conditioning with incoming one
    """

    @classmethod
    def INPUT_TYPES(cls):
        labels = _bundle_labels()
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "bundle_id": (labels, {"default": labels[0]}),
                "learn": ("BOOLEAN", {"default": False}),
                "force": ("BOOLEAN", {"default": False}),
                "mode": (["prepend", "append", "replace"], {"default": "prepend"}),
            },
            "hidden": {
                "node_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    CATEGORY = CATEGORY_EMBED
    FUNCTION = "apply"
    OUTPUT_NODE = True

    def apply(self, conditioning, bundle_id, learn, force, mode, node_id=None):
        # Normalize incoming
        try:
            uc = conditioning if isinstance(conditioning, UsefulConditioning) \
                 else ConditioningHelper.convert_conditioning(conditioning)
        except Exception as e:
            raise ValueError(f"[ABS_ShaperEmbedding] invalid conditioning: {e}")

        bank = _bank(force=force if learn else False)

        if learn:
            trigger = bundle_id if bundle_id and bundle_id != "<no bundles>" else f"auto_{uuid.uuid4().hex[:8]}"
            bundle_id = bank.save_bundle(
                trigger=trigger,
                conditioning=uc.to_list(),
                prompt_text="",
                folder=bank.paths.dir,
                meta_extra={},
            )
            try:
                PromptServer.instance.send_sync(EVENT_REFRESH, {
                    "bundle_id": bundle_id,
                    "trigger": trigger,
                    "node": node_id or "",
                })
            except Exception:
                pass
            return (conditioning,)

        # load & merge
        if bundle_id == "<no bundles>":
            raise ValueError("No bundle selected for loading")

        loaded = bank.load_by_id(bundle_id, to_device="cuda" if torch.cuda.is_available() else "cpu")

        if mode == "replace":
            return (loaded,)
        elif mode == "append":
            merged = UsefulConditioning(uc.to_list() + loaded)
            return (merged.to_list(),)
        else:  # prepend
            merged = UsefulConditioning(loaded + uc.to_list())
            return (merged.to_list(),)


# -------------------------------------------------------------------------
# 4 ▸ EMBEDDING INSPECTOR NODE
class ABS_InspectEmbedding:
    """Summarize a saved bundle (shapes, pooled presence, and sidecar fields)."""

    @classmethod
    def INPUT_TYPES(cls):
        labels = _bundle_labels()
        return {
            "required": {
                "bundle_id": (labels, {"default": labels[0]}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("info", "prompt_text", "metadata")
    CATEGORY = CATEGORY_EMBED
    FUNCTION = "inspect"
    OUTPUT_NODE = True

    def inspect(self, bundle_id):
        if bundle_id == "<no bundles>":
            return ("No bundles available", "", "{}")

        bank = _bank()
        side = bank.info(bundle_id)  # copy of sidecar dict
        cond = bank.load_by_id(bundle_id, to_device="cpu")  # shapes are readable on CPU

        lines = [
            f"Bundle ID: {side.get('bundle_id','')[:16]}...",
            f"Trigger: {side.get('prompt_trigger','')}",
            f"Created: {side.get('created_at','unknown')}",
            f"Entries: {len(cond)}",
        ]
        # Tensor shapes / pooled presence
        for i, entry in enumerate(cond):
            t = entry[0]
            meta = entry[1] if len(entry) > 1 and isinstance(entry[1], dict) else {}
            pooled = meta.get("pooled_output", None)
            lines.append(f"  Entry {i}: tensor={tuple(t.shape)}, pooled={'yes' if isinstance(pooled, torch.Tensor) else 'no'}")

        # Slim metadata for display (hide raw tensor key list if too long)
        disp = {k: v for k, v in side.items() if k not in ["tensor_keys"]}
        return ("\n".join(lines), side.get("prompt_text", ""), json.dumps(disp, indent=2))


# -------------------------------------------------------------------------
# 5 ▸ SYMBOLIC NODES (kept minimal, behavior unchanged)
BEATRIX_CATEGORIES = [
    "<subject>", "<subject1>", "<subject2>",
    "<pose>", "<emotion>", "<surface>",
    "<lighting>", "<material>", "<accessory>",
    "<footwear>", "<upper_body_clothing>", "<hair_style>",
    "<hair_length>", "<headwear>", "<texture>",
    "<pattern>", "<grid>", "<zone>",
    "<offset>", "<object_left>", "<object_right>",
    "<relation>", "<intent>", "<style>",
    "<fabric>", "<jewelry>",
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
    CATEGORY = CATEGORY_SYM
    OUTPUT_NODE = True

    def run(self, prompt, encoder_pipe, pad_first, slice_length, max_length, each_mode, top_k):
        encoder = encoder_pipe[0]
        if encoder is None:
            raise ValueError("No encoder pipeline provided")

        if "bert" not in encoder.get("config", {}).get("model_type", "").lower():
            logger.warning(f"[SymbolicPromptRouter] pipeline not 'bert', got: {encoder.get('config', {}).get('model_type','')}")
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
        for m in matches:
            logger.info(f"[SymbolicPromptRouter] match: {m.get('trigger','<unk>')} score={m.get('score',0.0)}")
        return (matches,)


class BertPromptSimilarityFlood:
    """
    Dynamically flood a prompt with symbolic captions generated from SymbolicCaptionGenerator.
    """

    @classmethod
    def INPUT_TYPES(cls):
        categories: List[str] = ["all"] + BEATRIX_CATEGORIES
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
    CATEGORY = CATEGORY_SYM

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
