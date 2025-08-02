# abs_nodes/embedding_nodes.py
# ============================================================
import json, logging, uuid, re, torch
from pathlib import Path
from typing import List, Tuple, Optional

from folder_paths import get_folder_paths
from ..embedding.embedding_manager import get_bank, EmbeddingManager

logger = logging.getLogger(__name__)

from ..utils.conditioning_helper import ConditioningHelper, UsefulConditioning

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
    _, obj = cond  # Fixed: was unpacking incorrectly
    if isinstance(obj, dict) and "token_ids" in obj:
        return list(map(int, obj["token_ids"]))
    return []


# -------------------------------------------------------------------------
# 1 ▸ SAVE NODE
class ABS_SaveEmbedding:
    """
    Save CONDITIONING bundle into the ABS bank.
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
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    CATEGORY = "utils/embedding"
    FUNCTION = "save"
    OUTPUT_NODE = True

    def save(self, conditioning, trigger, prompt, meta_json, force):
        conditioning = UsefulConditioning(conditioning) if isinstance(conditioning, list) else conditioning
        if not conditioning:
            raise ValueError("conditioning list is empty")

        bank = _bank(force=force)

        if not trigger:
            trigger = f"auto_{uuid.uuid4().hex[:8]}"

        # Check if trigger already exists
        if not force:
            for bundle_id, meta in bank.meta.items():
                if meta.get("prompt_trigger") == trigger:
                    logger.info(f"[ABS] trigger '{trigger}' exists – skip")
                    return (conditioning,)

        try:
            extra_meta = json.loads(meta_json or "{}")
        except json.JSONDecodeError:
            logger.warning(f"[ABS] Invalid meta_json, using empty dict")
            extra_meta = {}

        bundle_id = bank.save_bundle(
            trigger=trigger,
            conditioning=conditioning,
            prompt_text=prompt,
            folder=bank.path,
            meta_extra=extra_meta
        )

        logger.info(f"[ABS] saved bundle {bundle_id[:8]} as '{trigger}' | prompt='{prompt[:64]}…'")
        return (conditioning,)


# -------------------------------------------------------------------------
# dropdown helper
def _bundle_lists():
    """Get lists of bundle IDs for dropdown display."""
    b = _bank()
    if not b.meta:
        return (["<no bundles>"], [""])

    vis, true = [], []
    for bundle_id, meta in b.meta.items():
        display_name = meta.get("prompt_trigger", bundle_id[:8])
        vis.append(display_name)
        true.append(bundle_id)
    return vis, true


# -------------------------------------------------------------------------
# 2 ▸ LOAD NODE
class ABS_LoadEmbedding:
    """Load a stored bundle as CONDITIONING using UsefulConditioning."""

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
                "device": (["auto", "cpu", "cuda"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    CATEGORY = "utils/embedding"
    FUNCTION = "load"

    def _resolve_bundle(self, ident: str) -> str:
        """Resolve bundle ID from trigger name or partial ID."""
        bank = _bank()

        # Direct ID match
        if ident in bank.meta:
            return ident

        # Search by trigger or partial ID
        for bundle_id, meta in bank.meta.items():
            if meta.get("prompt_trigger") == ident or bundle_id.startswith(ident):
                return bundle_id

        raise KeyError(f"bundle '{ident}' not found")

    def load(self, bundle_id, device):
        if bundle_id == "<no bundles>":
            raise ValueError("No bundles available to load")

        bank = _bank()

        # Resolve the actual bundle ID
        resolved_id = self._resolve_bundle(bundle_id)

        # Load the bundle as UsefulConditioning
        useful_cond = bank.load_bundle(resolved_id)

        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Convert to standard CONDITIONING format with proper device placement
        conditioning_list = []
        for i in range(len(useful_cond)):
            tensor = useful_cond.get_tensor(i).to(device)
            meta = useful_cond.get_all_metadata()[i].copy()

            # Ensure pooled_output is on correct device if present
            if "pooled_output" in meta and isinstance(meta["pooled_output"], torch.Tensor):
                meta["pooled_output"] = meta["pooled_output"].to(device)

            conditioning_list.append([tensor, meta])

        logger.info(
            f"[ABS] loaded bundle {resolved_id[:8]} (trigger: {bank.meta[resolved_id].get('prompt_trigger', 'none')})")

        return (conditioning_list,)


# -------------------------------------------------------------------------
# 3 ▸ SHAPER NODE (Fixed)
class ABS_ShaperEmbedding:
    """
    learn=True  → save incoming conditioning under trigger
    learn=False → load and merge saved embedding with incoming conditioning
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
                "learn": ("BOOLEAN", {"default": False}),
                "force": ("BOOLEAN", {"default": False}),
                "mode": (["prepend", "append", "replace"], {"default": "prepend"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    CATEGORY = "utils/embedding"
    FUNCTION = "apply"
    OUTPUT_NODE = True

    def apply(self, conditioning, bundle_id, learn, force, mode):
        if not conditioning:
            raise ValueError("conditioning list empty")

        if learn:
            # Save mode - use bundle_id as trigger
            if bundle_id == "<no bundles>":
                bundle_id = f"auto_{uuid.uuid4().hex[:8]}"

            saver = ABS_SaveEmbedding()
            return saver.save(
                conditioning=conditioning,
                trigger=bundle_id,
                prompt="",
                meta_json="{}",
                force=force
            )
        else:
            # Load and merge mode
            if bundle_id == "<no bundles>":
                raise ValueError("No bundle selected for loading")

            # Load the saved embedding
            loader = ABS_LoadEmbedding()
            loaded_cond = loader.load(bundle_id, "auto")[0]

            # Merge based on mode
            if mode == "replace":
                return (loaded_cond,)
            elif mode == "append":
                return (conditioning + loaded_cond,)
            else:  # prepend
                return (loaded_cond + conditioning,)


# -------------------------------------------------------------------------
# 4 ▸ EMBEDDING INSPECTOR NODE (New)
class ABS_InspectEmbedding:
    """Inspect the contents of a saved embedding bundle."""

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
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("info", "prompt_text", "metadata")
    CATEGORY = "utils/embedding"
    FUNCTION = "inspect"
    OUTPUT_NODE = True

    def inspect(self, bundle_id):
        if bundle_id == "<no bundles>":
            return ("No bundles available", "", "{}")

        bank = _bank()

        # Resolve bundle
        resolved_id = None
        for bid, meta in bank.meta.items():
            if meta.get("prompt_trigger") == bundle_id or bid.startswith(bundle_id):
                resolved_id = bid
                break

        if not resolved_id:
            return (f"Bundle '{bundle_id}' not found", "", "{}")

        # Get metadata
        meta = bank.meta[resolved_id]
        useful_cond = bank.load_bundle(resolved_id)

        # Build info string
        info_parts = [
            f"Bundle ID: {resolved_id[:16]}...",
            f"Trigger: {meta.get('prompt_trigger', 'none')}",
            f"Created: {meta.get('created_at', 'unknown')}",
            f"Entries: {len(useful_cond)}",
            f"Tensor Keys: {', '.join(meta.get('tensor_keys', []))}",
        ]

        # Add tensor shapes
        for i in range(len(useful_cond)):
            tensor = useful_cond.get_tensor(i)
            pooled = useful_cond.get_pooled(i)
            info_parts.append(
                f"  Entry {i}: tensor={tuple(tensor.shape)}, pooled={'yes' if pooled is not None else 'no'}")

        info = "\n".join(info_parts)
        prompt_text = meta.get("prompt_text", "")

        # Clean metadata for display
        display_meta = {
            k: v for k, v in meta.items()
            if k not in ["conditioning_extras", "tensor_keys", "dims"]
        }
        metadata = json.dumps(display_meta, indent=2)

        return (info, prompt_text, metadata)


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
