import torch
import logging

logger = logging.getLogger(__name__)

MODEL_EXPECTATIONS = {
    "sdxl": {
        "slices": {
            "clip_l": ("full", 0, 768),
            "clip_g": ("full", 768, 2048),
        },
        "pooled": [
            { "clip_g": "no_token" }
        ],
        "pooled_strategy": "last"
    },
    "sd1": {
        "slices": {
            "clip_l": ("full", 0, 768),
        },
        "pooled": [
            { "clip_l": "no_token" }
        ],
        "pooled_strategy": "last"
    },
    "flux": {
        "slices": {
            "t5": ("full", None, None),
            "clip_l": ("full", 0, 768),
        },
        "pooled": [
            { "clip_l": "full" }
        ],
        "pooled_strategy": "last"
    },
    "hidream": {
        "slices": {
            "clip_l": ("no_token", 0, 768),
            "clip_g": ("no_token", 768, 2048),
            "t5xxl":  ("full", None, None),
            "llama":  ("extended", None, None),
        },
        "pooled": [
            { "clip_l": "no_token" },
            { "clip_g": "no_token" }
        ],
        "pooled_strategy": "concat",
        "special_processing": [
            "llama → meta['conditioning_llama3']"
        ]
    },
    "sdxl-clip-only": {
        "slices": {
            "clip": ("full", 0, 2048),
        },
        "pooled": [
            { "clip": "no_token" }
        ],
        "pooled_strategy": "last"
    }
}


import torch
from typing import List, Dict, Union
import logging



import torch
from typing import List, Optional, Union, Iterator
from collections.abc import MutableSequence

class UsefulConditioning(MutableSequence):
    """
    Canonical, list-compatible wrapper for conditioning bundles:
      - [cond_tensor, metadata]
      - [cond_tensor, pooled_tensor, metadata]
    Always normalizes to: [cond_tensor, {pooled_output, ...}]
    """

    def __init__(self, conds: List[list]):
        self.conds: List[List[Union[torch.Tensor, dict]]] = []

        for entry in conds:
            if len(entry) == 2:
                cond, meta = entry
            elif len(entry) == 3:
                cond, pooled, meta = entry
                if isinstance(meta, dict):
                    meta = dict(meta)
                    meta.setdefault("pooled_output", pooled)
                else:
                    raise TypeError("Expected dict as third element for 3-part conditioning")
            else:
                raise ValueError(f"Unsupported conditioning entry: {entry}")

            assert isinstance(cond, torch.Tensor), "First element must be tensor"
            assert isinstance(meta, dict), "Last element must be dictionary"
            self.conds.append([cond, meta])

    # --- List Interface ---
    def __len__(self): return len(self.conds)
    def __getitem__(self, idx): return self.conds[idx]
    def __setitem__(self, idx, value):
        assert isinstance(value, list) and len(value) in (2, 3)
        self.conds[idx] = UsefulConditioning([value])[0]
    def __delitem__(self, idx): del self.conds[idx]
    def __iter__(self) -> Iterator: return iter(self.conds)
    def __contains__(self, item): return item in self.conds
    def insert(self, index, item):
        self.conds.insert(index, UsefulConditioning(item)[0])
    def append(self, item): self.insert(len(self.conds), item)
    def extend(self, items): [self.append(i) for i in items]
    def pop(self, index=-1): return self.conds.pop(index)
    def index(self, value): return self.conds.index(value)
    def count(self, value): return self.conds.count(value)

    def to_list(self) -> List[List[Union[torch.Tensor, dict]]]:
        return self.conds

    # --- Semantic Access ---
    def get_tensor(self, index: int) -> torch.Tensor:
        return self.conds[index][0]

    def set_tensor(self, index: int, tensor: torch.Tensor):
        self.conds[index][0] = tensor

    def get_pooled(self, index: int) -> Optional[torch.Tensor]:
        return self.conds[index][1].get("pooled_output", None)

    def set_pooled(self, index: int, pooled: torch.Tensor):
        self.conds[index][1]["pooled_output"] = pooled

    def get_field(self, index: int, key: str) -> Optional[torch.Tensor]:
        return self.conds[index][1].get(key, None)

    def set_field(self, index: int, key: str, value: torch.Tensor):
        self.conds[index][1][key] = value

    def get_all_tensors(self) -> List[torch.Tensor]:
        return [entry[0] for entry in self.conds]

    def get_all_pooled(self) -> List[Optional[torch.Tensor]]:
        return [entry[1].get("pooled_output", None) for entry in self.conds]

    def get_all_metadata(self) -> List[dict]:
        return [entry[1] for entry in self.conds]

    def clone(self, device="cpu") -> "UsefulConditioning":
        cloned = []
        for tensor, meta in self.conds:
            cloned_tensor = tensor.clone().to(device).detach().contiguous()
            cloned_meta = {k: (v.clone().to(device).detach().contiguous() if torch.is_tensor(v) else v) for k, v in meta.items()}
            cloned.append([cloned_tensor, cloned_meta])
        return UsefulConditioning(cloned)

    def slice(self, start: int, end: Optional[int] = None) -> "UsefulConditioning":
        return UsefulConditioning(self.conds[start:end])



class ConditioningHelper:
    """
    This is a helper util meant to provide common functionality for conditioning nodes.
    """

    @staticmethod
    def convert_conditioning(conditioning: list):
        """
        Convert a list of conditioning tensors to a UsefulConditioning object.
        """
        if not ConditioningHelper.verify(conditioning, silent=True):
            raise ValueError("Invalid conditioning format.")

        # Convert the list of lists to UsefulConditioning
        return UsefulConditioning(conditioning)

    @staticmethod
    def verify(conditioning: list, silent: bool = False) -> bool:
        """
        Verify that the conditioning is a list of tensors.
        """
        if not isinstance(conditioning, list):
            if not silent:
                logger.error("Conditioning must be a list of tensors.")
            return False
        # okay it's a list, so lets figure out if the first item is also a list
        if not conditioning:
            if not silent:
                logger.error("Conditioning list is empty.")
            return False # no list, no tensors
        for item in conditioning:
            if not isinstance(item, list):
                if not silent:
                    logger.error("Conditioning item is not a list.")
                return False # no list inside the list, so clearly not a proper conditioning
            else: # okay we have a list within a list, lets see if the first item is a tensor
                if not isinstance(item[0], torch.Tensor):
                    if not silent:
                        logger.error("Conditioning item 0 is not a tensor.")
                    return False # nope 0th item is not a tensor, so not a proper conditioning
                # Okay we have a list with a tensor at the 0th position, lets check the pooled_output to see if it exists and is a tensor
                if len(item) > 1 and not isinstance(item[1], dict):
                    if not silent:
                        logger.error("Conditioning item 1 is not a dict.")
                    return False # 1st item is not a dict, so not a proper conditioning
                if len(item) > 1 and "pooled_output" in item[1] and not isinstance(item[1]["pooled_output"], torch.Tensor):
                    if not silent:
                        logger.error("Conditioning item 1 pooled_output is not a tensor.")
                    return False # pooled_output is not a tensor, so not a proper conditioning
        # if we made it this far, then we have a proper conditioning
        return True

import torch
from typing import List, Dict, Union
import logging


class ModelSlicer:
    @staticmethod
    def slice(
        conds: Union[UsefulConditioning, List[List[Union[torch.Tensor, dict]]]],
        model_type: str,
        device: str = "cpu"
    ) -> UsefulConditioning:
        """
        Slices and tags a full conditioning input into its component parts based on MODEL_EXPECTATIONS.
        Preserves all original metadata and injects slicing tags into `meta['slicer_info']`.
        """
        uc = conds if isinstance(conds, UsefulConditioning) else UsefulConditioning(conds)
        expectation = MODEL_EXPECTATIONS.get(model_type)

        if not expectation:
            raise ValueError(f"[ModelSlicer] No expectations found for model type: {model_type}")

        slices_def = expectation.get("slices", {})
        if not slices_def:
            raise ValueError(f"[ModelSlicer] No slices defined for model type: {model_type}")

        output = []

        for entry_index in range(len(uc)):
            tensor = uc.get_tensor(entry_index)
            pooled = uc.get_pooled(entry_index)
            meta = uc.get_all_metadata()[entry_index]

            for key, (scope, start, end) in slices_def.items():
                if scope == "full":
                    sliced = tensor[:, :, start:end] if start is not None else tensor
                elif scope == "no_token":
                    sliced = tensor[:, -1:, start:end] if start is not None else tensor[:, -1:]
                elif scope == "extended":
                    sliced = tensor
                else:
                    logger.warning(f"[ModelSlicer] Unknown scope: {scope}, skipping key: {key}")
                    continue

                tagged_meta = dict(meta)  # full clone
                tagged_meta.setdefault("slicer_info", {})
                tagged_meta["slicer_info"].update({
                    "key": key,
                    "origin": model_type,
                    "scope": scope,
                })

                if pooled is not None and "pooled_output" not in tagged_meta:
                    try:
                        pooled_slice = pooled[:, start:end] if start is not None else pooled
                        tagged_meta["pooled_output"] = pooled_slice
                    except Exception:
                        tagged_meta["pooled_output"] = pooled

                output.append([sliced.to(device), tagged_meta])

        # --- Special Processing ---
        specials = expectation.get("special_processing", [])
        for rule in specials:
            try:
                src_key, target_expr = [x.strip() for x in rule.split("→")]
                for _, meta in output:
                    if meta.get("slicer_info", {}).get("key") == src_key:
                        exec(target_expr, {}, {"meta": meta})
            except Exception as e:
                logger.warning(f"[ModelSlicer] Failed special processing: {rule} → {e}")

        return UsefulConditioning(output)
