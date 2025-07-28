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




class ConditioningHelper:
    """
    This is a helper util meant to provide common functionality for conditioning nodes.
    """

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

    @staticmethod
    def extract_tensors(conditioning: list) -> list:
        return [entry[0] for entry in conditioning if isinstance(entry, list) and isinstance(entry[0], torch.Tensor)]

    @staticmethod
    def get_pooled(conditioning: list, default=None) -> list:
        pooled = []
        for entry in conditioning:
            if isinstance(entry, list) and len(entry) > 1:
                pool = entry[1].get("pooled_output", default)
                pooled.append(pool)
        return pooled


    @staticmethod
    def to(conditioning: list, dtype=None, device=None) -> list:
        """
        Move all tensors in the conditioning structure to the specified dtype and/or device,
        without modifying non-tensor metadata entries. Movement is skipped if not required.
        """
        new_conditioning = []

        for entry in conditioning:
            if not isinstance(entry, list) or not isinstance(entry[0], torch.Tensor):
                continue  # Skip malformed entry

            tensor = entry[0]
            meta = entry[1] if len(entry) > 1 and isinstance(entry[1], dict) else {}
            target_dtype = dtype or tensor.dtype
            target_device = device or tensor.device

            # Move main tensor if needed
            new_tensor = tensor
            if tensor.device != target_device or tensor.dtype != target_dtype:
                new_tensor = tensor.to(dtype=target_dtype, device=target_device)

            # Preserve all metadata, only moving tensors
            new_meta = {}
            for key, val in meta.items():
                if isinstance(val, torch.Tensor):
                    if val.device != target_device or val.dtype != target_dtype:
                        new_meta[key] = val.to(dtype=target_dtype, device=target_device)
                    else:
                        new_meta[key] = val
                else:
                    new_meta[key] = val  # pass through unmodified

            new_conditioning.append([new_tensor, new_meta])

        return new_conditioning

    @staticmethod
    def get_slices(conditioning: list, mode: str, do_clone: bool = False) -> list:
        """
        Slice a conditioning list into per-entry dictionaries using MODEL_EXPECTATIONS[mode].
        Includes all tensor fields from the metadata, including pooled_output.

        Args:
            conditioning: The conditioning list to slice.
            mode: Model key in MODEL_EXPECTATIONS.
            do_clone: If True, deep clone returned tensors.

        Returns:
            List[Dict[str, Tensor]]: One dict per conditioning entry.
        """

        if mode not in MODEL_EXPECTATIONS:
            raise ValueError(f"[ConditioningHelper] Unknown model mode: {mode}")

        expectations = MODEL_EXPECTATIONS[mode]["slices"]
        sliced_conditionings = []

        if not ConditioningHelper.verify(conditioning, silent=True):
            raise ValueError("Invalid conditioning input format.")

        for entry in conditioning:
            tensor = entry[0]
            meta = entry[1] if len(entry) > 1 and isinstance(entry[1], dict) else {}
            slice_dict = {}

            # Handle explicit slices
            for name, spec in expectations.items():
                source_type, start, end = spec
                if source_type == "full":
                    src = tensor
                elif source_type == "no_token":
                    src = meta.get("pooled_output", None)
                elif source_type == "extended":
                    src = meta.get(name, None)
                else:
                    raise ValueError(f"Unknown slice source type: {source_type}")

                if src is None:
                    continue

                if start is not None:
                    if src.dim() == 3:
                        sliced = src[:, :, start:end]
                    elif src.dim() == 2:
                        sliced = src[:, start:end]
                    else:
                        sliced = src
                else:
                    sliced = src

                slice_dict[name] = sliced.clone() if do_clone else sliced

            # Pull in any other tensors from meta that weren’t explicitly sliced
            for k, v in meta.items():
                if k not in slice_dict and isinstance(v, torch.Tensor):
                    slice_dict[k] = v.clone() if do_clone else v

            sliced_conditionings.append(slice_dict)

        return sliced_conditionings