import torch
from dataclasses import dataclass


@dataclass
class ConditioningData:
    differentiator: str # E.g. "positive", "negative", "undefined"
    identifier: str # Unique identifier for the conditioning (e.g. "clip", "t5", "bert") for pooling
    conditioning_tensor: torch.Tensor # The conditioning tensor, 3D tensor [batch_size, sequence_length, embedding_dim]
    metadata: dict # Additional conditioning metadata for debug, e.g. {"source": "clip", "model": "clip-vit-base-patch16"}
    timestep: dict
    #{  # Timestep information for conditioning
    #    "start": float,  # Start of the timestep range
    #    "end": float,     # End of the timestep range
    #    "width": float,  # Width of the timestep range
    #    "height": float, # Height of the timestep range
    #    "steps": int,   # Number of steps in the timestep range
    #}

class ConditioningContainer:
    def __init__(self, conditionings=None):
        if conditionings is None:
            conditionings = {}
        self.conditionings: dict = conditionings

    def register_conditioning(self, differentiator: str, identifier: str,
                              tensor: torch.Tensor, metadata: dict, timestep: dict):
        self.conditionings[identifier] = ConditioningData(
            differentiator=differentiator,
            identifier=identifier,
            conditioning_tensor=tensor,
            metadata=metadata,
            timestep=timestep
        )

    def get_pair(self, a_id: str, b_id: str):
        a = self.conditionings[a_id].conditioning_tensor
        b = self.conditionings[b_id].conditioning_tensor
        d = b - a
        return a, b, d

    def get_by_diff(self, diff_type: str) -> list:
        return [c for c in self.conditionings.values() if c.differentiator == diff_type]
