import comfy
import torch
import torch.nn as nn

from torch.nn.functional import embedding
from typing import List, Optional, Dict, Union
from dataclasses import dataclass







class EncodingType:
    """
    Enum-like class to represent different encoding types.
    """

    VISION = "vision"
    TEXT = "text"
    HYBRID = "hybrid"
    AUDIO = "audio"
    SIMILARITY = "similarity"
    SHUNTED = "shunted"


class GeneratedType:
    """
    Enum-like class to represent different generated types.
    """

    IMAGE = "image"
    TEXT = "text"
    AUDIO = "audio"
    VIDEO = "video"
    SIMILARITY = "similarity"
    SHUNTED = "shunted"


class EncoderPipe:

    def __init__(self):
        self.encoders = {}
        self.config = {}

    def add_encoder(self, encoder, config):
        """
        Adds an encoder to the pipe.
        :param encoder: The encoder to add.
        :param config: Configuration for the encoder.
        """
        # use existing config if it exists; clone and merge with new config
        if encoder.name in self.config:
            existing_config = self.config[encoder.name]
            config = {**existing_config, **config}
        self.encoders[encoder.name] = config

    def clone(self):
        """
        Clones the current EncoderPipe instance.
        :return: A new EncoderPipe instance with the same encoders and config.
        """
        new_pipe = EncoderPipe()
        new_pipe.encoders = self.encoders.copy()
        new_pipe.config = self.config.copy()
        return new_pipe


@dataclass
class EncodedNode:
    # represents a node for the ConditionPipe system
    embedding: torch.Tensor                 # embedding tensor for the node
    config: Optional[dict] = None           # additional configuration for the node
    loggings: dict = None                   # additional logging information for the node

    def log(self, key: str, value: str):
        """
        Logs a key-value pair to the node's logging dictionary.
        :param key: The key to log.
        :param value: The value to log.
        """
        if self.loggings is None:
            self.loggings = {}
        self.loggings[key] = value


class ModelConditioningExpectation:
    """
    Represents a model's expected conditioning models and pooled models.
    This determines how the outputs of a model should be expected for converting to a ComfyUI consumable format.

    This requires at least one expected conditioner and most likely one expected pooled model.
    Omitting a pooled model is acceptable, but will probably fail for most models.
    """

    def __init__(self,
                 model_name: str,
                 expected_conditioners: list = [], # the expected conditioning models
                 expected_pooled: list = [], # the expected pooled models
                 config: Optional[dict] = None):
        self.model_name = model_name
        self.expected_conditioners = expected_conditioners
        self.expected_pooled = expected_pooled
        self.config = config

    # models require certain conditioning sizes and a pooled size
    # these should be streamlined so the user doesn't end up needing to count everything
    # however, the expert may want to substitute things for other things
    # that's why we provide so many projection options with other nodes


    def create(self,
             node: any, #ConditionNode,
             format: str = "comfyui"):
        """
        Creates a ComfyUI consumable representation of the daisy chained nodes.
        """
        #todo: create the translation to ComfyUI consumable format after the surrounding structure is ready
        return node



@dataclass
class SimpleEmbeddingNode:
    """
    Represents a node that contains an embedding tensor.
    This is used to store and process embeddings in the ConditionPipe system.
    """
    name: str  # name of the node
    embedding: torch.Tensor  # embedding tensor for the node


@dataclass
class ConditionEmbeddingNode:
    """
    Represents a single conditioning node in the ConditionPipe system.
    This node can be used to process inputs and generate outputs based on the provided embeddings.
    """
    name: str  # name of the node
    embedding: torch.Tensor  # embedding tensor for the node
    embedding_mask: torch.Tensor # alpha or binary mask for the node, used to control which parts of the embedding are active

    trajectory: torch.Tensor  # trajectory tensor for the node, used as a modifier for the embedding
    trajectory_mask: torch.Tensor  # mask for the trajectory, used to control which parts of the trajectory are active

    symbolic: torch.Tensor  # symbolic tensor for the node, used to represent symbolic information
    symbolic_mask: torch.Tensor  # mask for the symbolic tensor, used to control which parts of the symbolic information are active

    config: Optional[dict] = None  # additional configuration for the node

    def __del__(self):
        """
        Destructor to clean up the node.
        This is called when the node is no longer needed.
        """
        # Clean up resources if necessary, required for anything vram related.
        self.embedding.detach() if self.embedding else None
        self.embedding_mask.detach() if self.embedding_mask else None
        self.trajectory.detach() if  self.trajectory else None
        self.trajectory_mask.detach() if self.trajectory_mask else None
        self.symbolic.detach() if self.symbolic else None
        self.symbolic_mask.detach() if self.symbolic_mask else None



class ConditionPipe:
    """
    The ConditionPipe is the central container for symbolic and trajectory-aware conditioning nodes.

    This object allows for:
        - Dynamic collection of ConditionEmbeddingNodes
        - Alignment and masking operations across symbolic, trajectory, and latent embeddings
        - Model expectation-based routing
        - Exporting to ComfyUI-style conditioning format or future Rosa Cognita protocols

    Each node is aware of its:
        - Embedding vector (conditioning or modulation)
        - Trajectory vector (directional latent guidance)
        - Symbolic vector (conceptual alignment)
        - Associated masking fields for selective control

    This object is meant to:
        - Replace the old list-of-triplets paradigm
        - Enable fine-grained symbolic shunt routing and visual trajectory binding
        - Maintain harmony across models, encoders, and loss systems
    """

    def __init__(self):
        self.nodes: List[ConditionEmbeddingNode] = []

    def add(self, node: 'ConditionEmbeddingNode'):
        """
        Add a single ConditionEmbeddingNode to the pipe.
        Args:
            node (ConditionEmbeddingNode): A node containing the embedding, trajectory, and symbolic fields
        """
        self.nodes.append(node)

    def extend(self, nodes: List['ConditionEmbeddingNode']):
        """
        Add multiple nodes at once.
        Args:
            nodes (List[ConditionEmbeddingNode]): List of nodes to add
        """
        self.nodes.extend(nodes)

    def clone(self) -> 'ConditionPipe':
        """
        Create a copy of the ConditionPipe.
        Useful when branching flows or caching snapshots.
        """
        new_pipe = ConditionPipe()
        new_pipe.nodes = self.nodes.copy()  # Deep copy can be applied if needed
        return new_pipe

    def filter_by_encoder(self, encoder_id: str) -> List['ConditionEmbeddingNode']:
        """
        Retrieve all nodes with a specific encoder ID (from config).
        Args:
            encoder_id (str): The encoder name or ID to match
        Returns:
            List of matching nodes
        """
        return [n for n in self.nodes if n.config and n.config.get("encoder_id") == encoder_id]

    def select_by_role(self, role: str) -> List['ConditionEmbeddingNode']:
        """
        Select all nodes tagged with a specific role (e.g., 'conditioning', 'pooled', 'modulation').
        Args:
            role (str): Role key stored in the config
        """
        return [n for n in self.nodes if n.config and n.config.get("role") == role]

    def extract_tensor_stack(self, field: str = "embedding") -> torch.Tensor:
        """
        Stack a given field (e.g., 'embedding', 'trajectory') across all nodes.
        Args:
            field (str): The tensor attribute to stack from each node
        Returns:
            torch.Tensor: A batch of the requested tensor
        """
        return torch.stack([getattr(n, field) for n in self.nodes])

    def get_node(self, name: str) -> Optional['ConditionEmbeddingNode']:
        """
        Retrieve a node by its name.
        Args:
            name (str): Name field from the node
        Returns:
            The node or None if not found
        """
        return next((n for n in self.nodes if n.name == name), None)

    def to_comfy_conditioning(self) -> List:
        """
        Convert this pipeline into ComfyUI's legacy CONDITIONING format:
            [
                [embedding, {"pooled_output": symbolic}],
                ...
            ]
        Returns:
            List of 2-tuples
        """
        return [[n.embedding, {"pooled_output": n.symbolic}] for n in self.nodes]

    def from_comfy_conditioning(self, conditioning_list: List) -> None:
        """
        Populate this ConditionPipe from ComfyUI's CONDITIONING format.
        Args:
            conditioning_list: List of [embedding, {"pooled_output": X}]
        """
        self.nodes = []
        for i, (embedding, info) in enumerate(conditioning_list):
            name = f"legacy_{i}"
            symbolic = info.get("pooled_output", torch.zeros_like(embedding[:, 0]))  # fallback
            node = ConditionEmbeddingNode(
                name=name,
                embedding=embedding,
                embedding_mask=torch.ones_like(embedding[..., 0]),  # default fully active
                trajectory=torch.zeros_like(embedding),             # blank trajectory
                trajectory_mask=torch.zeros_like(embedding[..., 0]),
                symbolic=symbolic,
                symbolic_mask=torch.ones_like(symbolic[..., 0]),
                config={"encoder_id": name, "role": "conditioning"}
            )
            self.add(node)

    def summary(self) -> str:
        """
        Print a summary of all nodes.
        Returns:
            A string summary
        """
        lines = [f"[ConditionPipe Summary] {len(self.nodes)} nodes"]
        for i, n in enumerate(self.nodes):
            role = n.config.get("role", "unknown") if n.config else "?"
            eid = n.config.get("encoder_id", "none") if n.config else "?"
            lines.append(f"  • [{i}] {n.name}  role={role}, encoder={eid}, shape={tuple(n.embedding.shape)}")
        return "\n".join(lines)
