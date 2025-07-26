"""
    Trajectory Nodes
    By AbstractPhil

    This module contains the trajectory nodes for the node system.

    These can be encodings, embeddings, entire loras, entire text model weights, whatever.

    Honestly, there is no spoon - and there never was. We just thought there was.

"""

import comfy
import logging



logger = logging.getLogger(__name__)



class TrajectoryNode:
    """
    A node to create a trajectory node for the ConditionPipe system.
    This node allows you to create a trajectory node with an embedding and optional configuration.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any_encodings_1": ("ENCODED_PIPE", ),
                "any_encodings_2": ("ENCODED_PIPE", ),
                "config": ("DICT", {"default": {}}),
            },
        }

    RETURN_TYPES = ("TRAJECTORY_NODE", )
    RETURN_NAMES = ("trajectory_node", )
    FUNCTION = "create_trajectory_node"
    CATEGORY = "utils/trajectory"

    def create_trajectory_node(self, any_encodings_1: list,
                               any_encodings_2: list,
                               config: dict = None) -> tuple:
        """
        Creates a trajectory node with the provided encodings and configuration.
        :param any_encodings_1: The first set of encodings.
        :param any_encodings_2: The second set of encodings.
        :param config: Optional configuration for the trajectory node.
        :return: A tuple containing the created trajectory node.
        """
        if config is None:
            config = {}

        trajectory_node = {
            "encodings_1": any_encodings_1,
            "encodings_2": any_encodings_2,
            "config": config
        }

        return (trajectory_node, )