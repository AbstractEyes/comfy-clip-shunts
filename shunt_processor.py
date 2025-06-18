"""
    Shunt PreProcessor and Processor Classes
    Author: AbstractPhil

    Description: This module contains the Shunt PreProcessor and Processor classes
    These are used to handle the various model interactions and configurations
    for the Shunt adapters in ComfyUI.

    The ShuntUtil class is used to manage configurations and utility functions,
    while this class is used to process inputs and outputs for the Shunt adapters directly.

"""

from .configs import ShuntUtil
from .model_manager import get_model_manager

manager = get_model_manager() # singleton instance of ModelManager


class ShuntPreProcessor:
    """
    PreProcessor for Shunt adapters.
        Handles the input preprocessing for Shunt adapters,
        The model stack must validate or project the input to the expected dimensions.
    """

    def __init__(self, shunt_config):
        # retains only the config to prevent vram from accumulating
        self.shunt_config = shunt_config

    def get_target_models(self):
        """
        Returns the target models for the Shunt adapter based on the configuration.
        """
        return ShuntUtil.get(self.config)

    def validate_stack(self):
        # check if the model stack using the model_manager
        if not manager.is_loaded(self.config):
            raise ValueError(f"Invalid model stack configuration: {self.config}")

    def preprocess(self, input_data):
        if not self.validate_stack():
            raise ValueError("Model stack is not valid or not loaded.")
        # Preprocess the input data according to the Shunt configuration






# ─── Shunt PreProcessor and Processor Classes ─────────────────────────────────────