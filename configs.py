from __future__ import annotations
from typing import Optional
import logging


DEFAULT_REPOS = {
    "clip_g": "laion/CLIP-ViT-bigG-14-laion2B-s32B-b79K",
    "vit-bigG-14": "laion/CLIP-ViT-bigG-14-laion2B-s32B-b79K",
    "clip_l": "openai/clip-vit-large-patch14",
    "vit-l-14": "openai/clip-vit-large-patch14",
    "clip_l_4h": "openai/clip-vit-large-patch14",
    "clip_l_x52": "AbstractPhil/beatrix-x52",
    "clip_h": "openai/clip-vit-large-patch14-h",
    "clip_vision": "openai/clip-vit-large-patch14-vision",
    "t5_base": "google/flan-t5-base",
    "t5_small": "google-t5/t5-small",
    "t5_unchained": "AbstractPhil/t5-unchained",
    "bert_beatrix": "AbstractPhil/bert-beatrix-2048",
    "nomic_bert": "nomic-ai/nomic-bert-2048",
    "mobilebert": "google/mobilebert-uncased",
    "bert_base_uncased": "bert-base-uncased",
    "bert_large_uncased": "bert-large-uncased",
    "bert_base_cased": "bert-base-cased",
    "bert_base_multilingual": "bert-base-multilingual-uncased",
    "bert_base_multilingual_cased": "bert-base-multilingual-cased",
}


HARMONIC_SHUNT_REPOS = {
    "clip_g": {
        "models": ["clip_g", 't5_base'],
        "repo": "AbstractPhil/t5-flan-base-vit-bigG-14-dual-stream-adapter",
        "shunts_available": {
            "shunt_type_name": "DualStreamAdapter-G",
            "config_file_name": "config.json",
            "shunt_list": [
                "t5-flan-vit-bigG-14-dual_shunt_caption.safetensors",
                "t5-flan-vit-bigG-14-dual_shunt_no_caption_e1.safetensors",
                "t5-flan-vit-bigG-14-dual_shunt_no_caption_e2.safetensors",
                "t5-flan-vit-bigG-14-dual_shunt_no_caption_e3.safetensors",
                "t5-flan-vit-bigG-14-dual_shunt_summarize.safetensors",
                "dual_shunt_omega_no_caption_e1_step_10000.safetensors",
                "dual_shunt_omega_no_caption_noised_e1_step_1000.safetensors",
                "dual_shunt_omega_no_caption_noised_e1_step_4000.safetensors",
                "dual_shunt_omega_no_caption_noised_e1_step_10000.safetensors",
            ],
        },
        "config": {
            "adapter_id": "003", "name": "DualShuntAdapter-G",
            "condition_encoders": [{
                "type": "t5_base",
                "model": "google/flan-t5-base",
                "hidden_size": 768
            }],
            "modulation_encoders": [
                {
                    "type": "clip_g",
                    "model": "openai/clip-vit-large-patch14",
                    "hidden_size": 1280
                }
            ],
            "hidden_size": 1280,  # This is the adapter's output size
            "bottleneck": 640, "heads": 20,
            "max_guidance": 10.0, "tau_init": 0.1,
            "proj_layers": 2, "layer_norm": True, "dropout": 0.0,
            "use_dropout": False, "use_proj_stack": True, "assert_input_dims": True,
            "routing": {"type": "cross_attention", "enable_causal_mask": False, "bidirectional": True},
            "version": "v0.3.2"
        },

    },
    "clip_l_4h": {
        "models": ["vit-l-14", 'flan-t5-base'],
        "repo": "AbstractPhil/t5-flan-base-vit-l-14-dual-stream-adapter",
        "shunts_available": {
            "shunt_type_name": "DualStreamAdapter-L",
            "config_file_name": "config.json",
            "shunt_list": [
                "t5-vit-l-14-dual_shunt_booru_13_000_000.safetensors",
                "t5-vit-l-14-dual_shunt_booru_51_200_000.safetensors"
            ],
        },
        "config": {
            "adapter_id": "003",
            "name": "DualShuntAdapter",
            "condition_encoders": [{
                "type": "t5_base",
                "model": "google/flan-t5-base",
                "hidden_size": 768
            }],
            "modulation_encoders": [{
                "type": "clip_l",
                "model": "openai/clip-vit-large-patch14",
                "hidden_size": 768
            }],
            "hidden_size": 768,  # This is the adapter's output size
            "bottleneck": 384, "heads": 4,
            "max_guidance": 10.0, "tau_init": 0.1,
            "proj_layers": 2, "layer_norm": True, "dropout": 0.0,
            "use_dropout": False, "use_proj_stack": True, "assert_input_dims": True,
            "routing": {"type": "cross_attention", "enable_causal_mask": False, "bidirectional": True},
            "version": "v0.3.2"
        },
    },
    "clip_l_x52": {
        "models": ['bert-beatrix-2048', 'vit-l-14'],
        "repo": "AbstractPhil/beatrix-x52",
        "shunts_available": {
            "shunt_type_name": "HarmonicBank-x52",
            "config_file_name": "config.json",
            "shunt_list": [
                "AbstractPhil/beatrix-x52-v0001.safetensors",
            ]
        },
        "config": {
            "adapter_id": "072",
            "name": "DualShuntAdapter",
            "condition_encoders": [
                {
                    "model": "AbstractPhil/bert-beatrix-2048"
                },
            ],
            "modulation_encoders": [
                {
                    "model": "openai/clip-vit-large-patch14",
                    "offset_slip": 0.0,
                    "slip_frequency": 0.27
                },
                {
                    "model": "AbstractPhil/clips/Omega-SIM-ViT-CLIP_L_FP32.safetensors",
                    "offset_slip": 1.0,
                    "slip_frequency": 0.27
                },
                {
                    "model": "AbstractPhil/clips/ComfyUI_noobxl-R9_clip_l.safetensors",
                    "offset_slip": 2.0,
                    "slip_frequency": 0.27
                },
                {
                    "model": "AbstractPhil/clips/SIM-VPRED-Ω-73-clip_l.safetensors",
                    "offset_slip": 3.0,
                    "slip_frequency": 0.27
                },
            ],
            "hidden_size": 768,  # This is the adapter's output size
            "resonant_heads": 2000, # number of resonant heads to distribute across the shunts
            "spin": 0.5,  # the spin factor that this shunt was trained on
            "conv_frequency": 0.29152, # the differentiation frequency of each rope layered phase
            "conv_layers": 52000, # number of convolutional layers
            "use_bottleneck": False,  # bottleneck can be enabled for a much more expensive overhead
            "bottleneck": 32,  # This is the bottleneck dim size per shunt if used
            "heads": 2, # number of heads per shunt if bottleneck enabled eg 104,000 heads total which is a shitload
            "max_guidance": 10.0,
            "tau_init": 0.1,
            "proj_layers": 16,
            "layer_norm": True,
            "dropout": 0.0,
            "use_dropout": False,
            "use_proj_stack": True,
            "assert_input_dims": True,
            "routing": {
                "type": "phase_gate",
                "math": "tau",
                "rope_phase_offset": 0.0,
                "omnidirectional": False,
                "loosely_coupled": True,
            },
            "version": "v2.0.0",

        },
    },
    "clip_l": {
        "models": ["vit-l-14", 'flan-t5-base'],
        "config": {
            "adapter_id": "002",
            "name": "DualShuntAdapter",
            "condition_encoders": {
                "type": "t5_base",
                "model": "google/flan-t5-base",
                "hidden_size": 768
            },
            "modulation_encoders": {
                "type": "clip_l",
                "model": "openai/clip-vit-large-patch14",
                "hidden_size": 768
            },
            "hidden_size": 768,  # This is the adapter's output size
            "bottleneck": 384, "heads": 12,
            "max_guidance": 10.0, "tau_init": 0.1,
            "proj_layers": 2, "layer_norm": True, "dropout": 0.0,
            "use_dropout": False, "use_proj_stack": True, "assert_input_dims": True,
            "routing": {"type": "cross_attention", "enable_causal_mask": False, "bidirectional": True},
            "version": "v0.3.2"
        },
        "repo": "AbstractPhil/t5-flan-base-vit-l-14-dual-stream-adapter",
        "shunts_available": {
            "shunt_type_name": "DualStreamAdapter-L",
            "config_file_name": "config.json",
            "shunt_list": [
                "t5-vit-l-14-dual_shunt_caption.safetensors",
                "t5-vit-l-14-dual_shunt_no_caption.safetensors",
                "t5-vit-l-14-dual_shunt_summarize.safetensors",
            ],
        },
    }
}

# ─── Adapter Configs ─────────────────────────────────────────────

BERT_CONFIGS = {
    "bert-beatrix-2048": {
        "repo_name": "AbstractPhil/bert-beatrix-2048",
        "use_huggingface": True,  # defaults to simple loading from HuggingFace, if False, will use repo_name and subfolder
        "use_remote_code": True,
        "subfolder": "",
    },
    "nomic-bert-2048": {
        "repo_name": "nomic-ai/nomic-bert-2048",
        "use_huggingface": True,  # defaults to simple loading from HuggingFace, if False, will use repo_name and subfolder
        "use_remote_code": True,
        "subfolder": "",
    },
    "mobilebert-base-uncased": {
        "repo_name": "google/mobilebert-uncased",
        "use_huggingface": True,  # defaults to simple loading from HuggingFace, if False, will use repo_name and subfolder
        "subfolder": "",
    },
    "bert-base-uncased": {
        "repo_name": "bert-base-uncased",
        "use_huggingface": True,  # defaults to simple loading from HuggingFace, if False, will use repo_name and subfolder
    },
    "bert-large-uncased": {
        "repo_name": "bert-large-uncased",
        "use_huggingface": True,  # defaults to simple loading from HuggingFace, if False, will use repo_name and subfolder
    },
    "bert-base-cased": {
        "repo_name": "bert-base-cased",
        "use_huggingface": True,  # defaults to simple loading from HuggingFace, if False, will use repo_name and subfolder
    }
}

T5_CONFIGS = {
    "t5xxl": {
        "repo_name": "google/t5-xxl-lm-adapt",
        "use_huggingface": True,  # defaults to simple loading from HuggingFace, if False, will use repo_name and subfolder
        "subfolder": "",
    },
    "t5-unchained": {
        "repo_name": "AbstractPhil/t5-unchained",
        "use_huggingface": True,  # defaults to simple loading from HuggingFace, if False, will use repo_name and subfolder
        "subfolder": "",
        "tokenizer": "t5-unchained",
        "file_name": "model.safetensors",
        "config": {
            "config_file_name": "config.json",
            "architectures": [
                "T5ForConditionalGeneration"
            ],
            "attention_dropout": 0.0,
            "classifier_dropout": 0.0,

            "d_ff": 10240,
            "d_kv": 64,
            "d_model": 4096,
            "decoder_start_token_id": 0,
            "dropout_rate": 0.0,
            "eos_token_id": 1,
            "dense_act_fn": "gelu_pytorch_tanh",
            "initializer_factor": 1.0,
            "is_encoder_decoder": True,
            "is_gated_act": True,
            "layer_norm_epsilon": 1e-06,
            "model_type": "t5",
            "num_decoder_layers": 24,
            "num_heads": 64,
            "num_layers": 24,
            "output_past": True,
            "pad_token_id": 0,
            "relative_attention_num_buckets": 32,
            "tie_word_embeddings": False,
            "vocab_size": 69328,
        }
    },
    "flan-t5-base": {
        "repo_name": "google/flan-t5-base",
        "use_huggingface": True, # defaults to simple loading from HuggingFace, if False, will use repo_name and subfolder
    },
    "t5-small": {
        "repo_name": "google-t5/t5-small",
        "use_huggingface": True, # defaults to simple loading from HuggingFace, if False, will use repo_name and subfolder
    },
    "t5_small_human_attentive_try2_pass3": {
        "repo_name": "AbstractPhil/t5_small_human_attentive_try2_pass3",
        "use_huggingface": True, # defaults to simple loading from HuggingFace, if False, will use repo_name and subfolder
        # the necessary config is present here for posterity in case it fails to load from HuggingFace.
        "subfolder": "",
        "tokenizer": "t5-small",
        "file_name": "model.safetensors",
        "config": {
              "config_file_name": "config.json",
              "architectures": [
                "T5ForConditionalGeneration"
              ],
              "attention_dropout": 0.0,
              "classifier_dropout": 0.0,
              "d_ff": 2048,
              "d_kv": 64,
              "d_model": 512,
              "decoder_start_token_id": 0,
              "dense_act_fn": "relu",
              "dropout_rate": 0.0, #0.3,                  # disable for generation
              "eos_token_id": 1,
              "feed_forward_proj": "relu",
              "initializer_factor": 1.0,
              "is_encoder_decoder": True,
              "is_gated_act": False,
              "layer_norm_epsilon": 1e-06,
              "model_type": "t5",
              "n_positions": 512,
              "num_decoder_layers": 6,
              "num_heads": 8,
              "num_layers": 6,
              "output_past": True,
              "pad_token_id": 0,
              "relative_attention_max_distance": 128,
              "relative_attention_num_buckets": 32,
              "task_specific_params": {
                "caption": {
                  "early_stopping": True,
                  "length_penalty": 1.0,
                  "max_length": 64,
                  "num_beams": 4,
                  "prefix": "caption: "
                }
              },
              "torch_dtype": "float32",
              "transformers_version": "4.51.3",
              "use_cache": True,
              "vocab_size": 32128
        }
    }
}



SHUNTS = []
""" 
    Populates the shunts list with available shunts from all the shunt dictionaries.
"""

for shunt_dict in HARMONIC_SHUNT_REPOS.values():
    if "shunts_available" in shunt_dict:
        shunts = shunt_dict["shunts_available"]["shunt_list"]
        for shunt in shunts:
            # populate the shunts list with a reference to the shunt dictionary
            SHUNTS.append({
                "name": shunt,
                "repo": shunt_dict["repo"],
                "config": shunt_dict["config"],
                "expected": shunt_dict["models"],
                "modulation_encoders": shunt_dict["config"]["modulation_encoders"],
                "condition_encoders": shunt_dict["config"]["condition_encoders"],
                "shunt_type_name": shunt_dict["shunts_available"]["shunt_type_name"],
                "config_file_name": shunt_dict["shunts_available"]["config_file_name"]
            })




class ShuntUtil:

    @staticmethod
    def get_encoder_repos_by_shunt_name(shunt_name: str) -> list[str]:
        """
        Returns the repository name of the encoder associated with the given shunt name.

        Args:
            shunt_name (str): The name of the shunt to search for.

        Returns:
            Optional[str]: The repository name if found, otherwise None.
        """
        shunt = ShuntUtil.get_shunt_by_name(shunt_name)
        prepared = []
        if shunt:
            for model in shunt["expected"]:
                if model in DEFAULT_REPOS:
                    prepared.append(DEFAULT_REPOS[model])
                else:
                    logging.warning(f"Model '{model}' not found in default repositories.")
            return prepared
        else:
            logging.warning(f"Shunt '{shunt_name}' not found.")

        return None

    @staticmethod
    def get_shunts_by_expected_model(model_name: str) -> list[dict]:
        """
        Returns a list of shunt configurations that match the expected model name.

        Args:
            model_name (str): The name of the model to filter shunts by.

        Returns:
            list[dict]: A list of shunt configuration dictionaries.
        """
        return [shunt for shunt in SHUNTS if model_name in shunt["repo"]]

    @staticmethod
    def get_shunt_by_name(name: str) -> Optional[dict]:
        """
        Returns the shunt configuration dictionary by its name.

        Args:
            name (str): The name of the shunt to retrieve.

        Returns:
            Optional[dict]: The shunt configuration dictionary if found, otherwise None.
        """
        for shunt in SHUNTS:
            if shunt["name"] == name:
                return shunt
        logging.warning(f"Shunt '{name}' not found.")
        return None

    @staticmethod
    def get_shunt_names() -> list[str]:
        """
        Returns a list of all available shunt names.

        Returns:
            list[str]: List of shunt names.
        """
        return [shunt["name"] for shunt in SHUNTS]


    @staticmethod
    def get_shunt_config_by_name(name: str) -> Optional[dict]:
        """
        Returns the shunt configuration by its name.

        Args:
            name (str): The name of the shunt to retrieve.

        Returns:
            Optional[dict]: The shunt configuration dictionary if found, otherwise None.
        """
        shunt = ShuntUtil.get_shunt_by_name(name)
        if shunt:
            return shunt["config"]
        return None