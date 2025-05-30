T5_SHUNT_REPOS = {
    "clip_g": {
        "models": ["vit-bigG-14", 'flan-t5-base'],
        "config": {
            "adapter_id": "003", "name": "DualShuntAdapter-G",
            "t5": {
                "model": "google/flan-t5-base",
                "hidden_size": 768
            },
            "clip": {
                "model": "openai/clip-vit-large-patch14",
                "hidden_size": 1280
            },
            "hidden_size": 1280,  # This is the adapter's output size
            "bottleneck": 640, "heads": 20,
            "tau_init": 0.1, "max_guidance": 10.0,
            "proj_layers": 2, "layer_norm": True, "dropout": 0.1,
            "use_dropout": True, "use_proj_stack": True, "assert_input_dims": True,
            "routing": {"type": "cross_attention", "enable_causal_mask": False, "bidirectional": True},
            "version": "v0.3.2"
        },
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
            ],
        }
    },
    "clip_l": {
        "models": ["vit-l-14", 'flan-t5-base'],
        "config": {
            "adapter_id": "002",
            "name": "DualShuntAdapter",
            "t5": {"model": "google/flan-t5-base", "hidden_size": 768},
            "clip": {"model": "openai/clip-vit-large-patch14", "hidden_size": 768},
            "hidden_size": 768,  # This is the adapter's output size
            "bottleneck": 384, "heads": 12,
            "tau_init": 0.1, "max_guidance": 10.0,
            "proj_layers": 2, "layer_norm": True, "dropout": 0.1,
            "use_dropout": True, "use_proj_stack": True, "assert_input_dims": True,
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
              "attention_dropout": 0.3,
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