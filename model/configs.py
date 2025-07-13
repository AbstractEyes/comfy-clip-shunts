from __future__ import annotations
from typing import Optional
import logging
logger = logging.getLogger(__name__)




DEFAULT_REPOS = {
    "clip_g": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    "vit-bigG-14": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
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
        "models": ['t5_base', "clip_g"],
        "repo": "AbstractPhil/t5-flan-base-vit-bigG-14-dual-stream-adapter",
        "shunts_available": {
            "shunt_type_name": "DualStreamAdapter-G",
            "config_file_name": "config.json",
            "shunt_list": [
                {"name": "flan-t5-base+clip_g:caption", "file": "t5-flan-vit-bigG-14-dual_shunt_caption.safetensors" },
                {"name": "flan-t5-base+clip_g:noise-e1", "file": "t5-flan-vit-bigG-14-dual_shunt_no_caption_e1.safetensors" },
                {"name": "flan-t5-base+clip_g:noise-e3", "file": "t5-flan-vit-bigG-14-dual_shunt_no_caption_e2.safetensors" },
                {"name": "flan-t5-base+clip_g:noise-e3", "file": "t5-flan-vit-bigG-14-dual_shunt_no_caption_e3.safetensors" },
                {"name": "flan-t5-base+clip_g:summarize", "file": "t5-flan-vit-bigG-14-dual_shunt_summarize.safetensors" },
                {"name": "flan-t5-base+clip_g_omega32:noise-10000", "file": "dual_shunt_omega_no_caption_e1_step_10000.safetensors" },
                {"name": "flan-t5-base+clip_g_omega32:noise-1000", "file": "dual_shunt_omega_no_caption_noised_e1_step_1000.safetensors" },
                {"name": "flan-t5-base+clip_g_omega32:noise-4000", "file": "dual_shunt_omega_no_caption_noised_e1_step_4000.safetensors" },
                {"name": "flan-t5-base+clip_g_omega32:noise-10000v2", "file": "dual_shunt_omega_no_caption_noised_e1_step_10000.safetensors" },
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
                    "model": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
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
    "clip_g_8h": {
        "models": ['bert_beatrix-2048', "clip_g"],
        "repo": "AbstractPhil/bert-beatrix-2048-vit-bigG-14-dual-shunt-adapter",
        "shunts_available": {
            "shunt_type_name": "DualStreamAdapter-G",
            "config_file_name": "config.json",
            "shunt_list": [
                {"name": "beatrix+clip_g-laion_v1_8h:noise-e1", "file": "dual_shunt_g_laion_no_caption_e1.safetensors"},
                {"name": "beatrix+clip_g-booru_v1_8h:noise-3000", "file": "dual_shunt_g_booru_no_caption_noised_e1_step_3000.safetensors"},
                {"name": "beatrix+clip_g-booru_v1_8h:noise-5000", "file": "dual_shunt_g_booru_no_caption_noised_e1_step_5000.safetensors"},
                {"name": "beatrix+clip_g-booru_v1_8h:noise-7000", "file": "dual_shunt_g_booru_no_caption_noised_e1_step_7000.safetensors"},
                {"name": "beatrix+clip_g-booru_v1_8h:noise-10000", "file": "dual_shunt_g_booru_no_caption_noised_e1_step_10000.safetensors"},
                {"name": "beatrix+clip_g-booru_v1_8h:noise-14000", "file": "dual_shunt_g_booru_no_caption_noised_e1_step_14000.safetensors"},
                {"name": "beatrix+clip_g-booru_v1_8h:noise-20000", "file": "dual_shunt_g_booru_no_caption_noised_e1_step_20000.safetensors"},
                {"name": "beatrix+clip_g-booru_v1_8h:noise-25000", "file": "dual_shunt_g_booru_no_caption_noised_e1_step_25000.safetensors"},
                {"name": "beatrix+clip_g-booru_v1_8h:noise-30000", "file": "dual_shunt_g_booru_no_caption_noised_e1_step_30000.safetensors"},
                {"name": "beatrix+clip_g-booru_v1_8h:noise-e1", "file": "dual_shunt_g_booru_no_caption_noised_e1_final.safetensors"},
                {"name": "beatrix+clip_g-laion_v1_8h:noise-e2", "file": "dual_shunt_g_laion_no_caption_e2.safetensors"},
                {"name": "beatrix+clip_g-laion_v1_8h:noise-e5", "file": "dual_shunt_g_laion_no_caption_e5.safetensors"},
                {"name": "beatrix+clip_g-laion_v1_8h:noise-e8", "file": "dual_shunt_g_laion_no_caption_e8.safetensors"},
                {"name": "beatrix+clip_g-laion_v1_8h:noise-e10", "file": "dual_shunt_g_laion_no_caption_e10.safetensors"},
                {"name": "beatrix+clip_g-laion_v1_8h:noise-e12", "file": "dual_shunt_g_laion_no_caption_e12.safetensors"},
                {"name": "beatrix+clip_g-laion_v1_8h:noise-e15", "file": "dual_shunt_g_laion_no_caption_e15.safetensors"},
                {"name": "beatrix+clip_g-laion_v1_8h:noise-e16", "file": "dual_shunt_g_laion_no_caption_e16.safetensors"},
                {"name": "beatrix+clip_g-laion_v1_8h:noise-e17", "file": "dual_shunt_g_laion_no_caption_e17.safetensors"},
                {"name": "beatrix+clip_g-laion_v1_8h:noise-e18", "file": "dual_shunt_g_laion_no_caption_e18.safetensors"},
                {"name": "beatrix+clip_g-laion_v1_8h:noise-e19", "file": "dual_shunt_g_laion_no_caption_e19.safetensors"},
                {"name": "beatrix+clip_g-laion_v1_8h:noise-e20", "file": "dual_shunt_g_laion_no_caption_e20.safetensors"},
            ],
        },
        "config": {
            "adapter_id": "006", "name": "DualShuntAdapter-G",
            "condition_encoders": [{
                "type": "t5_base",
                "model": "AbstractPhil/bert-beatrix-2048",
                "hidden_size": 768
            }],
            "modulation_encoders": [
                {
                    "type": "clip_g",
                    "model": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
                    "hidden_size": 1280
                }
            ],
            "hidden_size": 1280,  # This is the adapter's output size
            "bottleneck": 640, "heads": 8,
            "max_guidance": 10.0, "tau_init": 0.1,
            "proj_layers": 2, "layer_norm": True, "dropout": 0.0,
            "use_dropout": False, "use_proj_stack": True, "assert_input_dims": True,
            "routing": {"type": "cross_attention", "enable_causal_mask": False, "bidirectional": True},
            "version": "v0.3.2"
        },
    },
#
    "clip_l_4h_bert": {
        "models": ["bert-beatrix-2048", "clip_l"],
        "repo": "AbstractPhil/bert-beatrix-2048-vit-l-14-dual-shunt-adapter",
        "shunts_available": {
            "shunt_type_name": "DualStreamAdapter-L",
            "config_file_name": "config.json",
            "shunt_list": [
                {"name": "beatrix+clip_l-4h:noise_5000", "file": "dual_shunt_l_booru_no_caption_noised_e1_step_5000.safetensors"},
                {"name": "beatrix+clip_l-4h:noise_10000", "file": "dual_shunt_l_booru_no_caption_noised_e1_step_10000.safetensors"},
                {"name": "beatrix+clip_l-4h:noise_15000", "file": "dual_shunt_l_booru_no_caption_noised_e1_step_15000.safetensors"},
                {"name": "beatrix+clip_l-4h:noise_20000", "file": "dual_shunt_l_booru_no_caption_noised_e1_step_20000.safetensors"},
                {"name": "beatrix+clip_l-4h:noise_25000", "file": "dual_shunt_l_booru_no_caption_noised_e1_step_25000.safetensors"},
            ],
        },
        "config": {
            "adapter_id": "005",
            "name": "DualShuntAdapter",
            "condition_encoders": [{
                "type": "bert_beatrix",
                "model": "AbstractPhil/bert-beatrix-2048",
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
    "clip_l_4h": {
        "models": ["vit-l-14", 'flan-t5-base'],
        "repo": "AbstractPhil/t5-flan-base-vit-l-14-dual-stream-adapter",
        "shunts_available": {
            "shunt_type_name": "DualStreamAdapter-L",
            "config_file_name": "config.json",
            "shunt_list": [
                {"name": "t5+clip_l_4h:noise_13m", "file": "t5-vit-l-14-dual_shunt_booru_13_000_000.safetensors"},
                {"name": "t5+clip_l_4h:noise_51.2m", "file": "t5-vit-l-14-dual_shunt_booru_51_200_000.safetensors"}
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
                {"name": "beatrix-x52", "file": "AbstractPhil/beatrix-x52-v0001.safetensors"},
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
        "models": ['flan-t5-base', "vit-l-14"],
        "repo": "AbstractPhil/t5-flan-base-vit-l-14-dual-stream-adapter",
        "config": {
            "adapter_id": "002",
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
            "bottleneck": 384, "heads": 12,
            "max_guidance": 10.0, "tau_init": 0.1,
            "proj_layers": 2, "layer_norm": True, "dropout": 0.0,
            "use_dropout": False, "use_proj_stack": True, "assert_input_dims": True,
            "routing": {"type": "cross_attention", "enable_causal_mask": False, "bidirectional": True},
            "version": "v0.3.2"
        },
        "shunts_available": {
            "shunt_type_name": "DualStreamAdapter-L",
            "config_file_name": "config.json",
            "shunt_list": [
                {"name": "t5+clip_l_12h:caption-20m", "file": "t5-vit-l-14-dual_shunt_caption.safetensors" },
                {"name": "t5+clip_l_12h:noise-20m", "file": "t5-vit-l-14-dual_shunt_no_caption.safetensors" },
                {"name": "t5+clip_l_12h:summarize-20m", "file": "t5-vit-l-14-dual_shunt_summarize.safetensors" },
            ],
        },
    },
    #AbstractPhil/bert-beatrix-2048-noobxl-epsilon-v11-dual-shunt-adapter_clip_g
    "clip_g_noob": {
        "models": ['bert-beatrix-2048', "clip_g"],
        "repo": "AbstractPhil/bert-beatrix-2048-noobxl-epsilon-v11-dual-shunt-adapter_clip_g",
        "config": {
            "adapter_id": "006",
            "name": "DualShuntAdapter",
            "condition_encoders": [{
                "type": "bert_beatrix",
                "model": "AbstractPhil/bert-beatrix-2048",
                "hidden_size": 768
            }],
            "modulation_encoders": [{
                "type": "clip_g",
                "model": "AbstractPhil/clips/NAI-11-epsilon_clip_g.safetensors",
                "hidden_size": 1280
            }],
            "hidden_size": 1280,  # This is the adapter's output size
            "bottleneck": 640, "heads": 8,
            "max_guidance": 10.0, "tau_init": 0.1,
            "proj_layers": 2, "layer_norm": True, "dropout": 0.0,
            "use_dropout": False, "use_proj_stack": True, "assert_input_dims": True,
            "routing": {"type": "cross_attention", "enable_causal_mask": False, "bidirectional": True},
            "version": "v0.3.2"
        },
        "shunts_available": {
            "shunt_type_name": "DualStreamAdapter-G",
            "config_file_name": "config.json",
            "shunt_list": [
                {"name": "beatrix+noob_g_8h:noised-1000", "file": "dual_shunt_g_noob_no_caption_e1_step_1000.safetensors"},
                {"name": "beatrix+noob_g_8h:noised-2000", "file": "dual_shunt_g_noob_no_caption_e1_step_2000.safetensors"},
                {"name": "beatrix+noob_g_8h:noised-3000", "file": "dual_shunt_g_noob_no_caption_e1_step_3000.safetensors"},
                {"name": "beatrix+noob_g_8h:noised-4000", "file": "dual_shunt_g_noob_no_caption_e1_step_4000.safetensors"},
                {"name": "beatrix+noob_g_8h:noised-5000", "file": "dual_shunt_g_noob_no_caption_e1_step_5000.safetensors"},
                {"name": "beatrix+noob_g_8h:noised-6000", "file": "dual_shunt_g_noob_no_caption_e1_step_6000.safetensors"},
                {"name": "beatrix+noob_g_8h:noised-7000", "file": "dual_shunt_g_noob_no_caption_e1_step_7000.safetensors"},
                {"name": "beatrix+noob_g_8h:noised-8000", "file": "dual_shunt_g_noob_no_caption_e1_step_8000.safetensors"},
            ]
        },
    },


    "clip_l_noob": {
        "models": ['bert-beatrix-2048', "vit-l-14"],
        "repo": "AbstractPhil/bert-beatrix-2048-noobxl-epsilon-v11-dual-shunt-adapter",
        "config": {
            "adapter_id": "002",
            "name": "DualShuntAdapter",
            "condition_encoders": [{
                "type": "bert_beatrix",
                "model": "AbstractPhil/bert-beatrix-2048",
                "hidden_size": 768
            }],
            "modulation_encoders": [{
                "type": "clip_l",
                "model": "AbstractPhil/clips/NAI-11-epsilon_clip_l.safetensors",
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
        "shunts_available": {
            "shunt_type_name": "DualStreamAdapter-L",
            "config_file_name": "config.json",
            "shunt_list": [
                {"name": "beatrix+noob_l_4h:noise-5000", "file": "beatrix_dual_shunt_l_noob_e1_step_5000.safetensors"},
                {"name": "beatrix+noob_l_4h:noise-10000", "file": "beatrix_dual_shunt_l_noob_e1_step_10000.safetensors"},
                {"name": "beatrix+noob_l_4h:noise-15000", "file": "beatrix_dual_shunt_l_noob_e1_step_15000.safetensors"},
                {"name": "beatrix+noob_l_4h:noise-20000", "file": "beatrix_dual_shunt_l_noob_e1_step_20000.safetensors"},
                {"name": "beatrix+noob_l_4h:noise-25000", "file": "beatrix_dual_shunt_l_noob_e1_step_25000.safetensors"},
                {"name": "beatrix+noob_l_4h:noise-30000", "file": "beatrix_dual_shunt_l_noob_e1_step_30000.safetensors"},
                {"name": "beatrix+noob_l_4h:noise-35000", "file": "beatrix_dual_shunt_l_noob_e1_step_35000.safetensors"},
                {"name": "beatrix+noob_l_4h:noise-40000", "file": "beatrix_dual_shunt_l_noob_e1_step_40000.safetensors"},
                {"name": "beatrix+noob_l_4h:noise-45000", "file": "beatrix_dual_shunt_l_noob_e1_step_45000.safetensors"},
                {"name": "beatrix+noob_l_4h:noise-50000", "file": "beatrix_dual_shunt_l_noob_e1_step_50000.safetensors"},
                {"name": "beatrix+noob_l_4h:noise-55000", "file": "beatrix_dual_shunt_l_noob_e1_step_55000.safetensors"},
                {"name": "beatrix+noob_l_4h:noise-60000", "file": "beatrix_dual_shunt_l_noob_e1_step_60000.safetensors"},
                {"name": "beatrix+noob_l_4h:noise-65000", "file": "beatrix_dual_shunt_l_noob_e1_step_65000.safetensors"},
                {"name": "beatrix+noob_l_4h:noise-70000", "file": "beatrix_dual_shunt_l_noob_e1_step_70000.safetensors"},
                {"name": "beatrix+noob_l_4h:noise-75000", "file": "beatrix_dual_shunt_l_noob_e1_step_75000.safetensors"},
                {"name": "beatrix+noob_l_4h:noise-80000", "file": "beatrix_dual_shunt_l_noob_e1_step_80000.safetensors"},
                {"name": "beatrix+noob_l_4h:noise-85000", "file": "beatrix_dual_shunt_l_noob_e1_step_85000.safetensors"},
                {"name": "beatrix+noob_l_4h:noise-90000", "file": "beatrix_dual_shunt_l_noob_e1_step_90000.safetensors"},
                {"name": "beatrix+noob_l_4h:noise-100000", "file": "beatrix_dual_shunt_l_noob_e1_step_100000.safetensors"},
                {"name": "beatrix+noob_l_4h:noise-120000", "file": "beatrix_dual_shunt_l_noob_e1_step_120000.safetensors"},
                {"name": "beatrix+noob_l_4h:noise-140000", "file": "beatrix_dual_shunt_l_noob_e1_step_140000.safetensors"},
                {"name": "beatrix+noob_l_4h:noise-150000", "file": "beatrix_dual_shunt_l_noob_e1_step_150000.safetensors"},
                {"name": "beatrix+noob_l_4h:noise-160000", "file": "beatrix_dual_shunt_l_noob_e1_step_160000.safetensors"},
                {"name": "beatrix+noob_l_4h:noise-180000", "file": "beatrix_dual_shunt_l_noob_e1_step_180000.safetensors"},
                {"name": "beatrix+noob_l_4h:noise-200000", "file": "beatrix_dual_shunt_l_noob_e1_step_200000.safetensors"},
                {"name": "beatrix+noob_l_4h:noise-225000", "file": "beatrix_dual_shunt_l_noob_e1_step_225000.safetensors"},
                {"name": "beatrix+noob_l_4h:noise-250000", "file": "beatrix_dual_shunt_l_noob_e1_step_250000.safetensors"},
                {"name": "beatrix+noob_l_4h:noise-275000", "file": "beatrix_dual_shunt_l_noob_e1_step_275000.safetensors"},
                {"name": "beatrix+noob_l_4h:noise-300000", "file": "beatrix_dual_shunt_l_noob_e1_step_300000.safetensors"},
                {"name": "beatrix+noob_l_4h:noise-310000", "file": "beatrix_dual_shunt_l_noob_e1_step_310000.safetensors"},
                {"name": "beatrix+noob_l_4h:noise-314000", "file": "beatrix_dual_shunt_l_noob_e1_step_314000.safetensors"},
            ],
        },
    }
}

# ─── Adapter Configs ─────────────────────────────────────────────

ENCODER_CONFIGS = {
    "bert-beatrix-2048": {
        "repo_name": "AbstractPhil/bert-beatrix-2048",
        "name": "bert-beatrix-2048",
        "type": "nomic_bert",
        "use_huggingface": True,  # defaults to simple loading from HuggingFace, if False, will use repo_name and subfolder
        "use_remote_code": True,
        "subfolder": "",
    },
    "nomic-bert-2048": {
        "repo_name": "nomic-ai/nomic-bert-2048",
        "name": "nomic-bert-2048",
        "type": "nomic_bert",
        "use_huggingface": True,  # defaults to simple loading from HuggingFace, if False, will use repo_name and subfolder
        "use_remote_code": True,
        "subfolder": "",
    },
    "mobilebert-base-uncased": {
        "repo_name": "google/mobilebert-uncased",
        "name": "mobilebert-base-uncased",
        "type": "mobilebert",
        "use_huggingface": True,  # defaults to simple loading from HuggingFace, if False, will use repo_name and subfolder
        "subfolder": "",
    },
    "bert-base-uncased": {
        "repo_name": "bert-base-uncased",
        "name": "bert-base-uncased",
        "type": "bert",
        "use_huggingface": True,  # defaults to simple loading from HuggingFace, if False, will use repo_name and subfolder
    },
    "bert-large-uncased": {
        "repo_name": "bert-large-uncased",
        "name": "bert-large-uncased",
        "type": "bert",
        "use_huggingface": True,  # defaults to simple loading from HuggingFace, if False, will use repo_name and subfolder
    },
    "bert-base-cased": {
        "repo_name": "bert-base-cased",
        "name": "bert-base-cased",
        "type": "bert",
        "use_huggingface": True,  # defaults to simple loading from HuggingFace, if False, will use repo_name and subfolder
    },
    "bert-large-cased": {
        "repo_name": "google-bert/bert-large-cased",
        "name": "bert-large-cased",
        "type": "bert",
        "use_huggingface": True,  # defaults to simple loading from HuggingFace, if False, will use repo_name and subfolder
    },
    "bert-base-multilingual-cased": {
        "repo_name": "google-bert/bert-base-multilingual-cased",
        "name": "bert-base-multilingual-cased",
        "type": "bert",
        "use_huggingface": True,  # defaults to simple loading from HuggingFace, if False, will use repo_name and subfolder
    },
    "t5xxl": {
        "repo_name": "google/t5-xxl-lm-adapt",
        "name": "t5-xxl",
        "type": "t5",
        "use_huggingface": True,  # defaults to simple loading from HuggingFace, if False, will use repo_name and subfolder
        "subfolder": "",
    },
    "distill-t5-base-4096": {
        "repo_name": "LifuWang/DistillT5",
        "name": "distill-t5-base-4096",
        "type": "t5_encoder_with_projection",
        "tokenizer": "t5_override", # only required if the tokenizer is missing from the model repo
        "use_huggingface": True,  # defaults to simple loading from HuggingFace, if False, will use repo_name and subfolder
        "subfolder": "",
    },
    "distill-t5-small-4096": {
        "repo_name": "LifuWang/DistillT5-Small",
        "name": "distill-t5-small-4096",
        "type": "t5_encoder_with_projection",
        "tokenizer": "t5_override", # only required if the tokenizer is missing from the model repo
        "use_huggingface": True,
        # defaults to simple loading from HuggingFace, if False, will use repo_name and subfolder
        "subfolder": "",
    },
    "distill-t5-large-4096": {
        "repo_name": "LifuWang/DistillT5-Large",
        "name": "distill-t5-large-4096",
        "type": "t5_encoder_with_projection",
        "tokenizer": "t5_override", # only required if the tokenizer is missing from the model repo
        "use_huggingface": True,
        # defaults to simple loading from HuggingFace, if False, will use repo_name and subfolder
        "subfolder": "",
    },

    "t5xxl-unchained": {
        "repo_name": "AbstractPhil/t5xxl-unchained",
        "name": "t5xxl-unchained-f16",
        "type": "t5",
        "use_huggingface": True,  # defaults to simple loading from HuggingFace, if False, will use repo_name and subfolder
        "subfolder": "",
        #"tokenizer": "t5-unchained",
        "file_name": "t5xxl-unchained-f16.safetensors",
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
    "flan-t5-small": {
        "repo_name": "google/flan-t5-small",
        "name": "flan-t5-small",
        "type": "t5",
        "use_huggingface": True,
        # defaults to simple loading from HuggingFace, if False, will use repo_name and subfolder
    },
    "flan-t5-base": {
        "repo_name": "google/flan-t5-base",
        "name": "flan-t5-base",
        "type": "t5",
        "use_huggingface": True, # defaults to simple loading from HuggingFace, if False, will use repo_name and subfolder
    },
    "flan-t5-large": {
        "repo_name": "google/flan-t5-large",
        "name": "flan-t5-large",
        "type": "t5",
        "use_huggingface": True, # defaults to simple loading from HuggingFace, if False, will use repo_name and subfolder
    },
    "flan-t5-xl": {
        "repo_name": "google/flan-t5-xl",
        "name": "flan-t5-xl",
        "type": "t5",
        "use_huggingface": True, # defaults to simple loading from HuggingFace, if False, will use repo_name and subfolder
    },
    "flan-t5-xxl": {
        "repo_name": "google/flan-t5-xxl",
        "name": "flan-t5-xxl",
        "type": "t5",
        "use_huggingface": True, # defaults to simple loading from HuggingFace, if False, will use repo_name and subfolder
    },
    "t5-small": {
        "repo_name": "google-t5/t5-small",
        "name": "t5-small",
        "type": "t5",
        "use_huggingface": True, # defaults to simple loading from HuggingFace, if False, will use repo_name and subfolder
    },
    "t5_small_human_attentive_try2_pass3": {
        "repo_name": "AbstractPhil/T5-Small-Human-Attentive-Try2-Pass3",
        "name": "t5_small_human_attentive_try2_pass3",
        "type": "t5",
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


# load these shunts in groups with types
GROUPED_SHUNTS = {
    # these are automatically loaded by the simple shunt organizer
    # for sdxl, they should have clip_l and clip_g
    # for flux, they should have clip_l or t5xxl or t5-unchained
    # for hidream, they can have clip_l, t5xxl, t5-unchained, or flan-t5-base, or llama - but llama isn't ready yet
    "beatrix": {
        "booru_sdxl_x2": [
            # clip_l heavy using 4 head shunts
            "beatrix+clip_l-4h:noise_25000",
            "beatrix+clip_g-booru_v1_8h:noise-30000"
        ],
        "noob_sdxl_x2": [
            # clip_l heavy using 4 head shunts
            "beatrix+noob_g_8h:noised-8000",
            "beatrix+noob_l_4h:noise-314000",
        ],
        "booru_": [
        ],
    },
    "flan-t5-base": {
        "booru_sdxl_12h": [
        ]

    }

}



SHUNT_DATAS: list[ShuntData] = []
""" 
    Populates the shunts list with available shunts from all the shunt dictionaries.
"""

ENCODER_DATAS: list[EncoderData] = []


class EncoderData:
    """
    Represents an encoder configuration with its associated properties.
    """
    def __init__(self,
                 name: str,
                 file: str,
                 repo: str,
                 config: dict,
                 type: Optional[str] = "t5",
                 tokenizer: Optional[str] = ""):
        self.name = name
        self.file = file
        self.repo = repo
        self.config = config
        self.type = type
        self.tokenizer = tokenizer

for encoder_dict in ENCODER_CONFIGS.values():
    if "repo_name" in encoder_dict:
        repo_name = encoder_dict["repo_name"]
        file_name = encoder_dict.get("file_name", "")
        # populate the encoders list with a reference to the encoder dictionary
        ENCODER_DATAS.append(EncoderData(
            name=encoder_dict["name"],
            file=file_name,
            repo=repo_name, # repo_name is the HuggingFace repo name
            tokenizer=encoder_dict.get("tokenizer", ""), # empty means use the default repo tokenizer, none found fails
            config=encoder_dict.get("config", {}),
            type=encoder_dict.get("type", "unknown")
        ))

class ShuntData:
    """
    Represents a shunt configuration with its associated properties.
    """

    def __init__(self, name: str, file: str, repo: str, config: dict, expected: list[str], modulation_encoders: list[dict], condition_encoders: list[dict], shunt_type_name: str, config_file_name: str):
        self.name = name
        self.file = file
        self.repo = repo
        self.config = config
        self.expected = expected
        self.modulation_encoders = modulation_encoders
        self.condition_encoders = condition_encoders
        self.shunt_type_name = shunt_type_name
        self.config_file_name = config_file_name


for shunt_dict in HARMONIC_SHUNT_REPOS.values():
    if "shunts_available" in shunt_dict:
        shunts = shunt_dict["shunts_available"]["shunt_list"]
        for shunt in shunts:
            name, file_name = shunt.get("name"), shunt.get("file")
            # populate the shunts list with a reference to the shunt dictionary
            SHUNT_DATAS.append(ShuntData(
                name=name,
                file=file_name,
                repo=shunt_dict["repo"],
                config=shunt_dict["config"],
                expected=shunt_dict["models"],
                modulation_encoders=shunt_dict["config"]["modulation_encoders"],
                condition_encoders=shunt_dict["config"]["condition_encoders"],
                shunt_type_name=shunt_dict["shunts_available"]["shunt_type_name"],
                config_file_name=shunt_dict["shunts_available"]["config_file_name"]
            ))



class ShuntUtil:

    @staticmethod
    def get_encoder_by_model_name(model_name: str) -> Optional[EncoderData]:
        """
        Returns the encoder configuration dictionary by its model name.
        Args:
            model_name (str): The name of the model to retrieve.

        Returns:
            Optional[EncoderData]: The encoder configuration dictionary if found, otherwise None.
        """
        logger.info(f"Searching for encoder with model name: {model_name}")
        logger.info(f"Available encoders: {[encoder.name for encoder in ENCODER_DATAS]}")
        for encoder in ENCODER_DATAS:
            logger.info(f"Checking encoder: {encoder.name} against model name: {model_name}")
            if encoder.name == model_name:
                return encoder
        logger.warning(f"Encoder '{model_name}' not found.")
        return None

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

                    logger.warning(f"Model '{model}' not found in default repositories.")
            return prepared
        else:
            logger.warning(f"Shunt '{shunt_name}' not found.")

        return None

    @staticmethod
    def get_shunt_by_name(name: str) -> Optional[ShuntData]:
        """
        Returns the shunt configuration dictionary by its name.

        Args:
            name (str): The name of the shunt to retrieve.

        Returns:
            Optional[dict]: The shunt configuration dictionary if found, otherwise None.
        """
        for shunt in SHUNT_DATAS:
            if shunt.name == name:
                return shunt
        logger.warning(f"Shunt '{name}' not found.")
        return None

    @staticmethod
    def get_shunt_names() -> list[str]:
        """
        Returns a list of all available shunt names.

        Returns:
            list[str]: List of shunt names.
        """
        return [shunt.name for shunt in SHUNT_DATAS]

