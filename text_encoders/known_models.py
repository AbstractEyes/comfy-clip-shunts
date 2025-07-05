# ------------------------------------------------------------------ #
#  Canonical repository map (every alias -> Hugging-Face repo slug)  #
# ------------------------------------------------------------------ #
MODEL_REPOS: dict[str, str] = {
    # CLIP / Long-CLIP
    "clip_l":          "openai/clip-vit-large-patch14",
    "clip_g":          "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    "clip_h":          "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "long_clip_l":     "BeichenZhang/LongCLIP-L",
    "long_clip_g":     "BeichenZhang/LongCLIP-G",

    # T5 family
    "t5_xxl":          "google/t5-v1_1-xxl",
    "t5_xxl_old":      "google/t5-xxl",          # v1  (Cosmos legacy)
    "t5_xl":           "google/t5-v1_1-xl",
    "t5_base":         "google/t5-base",

    # Multilingual / Universal T5
    "mt5_xl":          "google/mt5-xl",          # kept for completeness
    "umt5_xxl":        "google/umt5-xxl",

    # BERT / RoBERTa
    "bert_chinese":    "hfl/chinese-roberta-wwm-ext-large",

    # LLM text encoders
    "llama3_8b":       "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "gemma_2b":        "google/gemma-2-2b",
    "qwen25_3b":       "Qwen/Qwen2.5-3B",

    # Multimodal LLM (text + vision tower inside)
    "llava":           "llava-hf/llava-llama-3-8b-v1_1",

    # Custom extended-vocab T5
    "t5_xxl_unchained": "AbstractPhil/t5xxl-unchained",
}

# ------------------------------------------------------------------ #
#  Per-model encoder specification (text + multimodal where needed)  #
# ------------------------------------------------------------------ #
KNOWN_MODELS: dict[str, dict] = {
    # ---- Stable-Diffusion lineage ---------------------------------
    "SD15":                    {"encoders": ["clip_l"]},
    "SD20":                    {"encoders": ["clip_h"]},
    "SD21UnclipL":             {"encoders": ["clip_l"]},
    "SD21UnclipH":             {"encoders": ["clip_h"]},
    "SDXL":                    {"encoders": ["clip_l", "clip_g"]},
    "SDXLRefiner":             {"encoders": ["clip_g"]},
    "SSD1B":                   {"encoders": ["clip_l", "clip_g"]},
    "SD3":                     {"encoders": ["clip_l", "clip_g", "t5_xxl"]},

    # ---- Flux family ----------------------------------------------
    "Flux":                    {"encoders": ["clip_l", "t5_xxl"]},
    "FluxSchnell": {
        "encoders": ["clip_l", "t5_xxl"],
        "encoder_config": {"t5_xxl": {"max_tokens": 256}}
    },
    "FluxInpaint":             {"encoders": ["clip_l", "t5_xxl"]},

    # ---- PixArt ----------------------------------------------------
    "PixArtAlpha": {
        "encoders": ["t5_xxl"],
        "encoder_config": {"t5_xxl": {"max_tokens": 120}}
    },
    "PixArtSigma": {
        "encoders": ["t5_xxl"],
        "encoder_config": {"t5_xxl": {"max_tokens": 300}}
    },

    # ---- Cosmos (legacy T5-v1) ------------------------------------
    "CosmosT2V":               {"encoders": ["t5_xxl_old"]},
    "CosmosI2V":               {"encoders": ["t5_xxl_old"]},
    "CosmosT2IPredict2":       {"encoders": ["t5_xxl_old"]},
    "CosmosI2VPredict2":       {"encoders": ["t5_xxl_old"]},

    # ---- Other proprietary stacks ---------------------------------
    "AuraFlow":                {"encoders": ["t5_xl"]},
    "StableAudio":             {"encoders": ["t5_base"]},
    "ACE":                     {"encoders": ["t5_base"]},
    "ACEStep":                 {"encoders": ["t5_base"]},
    "Chroma":                  {"encoders": ["t5_xxl"]},
    "Mochi":                   {"encoders": ["t5_xxl"]},
    "GenmoMochi":              {"encoders": ["t5_xxl"]},
    "LTXV":                    {"encoders": ["t5_xxl"]},

    # ---- HiDream multimodal suite ---------------------------------
    "HiDream": {
        "encoders": ["llama3_8b", "long_clip_l", "long_clip_g", "t5_xxl"]
    },

    # ---- Research / speciality models -----------------------------
    "Lumina2":                 {"encoders": ["gemma_2b"]},
    "Omnigen2":                {"encoders": ["qwen25_3b"]},

    # ---- Stable-Cascade & Zero123 ---------------------------------
    "Stable_Cascade_C":        {"encoders": ["clip_g"]},
    "Stable_Cascade_B":        {"encoders": ["clip_g"]},
    "Stable_Zero123":          {"encoders": ["clip_l"]},

    # ---- KOALA checkpoints ----------------------------------------
    "KOALA_700M":              {"encoders": ["clip_l"]},
    "KOALA_1B":                {"encoders": ["clip_l"]},

    # ---- InstructPix2Pix forks ------------------------------------
    "SD15_instructpix2pix":    {"encoders": ["clip_l"]},
    "SDXL_instructpix2pix":    {"encoders": ["clip_l", "clip_g"]},

    # ---- Misc. CLIP-based stacks ----------------------------------
    "LotusD":                  {"encoders": ["clip_l"]},
    "Segmind_Vega":            {"encoders": ["clip_l", "clip_g"]},
    "SD_X4Upscaler":           {"encoders": ["clip_l"]},
    "SV3D_u":                  {"encoders": ["clip_l"]},
    "SV3D_p":                  {"encoders": ["clip_l"]},
}
