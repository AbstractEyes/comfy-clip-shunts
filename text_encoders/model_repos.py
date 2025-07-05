# we use these to extract configurations,  settings, shapes, and other metadata from the original models.
MODEL_REPOS = {
    # ------------------------------
    # 📎 CLIP Variants (Text/Visual)
    # ------------------------------
    "clip_l": "openai/clip-vit-large-patch14",  # Standard CLIP-L
    "clip_g": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",  # SDXL-level
    "clip_h": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",  # SD20
    "long_clip_l": "BeichenZhang/LongCLIP-L",  # Extended context CLIP
    "long_clip_g": "BeichenZhang/LongCLIP-G",  # Extended context CLIP (BigG variant)

    # ------------------------------
    # ✍️ T5 Series (Text Encoders)
    # ------------------------------
    "t5_xxl": "google/t5-v1_1-xxl",  # Primary T5 for SD3/Flux/etc
    "t5_xxl_old": "google/t5-xxl",  # Legacy version (used by Cosmos)
    "t5_xl": "google/t5-v1_1-xl",  # AuraFlow, alternate
    "t5_base": "google/t5-base",  # Smaller variant (StableAudio, ACE)

    # ------------------------------
    # 🌐 Multilingual T5 Extensions
    # ------------------------------
    "mt5_xl": "google/mt5-xl",  # Multilingual T5 for Hunyuan
    "umt5_xxl": "google/umt5-xxl",  # Universal MT5 for WAN

    # ------------------------------
    # 🈷️ BERT / RoBERTa Models
    # ------------------------------
    "bert_chinese": "hfl/chinese-roberta-wwm-ext-large",  # HunyuanDiT (Chinese)

    # ------------------------------
    # 🧠 LLM Text Encoders
    # ------------------------------
    "llama3_8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",  # HiDream
    "gemma_2b": "google/gemma-2-2b",  # Lumina2
    "qwen25_3b": "Qwen/Qwen2.5-3B",  # Omnigen2

    # ------------------------------
    # 🧠+🖼️ Multimodal (LLaVA)
    # ------------------------------
    "llava": "llava-hf/llava-llama-3-8b-v1_1",  # HunyuanVideo

    # ------------------------------
    # 🧬 Custom / Extended Vocab
    # ------------------------------
    "t5_xxl_unchained": "AbstractPhil/t5xxl-unchained",  # Extended tokenizer variant
}
