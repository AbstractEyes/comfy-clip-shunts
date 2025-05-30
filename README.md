# Comfy CLIP Shunts

This repository provides a collection of **ComfyUI** custom nodes implementing experimental "dual stream" shunt adapters. The adapters bridge `T5` language model embeddings to CLIP conditioning vectors, allowing more advanced prompt guidance during image generation.

This is heavily set up and programmed by AI passively, so keep in mind that this may be prone to error and halucination.

## Features

- Loading T5 models and shunt adapter weights on demand
- Utilities for stacking, scheduling and previewing adapters
- Helper functions to visualize the effect of a shunt
- Model manager with simple caching and unloading

## Installation

1. Clone or copy this folder into your `comfyui/custom_nodes` directory:

```bash
comfyui/custom_nodes/comfy-clip-shunts/
```

2. Download the shunt weights from [HuggingFace](https://huggingface.co/AbstractPhil):
   - [t5-flan-base-vit-bigG-14-dual-stream-adapter](https://huggingface.co/AbstractPhil/t5-flan-base-vit-bigG-14-dual-stream-adapter)
   - [t5-flan-base-vit-bigL-14-dual-stream-adapter](https://huggingface.co/AbstractPhil/t5-flan-base-vit-bigL-14-dual-stream-adapter)

3. Place the downloaded `.safetensors` files in `comfyui/models/shunt_guides`.

The T5 model itself is cached via HuggingFace; you can override the default cache by setting the `HF_HOME` environment variable.

## Usage

After installation, the nodes become available in the ComfyUI interface under the `adapter` categories. Load a T5 model with **T5LoaderTest**, then load one or more shunts with **LoadAdapterShunt** and plug them into the conditioning nodes provided by ComfyUI.

## License

This project is released under the terms of the MIT License. See `LICENSE` for details.

