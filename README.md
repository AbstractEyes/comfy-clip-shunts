# Handholding Guidance Adapter ComfyUI Custom Nodes for Experimental Dual Stream Shunt

A small set of unique adapters meant to bridge the dual_stream_shunt trained for guiding prompt embeddings and diffusion.

Baseline is implemented.

Clone into the `comfyui/custom_nodes/comfy-abs-shunt-adapters` directory.

Directory should be structured as follows:
* `comfyui/custom_nodes/comfy-abs-shunt-adapters/`
  * `__init__.py`
  * `configs.py`
  * `dual_stream_shunt_adapter.py`
  * `gitignore`
  * `README.md`
  * `nodes.py`

Download the weights here;
For CLIP_G:
https://huggingface.co/AbstractPhil/t5-flan-base-vit-bigG-14-dual-stream-adapter

For CLIP_L:
https://huggingface.co/AbstractPhil/t5-flan-base-vit-bigL-14-dual-stream-adapter

Place the downloaded weights in the `comfyui/models/shunt_guides` directory.

It caches the t5-flan-base model wherever the huggingface cache is set, so you can set it to your own directory by setting the `HF_HOME` environment variable.

