# Handholding Guidance Adapter ComfyUI Custom Nodes for Experimental Dual Stream Shunt

A small set of unique adapters meant to bridge the dual_stream_shunt trained for guiding prompt embeddings and diffusion.

* Supporting a heavy refactor of the clip encoder system from stem to stern.
* Still uses some baseline comfyui behavior but this will slowly be replaced over time.

Clone into the `comfyui/custom_nodes/comfy-abs-shunt-adapters` directory.

The SHUNT weights download automatically through the scripts and are managed through the model/model_manager.py

Shunts Supported:
* todo

Shunts Supported encoders:
* AbstractPhil/bert-beatrix-2048 - requires remote_code
* nomicai/nomic-bert-2048 - requires remote_code
* bert-base-uncased
* bert-base-cased
* google/flan-t5-base
* google/flan-t5-small

Core ComfyUI Clip Loaders support:
* hidream
* * t5-unchained-fp8
* * t5-unchained-fp16
* flux
* * t5-unchained-fp8
* * t5-unchained-fp16
* * LifuWang/DistillT5

# Nodes
* Lots, and lots, and a sampler, and lots more.


Place the downloaded weights in the `comfyui/models/shunt_guides` directory.

It caches the t5-flan-base model wherever the huggingface cache is set, so you can set it to your own directory by setting the `HF_HOME` environment variable.

