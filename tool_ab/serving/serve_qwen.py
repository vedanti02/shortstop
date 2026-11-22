"""Serve Qwen3-Coder with vLLM on Modal behind an OpenAI-compatible, key-protected endpoint.

One-time setup (the key is any long random string you choose; keep it out of git):

    modal secret create shortstop-vllm-key VLLM_API_KEY=<random-key>

Then:

    modal deploy tool_ab/serving/serve_qwen.py     # prints the endpoint URL
    modal app stop shortstop-qwen                  # stop it when the runs are done

Put the URL (with /v1 appended) and the key in tool_ab/.env (see .env.example).

Spending guards: at most one GPU container, and it shuts down after 10 idle minutes. The
first start downloads about 31 GB of weights into a Modal volume; later starts reuse them.
"""

from __future__ import annotations

import os
import subprocess

import modal

MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
SERVED_NAME = "qwen3-coder"  # base.yaml uses hosted_vllm/qwen3-coder
GPU = os.environ.get("SHORTSTOP_GPU", "L40S")  # 48 GB fits the FP8 weights plus a KV cache
MAX_MODEL_LEN = 65536
PORT = 8000

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .pip_install("vllm==0.11.0", "huggingface_hub[hf_transfer]==0.35.3")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
hf_cache = modal.Volume.from_name("shortstop-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("shortstop-vllm-cache", create_if_missing=True)

app = modal.App("shortstop-qwen")


@app.function(
    image=image,
    gpu=GPU,
    max_containers=1,
    scaledown_window=10 * 60,
    timeout=24 * 60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    secrets=[modal.Secret.from_name("shortstop-vllm-key")],
)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=PORT, startup_timeout=30 * 60)
def serve() -> None:
    # vLLM reads VLLM_API_KEY from the environment and rejects requests without it.
    if not os.environ.get("VLLM_API_KEY"):
        raise RuntimeError("VLLM_API_KEY is missing from the shortstop-vllm-key secret")
    cmd = [
        "vllm", "serve", MODEL,
        "--served-model-name", SERVED_NAME,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", "0.92",
        "--enable-prefix-caching",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder",
        "--disable-fastapi-docs",  # the key guards /v1/*; do not publish /docs either
    ]
    subprocess.Popen(cmd)
