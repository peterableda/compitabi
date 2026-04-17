#!/usr/bin/env python3
"""Check which vLLM version first supports a given HuggingFace model.

Usage:
    python check.py meta-llama/Llama-3.2-11B-Vision-Instruct
"""

import json
import sys
from pathlib import Path

import requests

HF_API = "https://huggingface.co/api/models"
CACHE_FILE = "vllm_cache.json"


def get_architectures(model_id):
    r = requests.get(f"{HF_API}/{model_id}")
    if r.status_code == 404:
        sys.exit(f"Model not found on HuggingFace: {model_id}")
    r.raise_for_status()
    return r.json().get("config", {}).get("architectures", [])


def load_cache():
    if not Path(CACHE_FILE).exists():
        sys.exit(f"Cache not found. Run 'python build_cache.py' first.")
    with open(CACHE_FILE) as f:
        return json.load(f)


def check(model_id):
    archs = get_architectures(model_id)
    if not archs:
        sys.exit(f"No architectures in HuggingFace config for: {model_id}")

    cache = load_cache()
    arch_map = cache["architectures"]

    print(f"Model:         {model_id}")
    print(f"Architectures: {', '.join(archs)}")
    print(f"Cache built:   {cache.get('built_at', 'unknown')}\n")

    unsupported = []
    for arch in archs:
        if arch in arch_map:
            print(f"  {arch}  ->  first supported in vLLM {arch_map[arch]}")
        else:
            print(f"  {arch}  ->  NOT FOUND (not supported or cache is stale)")
            unsupported.append(arch)

    if unsupported:
        print(f"\nRe-run 'python build_cache.py' to refresh if a recent vLLM release added support.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(
            f"Usage: {sys.argv[0]} <hf_model_id>\n"
            f"Example: {sys.argv[0]} meta-llama/Llama-3.2-11B-Vision-Instruct"
        )
    check(sys.argv[1])
