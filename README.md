# compitabi

Answers the question: **"In which vLLM version is this HuggingFace model supported?"**

Given a model ID like `meta-llama/Llama-3.2-11B-Vision-Instruct`, it fetches the model's architecture from the HuggingFace API and looks it up against a local cache of vLLM release history.

## How it works

1. HuggingFace models declare their architecture in `config.architectures` (e.g. `MllamaForConditionalGeneration`)
2. vLLM registers supported architectures as dictionary keys in its model registry files
3. `build_cache.py` walks every vLLM GitHub release, extracts those architecture names, and records the first version each one appeared in
4. `check.py` fetches a model's architecture from HuggingFace and looks it up in the cache

## Setup

```bash
uv sync
```

## Usage

### 1. Build the cache

Fetches all vLLM releases from GitHub (~100 releases as of 2026). A `GITHUB_TOKEN` is strongly recommended to avoid the 60 req/hour unauthenticated rate limit.

```bash
GITHUB_TOKEN=your_token uv run python build_cache.py
```

The cache is saved to `vllm_cache.json` and is incremental — re-running only fetches releases not already cached.

### 2. Check a model

```bash
uv run python check.py meta-llama/Llama-3.2-11B-Vision-Instruct
```

Example output:

```
Model:         meta-llama/Llama-3.2-11B-Vision-Instruct
Architectures: MllamaForConditionalGeneration
Cache built:   2026-04-17T10:00:00+00:00

  MllamaForConditionalGeneration  ->  first supported in vLLM v0.10.0
```

If an architecture is not found, the model is either unsupported or support was added after the cache was last built — re-run `build_cache.py` to refresh.
