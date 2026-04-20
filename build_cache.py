#!/usr/bin/env python3
"""Build a cache mapping HuggingFace architecture names -> first vLLM version that supports them.

Run once to populate vllm_cache.json, re-run to pick up new releases incrementally.

Usage:
    GITHUB_TOKEN=your_token python build_cache.py
    python build_cache.py   # works without token but rate-limited to 60 req/hour
"""

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

VLLM_REPO = "vllm-project/vllm"
CACHE_FILE = "vllm_cache.json"

# Try registry.py first (introduced ~v0.8+), fall back to __init__.py
REGISTRY_PATHS = [
    "vllm/model_executor/models/registry.py",
    "vllm/model_executor/models/__init__.py",
]

# Matches HuggingFace architecture class names (dict keys in vLLM registry files)
ARCH_RE = re.compile(
    r'"([A-Z][a-zA-Z0-9]+'
    r'(?:ForCausalLM|ForConditionalGeneration|ForSeq2SeqLM'
    r'|ForTokenClassification|ForSequenceClassification'
    r'|ForImageTextToText|ForSpeechSeq2Seq|ForVision2Seq|ForMaskedLM|Model))"'
)


def gh_get(path, token=None, **params):
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.get(f"https://api.github.com{path}", headers=headers, params=params)
    r.raise_for_status()
    return r.json()


def get_all_releases(token):
    releases, page = [], 1
    while True:
        batch = gh_get(f"/repos/{VLLM_REPO}/releases", token, per_page=100, page=page)
        if not batch:
            break
        releases.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return releases


def fetch_raw(tag, path, token=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = f"https://raw.githubusercontent.com/{VLLM_REPO}/{tag}/{path}"
    r = requests.get(url, headers=headers)
    return r.text if r.status_code == 200 else None


def parse_architectures(content):
    return set(ARCH_RE.findall(content))


def version_tuple(tag):
    parts = re.split(r"[.\-]", tag.lstrip("v"))
    return tuple(int(p) if p.isdigit() else 0 for p in parts)


def version_sort_key(release):
    return version_tuple(release["tag_name"])


def build_cache(token=None):
    print("Fetching vLLM releases from GitHub...")
    all_releases = sorted(get_all_releases(token), key=version_sort_key)
    releases = all_releases
    print(f"Found {len(all_releases)} releases\n")

    # Load existing cache for incremental updates
    arch_first_seen, already_checked = {}, set()
    if Path(CACHE_FILE).exists():
        with open(CACHE_FILE) as f:
            old = json.load(f)
        arch_first_seen = old.get("architectures", {})
        already_checked = set(old.get("vllm_versions_checked", []))
        print(f"Existing cache: {len(arch_first_seen)} architectures, {len(already_checked)} versions checked\n")

    versions_checked = list(already_checked)

    for release in releases:
        tag = release["tag_name"]
        if tag in already_checked:
            continue

        print(f"  {tag}: ", end="", flush=True)

        content = None
        for path in REGISTRY_PATHS:
            content = fetch_raw(tag, path, token)
            if content:
                break

        if not content:
            print("no registry file found, skipping")
            versions_checked.append(tag)
            time.sleep(0.05)
            continue

        archs = parse_architectures(content)
        new_archs = [a for a in archs if a not in arch_first_seen]
        for a in new_archs:
            arch_first_seen[a] = tag
        versions_checked.append(tag)

        print(f"{len(archs)} architectures ({len(new_archs)} new)")
        time.sleep(0.05)

    cache = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "vllm_versions_checked": versions_checked,
        "architectures": arch_first_seen,
    }
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)

    print(f"\nSaved {len(arch_first_seen)} architectures across {len(versions_checked)} versions -> {CACHE_FILE}")


if __name__ == "__main__":
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Tip: set GITHUB_TOKEN to avoid GitHub's 60 req/hour rate limit\n")
    build_cache(token)
