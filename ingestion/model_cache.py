"""Configure a project-local Hugging Face cache before model imports.

HF_HOME has highest priority. AIC_HF_HOME overrides the project default without
changing the user's global Hugging Face configuration.
"""
from __future__ import annotations

import os
from pathlib import Path


DEFAULT_HF_CACHE = Path(__file__).resolve().parents[1] / "artifacts" / "hf-cache"


def setup_hf_cache(path: str | None = None) -> str | None:
    """Set HF_HOME only when unset; return the effective location if available."""
    if os.environ.get("HF_HOME"):
        return os.environ["HF_HOME"]
    target = Path(path or os.environ.get("AIC_HF_HOME") or DEFAULT_HF_CACHE)
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    os.environ["HF_HOME"] = str(target)
    return str(target)


# Hugging Face libraries read HF_HOME during import.
setup_hf_cache()
