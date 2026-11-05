"""T03 model cache must be portable and honor user configuration."""
import importlib
from pathlib import Path

from ingestion import model_cache


def test_default_hf_cache_stays_within_project(monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("AIC_HF_HOME", raising=False)
    importlib.reload(model_cache)
    actual = Path(model_cache.setup_hf_cache()).resolve()
    project = Path(model_cache.__file__).resolve().parents[1]
    assert actual.is_relative_to(project)


def test_explicit_hf_home_is_preserved(monkeypatch, tmp_path):
    user_cache = str(tmp_path / "chosen")
    monkeypatch.setenv("HF_HOME", user_cache)
    monkeypatch.setenv("AIC_HF_HOME", str(tmp_path / "ignored"))
    assert model_cache.setup_hf_cache() == user_cache
    assert Path(model_cache.setup_hf_cache()) == tmp_path / "chosen"
