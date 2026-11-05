# Rewind Lite — T03 environment and single-encoder smoke test

Date: 2026-11-06 (Asia/Bangkok). This task uses the local `video-test.mp4` supplied by the user. That video and generated indexes remain outside Git.

## Environment choice

The machine has an NVIDIA GeForce RTX 3060 Laptop GPU (6 GB) and driver 546.92. The pre-existing Python 3.14 environment has `torch 2.14.0+cpu` and reports `torch.cuda.is_available() == False`; it is not a GPU baseline. T03 uses project-local Python 3.11.15 in `artifacts/python/` and `.venv/`.

PyTorch's [official previous versions page](https://docs.pytorch.org/get-started/previous-versions/) lists a Windows/Linux pip wheel for PyTorch 2.5.1 with CUDA 12.1. NVIDIA's [CUDA compatibility table](https://docs.nvidia.com/cuda/archive/12.5.0/cuda-toolkit-release-notes/index.html) gives Windows driver 528.33 as the minimum for CUDA 12.x minor compatibility; 546.92 exceeds it. Actual GPU availability was verified in this venv below.

PowerShell setup from the repository root:

```powershell
$env:UV_CACHE_DIR = "$PWD\artifacts\uv-cache"
uv python install 3.11.15 --install-dir "$PWD\artifacts\python"
uv venv --python "$PWD\artifacts\python\cpython-3.11.15-windows-x86_64-none\python.exe" .venv
uv pip install --python .venv\Scripts\python.exe -r requirements-gpu.txt
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

`HF_HOME` wins if explicitly set. Otherwise `AIC_HF_HOME` may select a cache; the default is `artifacts/hf-cache/` inside this checkout. The CLI pins the single SigLIP2 encoder to a model commit and records the exact model ID/revision in its manifest.

## CLI contract

```powershell
.venv\Scripts\python.exe -m scripts.lite_search video-test.mp4 "người đang đi bộ" --index-dir artifacts/indexes/video-test --every 1 --max-frames 64 --top-k 5 --device auto
# Run again with another Vietnamese query: the saved index is loaded without embedding video frames again.
.venv\Scripts\python.exe -m scripts.lite_search video-test.mp4 "xe trên đường" --index-dir artifacts/indexes/video-test --every 1 --max-frames 64 --top-k 5 --device auto
```

The CLI uses only one image/text encoder, no OCR/ASR/caption/VLM rerank, and dense cosine search over the retained keyframes. It prints video ID, frame ID, timestamp, rank, score, device, frame counts, index time, and query time. `manifest.json` records video SHA-256/path/duration, sampling, dedup, model revision, index configuration, package versions and device. `last-run.json` records the latest result. `--rebuild` is required if the video, model or sampling configuration changes. A CPU run is labeled CPU and cannot establish GPU performance.

## Verification record

The local venv reports `torch 2.5.1+cu121`, CUDA runtime 12.1, `torch.cuda.is_available() == True`, and `NVIDIA GeForce RTX 3060 Laptop GPU`. The real encoder was `google/siglip2-base-patch16-256` at revision `6328738b83b563ff8396fdb87f8ce130a7b60d74`. The model weights are cached under `artifacts/hf-cache/`.

Source manifest: SHA-256 `8410556fea9c755db18ebd29316405065e4330af52aa21530604bb72abdc448c`; 77,574,329 bytes; 50 FPS; 1,193 frames; 23.86 seconds. The exact input path, package versions, model revision, sampling/dedup settings, index settings, and device are in the ignored `artifacts/indexes/video-test/manifest.json`.

Executed from the repository root (with `$env:HF_HUB_OFFLINE = '1'` after the model download):

```powershell
.venv\Scripts\python.exe -m scripts.lite_search video-test.mp4 "người đang đi bộ" --index-dir artifacts/indexes/video-test --every 1 --max-frames 64 --top-k 5 --device cuda
.venv\Scripts\python.exe -m scripts.lite_search video-test.mp4 "xe trên đường" --index-dir artifacts/indexes/video-test --every 1 --max-frames 64 --top-k 5 --device cuda
```

First run: `index_reused=false`, 2 sampled/indexed keyframes, index 9.332 seconds, query 0.065 seconds, Top-2 at 5.0 and 0.0 seconds. Second run: `index_reused=true`, query 0.412 seconds, Top-2 at 5.0 and 0.0 seconds. Both returned `video_id=video-test` and `device=cuda`; the second run loaded the saved Faiss index without video embedding. The existing histogram duplicate filter retained only two frames before semantic deduplication. This short sample does not establish retrieval accuracy or performance across diverse videos. There are no verified labels for this video.

Verification: `.venv\Scripts\python.exe -m pytest tests/test_lite_cli_t03.py tests/test_model_cache_t03.py tests/test_lite_cli_integration_t03.py -q` passed 11 tests. The full venv command `.venv\Scripts\python.exe -m pytest -o addopts='' -q` yielded `307 passed, 1 skipped in 7.96s`; the skip is a legacy test conditioned on torch being installed. `git diff --check` and Python compilation of the new CLI/cache module completed without errors.

The first model download encountered a Windows `MemoryError` while Hugging Face fell back to HTTP. A subsequent single-worker download completed; the successful smoke runs used the local cached snapshot. No model weights, source video, or generated index are tracked in Git.
