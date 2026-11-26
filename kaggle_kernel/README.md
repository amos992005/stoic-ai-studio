# `kaggle_kernel/` — remote GPU compute

This folder is uploaded verbatim to Kaggle by `scripts/run_kaggle_job.py`.

| File | Purpose |
| --- | --- |
| `character_forge.py` | The Micro-Pilot 1 kernel: golden reference portrait + 3 IP-Adapter Face-locked scene variations + contact sheet + `metrics.json`. |
| `kernel-metadata.json` | **Generated at run time** by CI (never committed) — `id`, `code_file`, GPU/internet flags. |

## How the kernel is configured

* `kernel_type: script`, `code_file: character_forge.py`, `language: python`
* `enable_gpu: "true"` → Dual Tesla T4 (16 GB each), `torch.float16` on `cuda:0`
* `enable_internet: "true"` → pulls `SG161222/RealVisXL_V5.0`,
  `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` and `h94/IP-Adapter` weights
* `is_private: "true"`

Artifacts are written to `/kaggle/working` (not a subfolder) so that
`kaggle kernels output ... -p ./outputs` lands them at `outputs/<file>`.

## Running it manually

```bash
export KAGGLE_USERNAME=...   # Kaggle → Settings → API
export KAGGLE_KEY=...        # 32-hex legacy key or `KGAT_...` API token
python scripts/run_kaggle_job.py
```

Useful environment overrides for the kernel itself (set through the metadata or
`OUTPUT_DIR` when running locally):

| Variable | Default | Meaning |
| --- | --- | --- |
| `SDXL_MODEL_ID` | `SG161222/RealVisXL_V5.0` | checkpoint |
| `BASE_SEED` | `42` | golden reference seed |
| `ALLOW_IP_ADAPTER_FALLBACK` | `0` | `1` = still produce panels if the face adapter fails (identity will drift) |
| `OUTPUT_DIR` | `/kaggle/working` or `./outputs` | artifact destination |
