# stoic-ai-studio

An automated, **$0-cost**, open-source AI virtual-creator pipeline.

> **Julian** — 42-year-old charismatic Stoic/mindset mentor. Sharp jawline, short
> salt-and-pepper beard, dark turtleneck/suit, moody chiaroscuro cinematic lighting.

We have no local GPU, so the architecture splits responsibilities:

```
GitHub Actions (CPU, free)            Kaggle Kernel (Dual Tesla T4, free)
─────────────────────────             ────────────────────────────────────
scripts/run_kaggle_job.py  ──push──▶  kaggle_kernel/character_forge.py
  • auth from secrets                   • RealVisXL V5.0 (SDXL, fp16) on cuda:0
  • generates metadata                  • Step A: 01_master_face.png (seed 42)
  • polls status (25 s / 25 min)        • Step B: IP-Adapter Face @ 0.65 →
  • downloads artifacts  ◀──output──      3 face-locked 768×1344 variations
  • commits contact_sheet.png,          • Step C: labelled contact_sheet.png
    metrics.json, build.log               + metrics.json
        │
        └─▶ commit back to the branch with `[skip ci]`
```

## Phase 1 · Micro-Pilot 1 — "The Character Forge"

**Goal:** prove we can generate one golden reference portrait and keep the *same
face* across radically different scenes — the foundation for a consistent virtual
creator.

| # | Artifact | Size | What it proves |
| --- | --- | --- | --- |
| 01 | `01_master_face.png` | 1024×1024 | Golden reference identity (seed `42`, 30 steps, CFG 5.5) |
| 02 | `02_library_scene.png` | 768×1344 | Candles + ancient library, same face |
| 03 | `03_rainy_street.png` | 768×1344 | Neon rain, trench coat, same face |
| 04 | `04_podcast_studio.png` | 768×1344 | Broadcast mic studio, same face |
| ★ | `contact_sheet.png` | grid | Labeled side-by-side inspection of all four |
| — | `metrics.json` | json | Seeds, prompts, timings, env fingerprint |

## Repository layout

```
.github/workflows/character_forge.yml   # orchestrator: dispatch + push on main
kaggle_kernel/
  character_forge.py                    # the GPU kernel (SDXL + IP-Adapter Face)
  kernel-metadata.json                  # generated at run time (not committed)
scripts/run_kaggle_job.py               # local/CI Kaggle CLI driver
tools/preview_contact_sheet.py          # CPU-only layout preview (no GPU needed)
outputs/                                # artifacts written back by CI
```

## Setup (one time)

1. **Kaggle API token** — <https://www.kaggle.com/settings> → *API* → *Create New
   Token* (a `kaggle.json`, or a `KGAT_...` API token).
2. **GitHub secrets** — *Settings → Secrets and variables → Actions*:

   | Secret | Value |
   | --- | --- |
   | `KAGGLE_USERNAME` | your Kaggle username (the handle in your profile URL) |
   | `KAGGLE_KEY` | the 32-hex key **or** the `KGAT_...` token |

   Quotes, spaces and newlines pasted with the secret are stripped automatically,
   and `~/.kaggle/kaggle.json` is written with mode `600` so both auth styles work.
3. **Phone verification on Kaggle** is required before GPU kernels are allowed —
   without it the kernel errors out on push.
4. *(Optional)* if you publish this repo as a private Kaggle dataset named
   `<username>/stoic-ai-studio`, set `KAGGLE_DATASET_SLUG` so the kernel can
   resolve its script reliably.

## Running

* **Automatic:** push to `main` touching `kaggle_kernel/**`, `scripts/**` or the
  workflow file.
* **Manual:** *Actions → Character Forge · Micro-Pilot 1 → Run workflow*
  (optional inputs: `base_seed`, `allow_fallback`).
* **Locally:** `pip install -r requirements.txt && python scripts/run_kaggle_job.py`
  with `KAGGLE_USERNAME` / `KAGGLE_KEY` exported.

Once the kernel finishes, the workflow force-adds only lightweight artifacts
(`contact_sheet.png`, `metrics.json`, `build.log`; anything ≥ 5 MB is skipped) and
commits them with `[skip ci]` so the pipeline cannot loop.

## Cost & guardrails

* $0: GitHub Actions free minutes + Kaggle's free GPU quota (30 h/week).
* `torch.float16` on `cuda:0`, VAE slicing/tiling and attention slicing keep a
  16 GB T4 comfortable; scenes render sequentially with cache flushes.
* If the IP-Adapter cannot load, the kernel **fails loudly** (`ALLOW_IP_ADAPTER_FALLBACK=1`
  opts into a degraded, non-face-locked run that is flagged in `metrics.json`).
* Polling is capped at 25 minutes with a 45-minute job timeout as a backstop.

## Next pilots

Phase 2 will build on `01_master_face.png` as the permanent identity anchor
(voice, script → TTS, talking-head animation) — the forge only has to succeed once.
