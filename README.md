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
scripts/run_kaggle_job.py               # Phase 1 Kaggle CLI driver
scripts/run_talking_job.py              # Phase 2 Kaggle CLI driver
kaggle_kernel_talking/                  # Phase 2 kernel + shipped portrait
assets/                                 # permanent crops (master face, podcast)
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

## Phase 2 · Micro-Pilot 2 — "Voice & Talking Avatar"

A **5.6 s one-shot speaking clip** of Julian: Kokoro-ONNX voice (blended style
vectors, no spacy/blis compilation) + audio-driven talking portrait, encoded to
H.264/AAC at 25 FPS on a Kaggle T4.

| Artifact | Detail |
| --- | --- |
| `outputs/pilot_talking.mp4` | 832×1024, 25 fps, 140 frames, 5.60 s, libx264 + yuv420p, AAC |
| `outputs/julian_voice.mp3` | 0.6·`bm_george` + 0.4·`am_michael` @ speed 0.92, mastered to −1.5 dBFS |
| `outputs/talking_contact_sheet.png` | 8 keyframes + mouth-zoom strip + waveform with lip-openness curve |
| `outputs/talking_metrics.json` | duration, fps, frame count, lip-openness↔RMS correlation, timings |

### Voice chain (zero C++ compilation)
`kokoro-onnx` → ONNX Runtime (CUDA provider when available) + `espeakng-loader`
for phonemes. Mastering is pure numpy: 80 Hz high-pass (24 dB/oct), +2.2 dB
@190 Hz warmth, −1.2 dB @4.2 kHz, peak-normalised to −1.5 dBFS.

### Lip-sync backend (always reported in `talking_metrics.json`)

The kernel reports exactly which tier ran; it never implies accuracy it does not have.

* **`wav2lip-onnx`** — attempted first. Weights are probed from public
  HuggingFace repos within a time budget, the ONNX signature is introspected at
  runtime (so any export layout works) and the first prediction is validated
  (shape/mean/std) before the clip commits to it. *On the 2026-11 runs every
  public `wav2lip*.onnx` repository returned 404/401, so this tier never engaged.*
* **`envelope-mouth-synthesiser`** — the tier that actually ran. Kokoro's
  `kokoro-v1.0.onnx` export has **no duration output**, so per-phoneme timings do
  not exist; the mouth opening is driven by the acoustic envelope, with a
  lower-lip translation plus shadowed oral cavity (no smearing) and a feathered
  unsharp mask over the mouth. It is *approximate*: mouth activity is synced to
  speech energy, not phoneme-accurate visemes.
* Motion layer: envelope-driven yaw/pitch/roll + sway, 4.2 s breathing, natural
  blinks, 0.5 s fade in/out.
* `mouth_open_vs_audio_rms_correlation` is `null` with an explanatory note when
  the openness curve is derived from the envelope (self-correlation is not a
  metric). Speech longer than the clip is time-stretched to fit (WSOLA overlap-add,
  pitch preserved) instead of being truncated.

### Running Phase 2
Push to any `arena/**` branch (or dispatch **Talking Pilot · Micro-Pilot 2**).
Phase 1 is now `workflow_dispatch`-only so it can never spend quota on push.

```
kaggle_kernel_talking/
  talking_pilot.py            # kernel: voice + lipsync + motion + encode + sheet
  portrait.png                # shipped with the push (1024x1024 master face)
scripts/run_talking_job.py    # Kaggle driver with the Phase 2 manifest
.github/workflows/talking_pilot.yml
```

### Verified locally before the first push
`kokoro-onnx` import + espeak phonemisation, voice blending, mastering, viseme
mapping, mouth synthesis, H.264/AAC encode, contact sheet, the metrics contract,
the orchestrator (stub CLI: metadata → push → poll → download → validation) and
the workflow's publish step (dry-run against a scratch clone).

## Next pilots

Phase 1 artifacts are permanent assets in `assets/` (`julian_master_face.png`,
`julian_podcast_studio.png`). Phase 3 can layer script generation and batch
renders on top of the Phase 2 talking-portrait pipeline.
