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
.github/workflows/character_forge.yml   # Phase 1 · dispatch only
.github/workflows/talking_pilot.yml     # Phase 2 · dispatch only
.github/workflows/luca_studio.yml       # Phase 3 · dispatch + push on arena/**
kaggle_kernel/
  character_forge.py                    # the GPU kernel (SDXL + IP-Adapter Face)
  kernel-metadata.json                  # generated at run time (not committed)
kaggle_kernel_talking/                  # Phase 2 kernel + shipped portrait
kaggle_kernel_luca/
  luca_studio.py                        # Phase 3 kernel (SDXL + masked IP-Adapter +
                                        #   SVD-XT + kokoro) — no auxiliary dataset
scripts/run_kaggle_job.py               # Phase 1 Kaggle CLI driver
scripts/run_talking_job.py              # Phase 2 Kaggle CLI driver
scripts/run_luca_job.py                 # Phase 3 Kaggle CLI driver
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
  metric). Speech longer than the clip was time-stretched to fit (WSOLA overlap-add)
  instead of being truncated.

> ⚠️ **Superseded.** WSOLA time-stretching and the envelope-driven mouth patch are
> now **banned** — the WSOLA path measurably shifted a 120 Hz test tone to 132 Hz,
> and the mouth patch produced visible "fake mouth" artefacts. Phase 3 replaces
> both: peak normalisation is the only audio processing, and every moving pixel
> comes from Stable Video Diffusion. This section is kept as an honest record of
> what the Phase 2 run actually did.

### Running Phase 2
**Manual dispatch only** (`Talking Pilot · Micro-Pilot 2`). The workflow no longer
auto-runs on push: Phase 3's real neural video supersedes the envelope-mouth
synthesiser, and this workflow exists to reproduce the Phase 2 artifacts on demand.

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

## Phase 3 · "Luca Moretti" — flagship character

Julian was the technical proof. Phase 3 is the product: a **27-year-old
effortlessly handsome young-millionaire playboy** whose life is yachts, VIP
nightclubs and supercars — the Instagram/TikTok lifestyle format.

> **Luca Moretti** — 27 · sun-kissed dirty blonde tousled hair, sharp jawline,
> light 3-day stubble, piercing blue-green eyes, athletic build.

He is described **only** by those attributes. Naming a real actor does not just
invite likeness problems, it *ages the character*: the model reproduces the
person it has seen in the training data, which is a fifty-something. The name
list lives in `FORBIDDEN_TERMS` and `check_prompt()` raises before a prompt ever
reaches a sampler.

### Architecture: identity in the weights, not in an adapter

Two Kaggle T4s, one run, two independent processes:

```
cuda:0  generate_images.py                    cuda:1  generate_video.py
──────────────────────────────                ─────────────────────────────
1. master portrait (plain SDXL)              1. Chatterbox-Turbo → luca_voice.wav
2. 15 variants via InstantID       ──────▶   2. polls for 01_luca_master.png
3. del InstantID + empty_cache              3. SadTalker (512, crop, gfpgan)
4. train SDXL LoRA (~10 min)                4. mandatory H.264/AAC mux
5. carousel: RealVisXL + LoRA only          5. ffprobe verdict
```

The four carousel images are generated by **RealVisXL + the LoRA and nothing
else** — no InstantID, no ControlNet, no reference image. That is the entire
point of training: once the identity is in the weights, the pipeline is plain
SDXL and every prompt is free to put him anywhere. InstantID is used strictly to
*manufacture the training set*, and both it and its 1.7 GB projection network are
unloaded before the first training step, because on a 14.5 GB card the training
UNet cannot coexist with them.

| Artifact | Detail |
| --- | --- |
| `01_luca_master.png` | 1024² close-up portrait · also the avatar's source image |
| `02_luca_yacht.png` | 832×1024 · full body at the bow, Monaco |
| `03_luca_porsche.png` | 832×1024 · Porsche 911 GT3 RS, Amalfi cliff road |
| `04_luca_party.png` | 832×1024 · Ibiza rooftop, on-camera flash |
| `luca_lora.safetensors` | the trained adapter — the identity, as weights |
| `luca_voice.wav` | Chatterbox-Turbo, 24 kHz mono, peak-normalised |
| `luca_reel.mp4` | SadTalker 512 + GFPGAN, H.264 High/yuv420p + AAC, faststart |
| `luca_metrics.json` | per-stage timings, LoRA loss curve, ffprobe verdict |

### Framing is prompt engineering, not luck

Every slide carries `candid 24mm iPhone 16 Pro photo`, `natural sunlight`,
`skin pores`, and an explicit `medium full shot` / `full body shot`, and the
negative prompt forbids `hands near face` and `car door clipping` — the
tokens that stop him being cropped at the waist by a car door or occluded by his
own hand. The master is the deliberate exception: it is a close-up, because it
has to work as a talking head.

### Training recipe

`peft` LoRA, `r=16`, `alpha=16`, `init_lora_weights="gaussian"`, targeting
`to_q/to_k/to_v/to_out.0` on the SDXL UNet. RealVisXL V5.0 base, float16 UNet with
gradient checkpointing, float32 trainable parameters, 768 px, batch 1, lr 1e-4,
400 steps, gradient clipping at 1.0.

Two things in `luca_lora.py` are load-bearing rather than stylistic:

* **`patch_vae_encoder()`** — `diffusers 0.42`'s `Encoder.forward` assigns the
  `down_blocks` tuple straight to `sample` and dies on the *encode* path. Pipelines
  only decode, so generation never hits it; latent caching always does. The patch
  wraps each down block to return `out[0]` and is idempotent.
* **float32 adapter parameters** — fp16 trainable weights with AdamW produce a
  `nan` loss by step six, and `save_adapter` will happily write a dead adapter.
  Non-finite loss is therefore fatal, not logged.

The adapter is saved with `base_model.model.` → `unet.` key renaming and float32
tensors, which is the layout `pipe.load_lora_weights(...)` expects.

### Video: SadTalker, then a mux that is not optional

`stabilityai/stable-video-diffusion` and MuseTalk are both gone — SVD-XT is
banned outright, and a lip-sync face swap was rejected. The avatar is SadTalker:
`--preprocess crop --size 512 --enhancer gfpgan`, and **no `--still`** (that flag
pins the head to the source framing, which is the opposite of what an avatar
needs).

SadTalker's own output is not guaranteed iPhone-playable, so the reel is always
re-encoded:

```
ffmpeg -y -i raw_video.mp4 -i luca_voice.wav \
  -c:v libx264 -pix_fmt yuv420p -profile:v high -crf 20 \
  -c:a aac -b:a 160k -movflags +faststart -shortest luca_reel.mp4
```

`yuv420p` matters because libx264 may otherwise pick a 4:4:4 profile that Safari
and QuickTime refuse to decode, and `+faststart` moves the moov atom to the front
so the reel starts before it has finished downloading. The result is then
**probed**, not assumed: `verify_reel()` fails the run unless ffprobe reports
`h264` + `yuv420p` + `aac`, at least two seconds, and under half a second of
audio drift.

### Running it

Push to any `arena/**` branch that touches `kaggle_kernel_luca/**`,
`scripts/luca_v2/**` or the workflow, or dispatch **Luca Studio · v2**. Exactly
one Kaggle job per iteration.

`kaggle kernels push` uploads only a single code file, so
`scripts/build_luca_kernel.py` generates `kaggle_kernel_luca/luca_studio.py` with
all seven sources embedded; `--check` in CI fails the build if the generated copy
has drifted from `scripts/luca_v2/`. The workers are then spawned with one
physical GPU each (`CUDA_VISIBLE_DEVICES=0` / `=1`), so code that allocates
without an explicit device cannot land on the wrong card.

### Verified locally before the Kaggle run

`python scripts/verify_luca.py` — ten checks, no GPU and no network:

```
PASS  identity spec: one description, varied dataset, no celebrity
PASS  carousel prompts carry explicit framing
PASS  shared metrics merge survives concurrent writers
PASS  LoRA trainer: VAE patch, adapter, float32 params
PASS  stage order and LoRA-only inference
PASS  reel gate verifies h264/yuv420p/aac + sync
PASS  SadTalker invocation and weights
PASS  launcher stages its payload and starts both workers
PASS  banned techniques absent
PASS  generated kernel matches its sources
```

The interesting ones are executed rather than asserted: `luca_lora.train()` runs
a real backward pass on a tiny SDXL, `verify_reel()` is fed four real ffprobe
verdicts (one good, three bad ones it must reject), the launcher is actually
spawned as a script and killed once both workers are alive, and the metrics
merge is hammered from two threads to prove the `flock` holds.

## Next pilots

Phase 1 artifacts are permanent assets in `assets/` (`julian_master_face.png`,
`julian_podcast_studio.png`). Phase 3 establishes the reusable core — masked
identity locking + real neural motion + a clean TTS chain — so Phase 4 can add
script generation, multi-slide batch renders and a scheduled posting loop.
