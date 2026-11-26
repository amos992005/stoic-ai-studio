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
effortlessly handsome young-millionaire playboy** (young Brad Pitt in *Ocean's
Eleven*) whose life is yachts, VIP nightclubs, supercars and skydiving — the
Instagram/TikTok lifestyle format.

> **Luca Moretti** — 27 · sun-bleached tousled dirty-blonde hair, sharp jawline,
> light 3-day stubble, piercing blue-green eyes, confident mischievous smirk,
> lean athletic swimmer/fighter physique.

One Kaggle T4 run produces the whole postable pack.

| Artifact | Detail |
| --- | --- |
| `luca_carousel_sheet.png` | Inspection grid: master + the four vertical slides |
| `luca_slide_01.jpg` … `05.jpg` | Post-ready JPEGs, quality 92 |
| `01_luca_master.png` | 1024×1024 identity master, seed `777`, 30 steps, CFG 5.5, DPM++ 2M Karras |
| `02_monaco_yacht_date.png` | 832×1024 · Monaco sunset, Riva yacht, brunette on the right |
| `03_vip_party_flash.png` | 832×1024 · Ibiza rooftop, direct on-camera flash |
| `04_porsche_gt3_pov.png` | 832×1024 · Amalfi cliff road, Porsche 911 GT3 RS, passenger POV |
| `05_extreme_action.png` | 832×1024 · open helicopter door above a turquoise coastline |
| `luca_voice.mp3` | kokoro-onnx `am_adam` @ speed 1.0 — **peak normalisation only** |
| `luca_reel_clip.mp4` | 576×1024 · 25 fps · 25 real **Stable Video Diffusion** frames → audio-matched |
| `luca_video_sheet.png` | 8-frame strip of the neural clip |
| `luca_metrics.json` | per-slide seeds/masks, timings, env fingerprint, video probe |

### Stopping face bleed with a masked IP-Adapter

The hard part of a two-person slide is that a face reference leaks onto whoever
else is in frame — Luca's identity would end up on the brunette. The fix is
spatial attention masking, not post-processing:

```python
cross_attention_kwargs={"ip_adapter_masks": [mask]}   # mask: [1, num_ip_images, H, W]
```

`mask` is `1.0` over the left half (where Luca stands), ramps smoothly to `0.0`
by 60 % of the width, so the woman on the right keeps her own face while the
man on the left keeps Luca's. Verified against `diffusers` 0.42 source and by
running a real `IPAdapterPlusImageProjection` UNet: left-vs-right masks change
the output by ~0.055 (mean abs), and mask/image count mismatches raise loudly.
Slides `04`/`05` are single-subject, so they use the unmasked (stronger) path.

If the masked call ever raises at runtime the kernel retries that slide
unmasked rather than losing it, and records `masked_ip_adapter: false` plus
`unmasked_retry: true` in `luca_metrics.json` — the degradation is always
visible in the metrics, never silent.

### Video: real neural motion, honest timing

`stabilityai/stable-video-diffusion-img2vid-xt` is public and needs **no auth
token**. It generates 25 genuine neural frames at 576×1024 (fp16, VAE
slicing/tiling, model CPU offload, `decode_chunk_size=4`, with a fallback
ladder for OOM). Those frames are then:

1. joined into a **ping-pong cycle** (forward, then backward) so the loop has no
   seam and no duplicated frame;
2. **retimed by cross-fade temporal upsampling** from SVD's native 7 fps to
   25 fps, so the motion keeps its true speed instead of playing 3.5× fast;
3. trimmed to exactly the natural length of the voice track, then muxed.

Every output pixel is a blend of two frames the model actually generated.
There is **no** procedural animation, no pixel-slice mouth patch, no drawn
ellipse anywhere in Phase 3.

### Audio: one multiply, nothing else

The banned techniques from Phase 2 are gone for good. The entire signal chain is:

```python
mastered = np.clip(audio / np.max(np.abs(audio)) * 0.85, -1.0, 1.0)
```

No WSOLA time-stretching (it shifted a 120 Hz test tone to 132 Hz), no
biquad high-pass, no parametric EQ. The voice style matrix is passed **by name**
so kokoro indexes the full `(510, 1, 256)` tensor — slicing it to `(1, 256)` is
what caused Phase 2's `Invalid rank for input: style` failure. The video is
fitted to the audio, never the other way round.

### Running Phase 3

Push to any `arena/**` branch touching `kaggle_kernel_luca/**`,
`scripts/run_luca_job.py` or the workflow, or dispatch **Luca Studio · Phase 3**
with optional `slide_steps` / `svd_frames` / `voice` inputs. Phases 1 and 2 are
`workflow_dispatch`-only: they can never spend quota on a push.

The kernel needs **no auxiliary dataset** — every weight (RealVisXL, CLIP-ViT-H,
IP-Adapter-Plus-Face, SVD-XT, Kokoro) comes from the public hub at run time.

### Verified locally before the first Kaggle run

Full `main()` executed end-to-end against stubbed pipelines (~14 s): all 17
artifacts produced, `576×1024` H.264 High + AAC 24 kHz mono muxed at 25 fps,
`luca_voice.mp3` peak −1.41 dBFS with zero clipped samples, per-slide mask
records correct (`left`/`left`/`full`/`full`, two with `masked_ip_adapter:
true`). Local rehearsal caught two real bugs pre-flight: a `set + set`
`TypeError` in the metrics finaliser and an ffmpeg deadlock that hung the
kernel for the full timeout when a frame write failed.

## Next pilots

Phase 1 artifacts are permanent assets in `assets/` (`julian_master_face.png`,
`julian_podcast_studio.png`). Phase 3 establishes the reusable core — masked
identity locking + real neural motion + a clean TTS chain — so Phase 4 can add
script generation, multi-slide batch renders and a scheduled posting loop.
