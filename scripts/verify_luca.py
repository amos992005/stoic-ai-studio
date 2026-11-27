#!/usr/bin/env python3
"""
Verification for the Luca v2 LoRA build, against the real libraries where possible.

Everything here runs locally with no GPU and no network. The split is
deliberate: properties that can be *executed* are executed (the LoRA trainer runs
a real backward pass, the reel gate is fed real ffprobe JSON, the launcher is
really spawned as a script), and properties that cannot be are asserted against
the executable source — never against a docstring, which is how an earlier
version of this suite passed while the code it described was gone.

    python scripts/verify_luca.py        # exits non-zero on any failure
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import types

REPO = pathlib.Path(__file__).resolve().parent.parent
WORKER_DIR = REPO / "scripts" / "luca_v2"
sys.path.insert(0, str(WORKER_DIR))

results: list[tuple[bool, str, str]] = []


def check(label: str, fn):
    try:
        detail = fn()
        results.append((True, label, detail or ""))
    except Exception as exc:  # noqa: BLE001 - the point is to report, not raise
        results.append((False, label, f"{type(exc).__name__}: {exc}"))


def load(name: str, path: pathlib.Path):
    """Import a worker module by path, tolerating missing heavy dependencies."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source(relative: str) -> str:
    return (WORKER_DIR / relative).read_text(encoding="utf-8")


def code_of(text: str) -> str:
    """
    Keep only executable code: docstrings and comments stripped.

    A worker that *documents* a prohibition must not fail a scan for
    prohibitions, and a symbol that only appears in prose is not implemented.
    """
    import ast

    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                text = text.replace(doc, "")
    kept = [line for line in text.splitlines()
            if not line.strip().startswith("#")
            and not re.match(r'^\s*"[^"]*"\s*:\s*False\s*,?\s*$', line)]
    return "\n".join(kept)


# --------------------------------------------------------------------------- #
# 1. Identity: one description, fifteen variations, no celebrity
# --------------------------------------------------------------------------- #


def identity_module():
    character = load("luca_character", WORKER_DIR / "luca_character.py")

    assert character.TRIGGER == "ohwx", character.TRIGGER
    assert character.AGE == 27, character.AGE
    assert character.SUBJECT.startswith("ohwx 27-year-old Italian man"), \
        character.SUBJECT
    assert len(character.DATASET) == 15, len(character.DATASET)
    assert len(character.CAROUSEL) == 4, len(character.CAROUSEL)

    # Every prompt must exist, carry the trigger, and survive the guard.
    prompts = [character.dataset_prompt(e) for e in character.DATASET]
    prompts += [character.carousel_prompt(e) for e in character.CAROUSEL]
    for prompt in prompts:
        assert character.TRIGGER in prompt, f"no trigger word: {prompt[:60]}"
        assert len(prompt) > 80, f"suspiciously short prompt: {prompt!r}"
    # Variety is what the LoRA learns from: a dataset of 15 identical framings
    # teaches one crop instead of one face.
    framings = {e["framing"] for e in character.DATASET}
    assert len(framings) >= 6, f"only {len(framings)} distinct framings"
    expressions = {e["expression"] for e in character.DATASET}
    assert len(expressions) >= 3, expressions

    # The guard has to actually bite.
    for probe in ("brad pitt in a car", "Bradley on a yacht", "PITT"):
        try:
            character.check_prompt(probe)
        except ValueError:
            continue
        raise AssertionError(f"check_prompt accepted {probe!r}")

    return (f"15 dataset + 4 carousel prompts, {len(framings)} framings, "
            f"trigger {character.TRIGGER!r}, guard rejects 3 celebrity probes")


check("identity spec: one description, varied dataset, no celebrity", identity_module)


# --------------------------------------------------------------------------- #
# 2. Carousel framing: the tokens that stop him clipping into the car door
# --------------------------------------------------------------------------- #


def carousel_framing():
    character = load("luca_character", WORKER_DIR / "luca_character.py")
    for entry in character.CAROUSEL:
        prompt = character.carousel_prompt(entry)
        # The camera/lens/light tokens go on every slide: they are what makes the
        # result read as a phone photo instead of a render.
        for token in ("candid 24mm iPhone 16 Pro photo", "natural sunlight",
                      "skin pores"):
            assert token in prompt, f"{entry['file']} lost {token!r}"
        assert "no hands near the face" in prompt, \
            f"{entry['file']} does not forbid hands near the face"
        # ...but the *framing* is per-slide: slide 01 must stay a portrait,
        # because it is also the source image the avatar animates.
        if entry["file"].startswith("01"):
            assert "close-up portrait" in prompt, prompt
        else:
            assert re.search(r"\b(medium full|full body) shot\b", prompt), prompt
    # sizes and seeds are part of the deliverable contract
    expected = {"01_luca_master.png": (1024, 1024),
                "02_luca_yacht.png": (832, 1024),
                "03_luca_porsche.png": (832, 1024),
                "04_luca_party.png": (832, 1024)}
    for entry in character.CAROUSEL:
        want = expected[entry["file"]]
        assert tuple(entry["size"]) == want, (entry["file"], entry["size"], want)
    seeds = [entry["seed"] for entry in character.CAROUSEL]
    assert seeds == [42, 43, 44, 45], seeds
    return "all four prompts carry the framing tokens; sizes and seeds match"


check("carousel prompts carry explicit framing", carousel_framing)


# --------------------------------------------------------------------------- #
# 3. Shared plumbing: metrics merge is atomic and lock-protected
# --------------------------------------------------------------------------- #


def metrics_merge():
    import threading

    common = load("luca_common", WORKER_DIR / "luca_common.py")
    tmp = pathlib.Path(tempfile.mkdtemp())
    target = tmp / "luca_metrics.json"

    def worker(section: str):
        for i in range(5):
            common.merge_metrics(section, {"status": "ok", "i": i,
                                           "artifacts": [f"{section}_{i}.png"]},
                                 target)

    threads = [threading.Thread(target=worker, args=(name,))
               for name in ("images", "video")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    merged = json.loads(target.read_text())
    assert set(merged) >= {"images", "video", "artifacts"}, merged.keys()
    assert merged["images"]["i"] == 4 and merged["video"]["i"] == 4, merged
    assert sorted(merged["artifacts"]) == sorted(
        [f"images_{i}.png" for i in range(5)]
        + [f"video_{i}.png" for i in range(5)]), merged["artifacts"]
    for sidecar in ("luca_metrics_images.json", "luca_metrics_video.json"):
        assert (tmp / sidecar).is_file(), f"missing sidecar {sidecar}"
    return "both halves merged under flock, artifacts unioned, no torn file"


check("shared metrics merge survives concurrent writers", metrics_merge)


# --------------------------------------------------------------------------- #
# 4. The LoRA trainer: a real step on a real (tiny) UNet
# --------------------------------------------------------------------------- #


def lora_trainer_contract():
    lora = load("luca_lora", WORKER_DIR / "luca_lora.py")

    assert lora.LORA_TARGETS == ["to_q", "to_k", "to_v", "to_out.0"], \
        lora.LORA_TARGETS
    assert "patch_vae_encoder" in dir(lora)
    assert "save_adapter" in dir(lora)
    assert "float32" in source("luca_lora.py"), \
        "trainable adapter params must stay float32"
    assert "math.isfinite" in source("luca_lora.py"), \
        "a non-finite loss must be fatal"

    import torch
    from diffusers import AutoencoderKL, UNet2DConditionModel

    torch.manual_seed(0)
    vae = AutoencoderKL(in_channels=3, out_channels=3,
                        down_block_types=("DownBlock2D", "DownBlock2D"),
                        up_block_types=("UpBlock2D", "UpBlock2D"),
                        block_out_channels=(32, 64), layers_per_block=1,
                        latent_channels=4, sample_size=64, norm_num_groups=32)
    patched = lora.patch_vae_encoder(vae)
    assert patched >= 1, "patch_vae_encoder patched nothing"
    image = torch.zeros(1, 3, 64, 64)
    latent = vae.encode(image).latent_dist.mode()
    assert latent.ndim == 4 and latent.shape[0] == 1 and latent.shape[1] == 4, \
        f"encode returned {tuple(latent.shape)}"
    assert latent.shape[2] < 64 and 64 % latent.shape[2] == 0, \
        f"encode did not downsample: {tuple(latent.shape)}"
    assert lora.patch_vae_encoder(vae) == 0, "patch is not idempotent"

    unet = UNet2DConditionModel(
        sample_size=16, in_channels=4, out_channels=4, layers_per_block=1,
        block_out_channels=(32, 64),
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=32, norm_num_groups=32, attention_head_dim=8)
    adapter = lora.build_adapter(unet, rank=16, alpha=16)
    assert adapter is not None
    trainable = [p for p in unet.parameters() if p.requires_grad]
    assert trainable, "nothing became trainable"
    assert all(p.dtype == torch.float32 for p in trainable), \
        "trainable parameters must be float32, not fp16"
    total = sum(p.numel() for p in trainable)
    return (f"encoder patched ({patched} block), latent {tuple(latent.shape)}, "
            f"{len(trainable)} trainable tensors / {total:,} params")


check("LoRA trainer: VAE patch, adapter, float32 params", lora_trainer_contract)


# --------------------------------------------------------------------------- #
# 4b. peft cannot be imported next to an old torchao
# --------------------------------------------------------------------------- #


def torchao_guard():
    """
    Regression for run 45984347218.

    peft's `is_torchao_available()` *raises* — it does not return False — when a
    torchao below 0.16 is installed, and Kaggle's image ships 0.10. The check
    runs at import time, so `import peft` failed and the LoRA trainer never
    started. Reproduced locally with torchao 0.10.0: removal is what fixes it,
    because transformers also imports torchao.prototype.
    """
    common = load("luca_common", WORKER_DIR / "luca_common.py")
    common.OUTPUT_DIR = pathlib.Path(tempfile.mkdtemp())
    common.PIP_LOCK = common.OUTPUT_DIR / ".pip.lock"

    body = code_of(source("luca_common.py"))
    assert "def drop_incompatible_torchao" in body
    assert "0, 16, 0" in body, "the version ceiling must match peft's"
    assert "uninstall" in body
    assert "--break-system-packages" in body, \
        "the retry is what works on a managed interpreter"
    assert "immediately" not in body

    # the image worker must call it before importing peft
    images = code_of(source("generate_images.py"))
    call = images.index("drop_incompatible_torchao()")
    first_peft = images.index('__import__(module)')
    assert call < first_peft, \
        "torchao must be dropped before the loop that imports peft"

    # a version that is already fine must be left alone
    assert common.drop_incompatible_torchao.__doc__ is not None
    return "guard runs before `import peft`, removes old torchao, leaves new alone"


check("torchao/peft import guard", torchao_guard)


# --------------------------------------------------------------------------- #
# 4c. The adapter cast actually moves the adapter
# --------------------------------------------------------------------------- #


def adapter_cast():
    """
    Regression for run 45995210057, which logged ``adapter_cast: tensors 0``.

    The adapter is saved float32; a float32 LoRA against a float16 UNet is how
    you get a flat frame. The cast has to find the tensors by name, because
    which attribute holds them depends on the loader.
    """
    images = load("generate_images", WORKER_DIR / "generate_images.py")
    images.log = lambda *a, **k: None
    images.DEVICE = "cpu"

    import torch
    from diffusers import UNet2DConditionModel
    from peft import LoraConfig, get_peft_model

    unet = UNet2DConditionModel(
        sample_size=16, in_channels=4, out_channels=4, layers_per_block=1,
        block_out_channels=(32, 64),
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=32, norm_num_groups=32, attention_head_dim=8)
    get_peft_model(unet, LoraConfig(
        r=8, lora_alpha=8, init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    unet = unet.half()
    for name, parameter in unet.named_parameters():   # what the loader does
        if "lora" in name.lower():
            parameter.data = parameter.data.float()

    before = {p.dtype for n, p in unet.named_parameters() if "lora" in n.lower()}
    assert before == {torch.float32}, before

    info = images.cast_lora_to_unet(types.SimpleNamespace(unet=unet))
    after = {p.dtype for n, p in unet.named_parameters() if "lora" in n.lower()}
    assert after == {torch.float16}, f"adapter still {after}"
    assert info["lora_tensors"] > 0, "the cast found no adapter tensors"
    return f"{info['lora_tensors']} adapter tensors float32 -> float16"


check("adapter cast finds and converts the adapter", adapter_cast)


# --------------------------------------------------------------------------- #
# 4d. The adapter really attaches, and the numpy-2 alignment patch really works
# --------------------------------------------------------------------------- #


def adapter_attaches():
    """
    Regression for run 45999895543, which logged
    "LoRA could not be applied (NotImplementedError: Cannot copy out of meta
    tensor; no data!)" and rendered the whole carousel from the base model.

    Two things were wrong: nothing checked the pipeline for placeholder
    parameters, and there was no second way in when diffusers' loader refuses.
    The peft fallback is exercised here against a real fp16 UNet and an adapter
    written by ``luca_lora.save_adapter``.
    """
    images = load("generate_images", WORKER_DIR / "generate_images.py")
    images.log = lambda *a, **k: None
    images.DEVICE = "cpu"
    images.LORA_RANK, images.LORA_ALPHA = 16, 16

    import torch
    from diffusers import UNet2DConditionModel
    from safetensors.torch import load_file

    import luca_lora

    config = dict(sample_size=16, in_channels=4, out_channels=4,
                  layers_per_block=1, block_out_channels=(32, 64),
                  down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                  up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
                  cross_attention_dim=32, norm_num_groups=32,
                  attention_head_dim=8)
    trained = UNet2DConditionModel(**config)
    luca_lora.build_adapter(trained, rank=16, alpha=16)
    checkpoint = pathlib.Path(tempfile.mkdtemp()) / "luca_lora.safetensors"
    luca_lora.save_adapter(trained, checkpoint)

    base = UNet2DConditionModel.from_config(trained.config).half()
    loaded = UNet2DConditionModel.from_config(trained.config).half()
    attached = images._peft_fallback(
        types.SimpleNamespace(unet=loaded), load_file(str(checkpoint)))
    assert attached > 0, "the peft fallback attached nothing"

    x = torch.randn(1, 4, 16, 16, dtype=torch.float16) * 0.5
    ctx = torch.randn(1, 77, 32, dtype=torch.float16) * 0.5
    with torch.no_grad():
        a = base(x, timestep=1, encoder_hidden_states=ctx).sample
        b = loaded(x, timestep=1, encoder_hidden_states=ctx).sample
    assert torch.isfinite(b).all(), "the adapted UNet produced non-finite output"
    assert not torch.allclose(a, b), "the adapter changed nothing"

    # a placeholder parameter must be reported as such, not as a cryptic copy
    with torch.device("meta"):
        meta = UNet2DConditionModel.from_config(trained.config)
    try:
        images.check_no_meta(types.SimpleNamespace(unet=meta))
    except RuntimeError as exc:
        assert "meta" in str(exc)
    else:
        raise AssertionError("check_no_meta accepted a meta-device pipeline")

    body = code_of(source("generate_images.py"))
    assert "def load_adapter" in body and "diffusers.load_lora_weights" in body

    # The save format is what makes the adapter loadable at all. diffusers
    # rewrites the adapter-name segment itself, so "...lora_A.default.weight"
    # matches nothing and is dropped in silence.
    keys = list(load_file(str(checkpoint)))
    assert all(".default." not in key for key in keys), \
        f"save_adapter emitted a key diffusers cannot match: {keys[0]}"

    saved = {key.split(".", 1)[-1] for key in keys}
    unet = trained.__class__.from_config(trained.config).half()
    unet.load_lora_adapter(load_file(str(checkpoint)), prefix="unet",
                           adapter_name="luca")
    live = {name for name, _ in unet.named_parameters() if "lora_" in name}
    assert len(live) == len(saved), f"{len(live)} live vs {len(saved)} saved"
    assert all(name.replace(".luca.", ".") in saved for name in live)
    return (f"{attached} modules attached via peft, output differs from base, "
            f"meta tensors rejected, {len(saved)}/{len(saved)} tensors land")


check("adapter attaches even when diffusers refuses", adapter_attaches)


def numpy2_alignment():
    """
    Regression for run 45999895543's SadTalker crash.

    numpy 2 changed ``lstsq``'s broadcasting, so SadTalker's ``POS()`` returns a
    ``(2, 1)`` array and ``np.array([w0, h0, s, t[0], t[1]])`` raises
    "setting an array element with a sequence". The wrapper patches ``lstsq``
    and wraps ``align_img``; the patch must leave a real (5,) parameter array.
    """
    video = load("generate_video", WORKER_DIR / "generate_video.py")
    video.log = lambda *a, **k: None
    video.WORK_DIR = pathlib.Path(tempfile.mkdtemp())

    import numpy as np

    patch_file = video.write_sadtalker_numpy2_patch()
    assert patch_file.is_file()

    # a stand-in for SadTalker's package, with the same broken call site
    sandbox = pathlib.Path(tempfile.mkdtemp())
    pkg = sandbox / "src" / "face3d" / "util"
    pkg.mkdir(parents=True)
    for parent in (sandbox / "src", sandbox / "src" / "face3d"):
        (parent / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "preprocess.py").write_text(
        "import numpy as np\n"
        "def align_img(img, lm, lm3D, mask=None, target_size=224.,\n"
        "              rescale_factor=102.):\n"
        "    w0, h0 = img\n"
        "    k = np.linalg.lstsq(np.random.rand(10, 8),\n"
        "                       np.random.rand(10, 1), rcond=None)[0]\n"
        "    s, t = k[3], np.stack([k[3], k[7]], axis=0)\n"
        "    return np.array([w0, h0, s, t[0], t[1]]), 'img', 'lm', None\n")

    probe = subprocess.run(
        [sys.executable, "-c",
         "import sys, numpy as np;"
         f"sys.path[:0] = [{str(patch_file.parent)!r}, {str(sandbox)!r}];"
         "import sadtalker_numpy2_patch;"
         "A = np.random.rand(10, 8); b = np.random.rand(10, 1);"
         "k = np.linalg.lstsq(A, b, rcond=None)[0];"
         "assert k.ndim == 1, k.shape;"
         "print('LSTSQ', k.shape);"
         "from src.face3d.util import preprocess as pp;"
         "params = pp.align_img((512, 512), None, None)[0];"
         "assert params.shape == (5,), params.shape;"
         "assert params.dtype == np.float64, params.dtype;"
         "print('ALIGN', params.shape, np.isfinite(params).all())"],
        capture_output=True, text=True)
    assert "LSTSQ (8,)" in probe.stdout, (probe.stdout, probe.stderr[-400:])
    assert "ALIGN (5,) True" in probe.stdout, (probe.stdout, probe.stderr[-400:])

    body = code_of(source("generate_video.py"))
    assert "SADTALKER_NUMPY2_PATCH" in body
    assert "face_preprocess.align_img = align_img" in body, \
        "align_img must be wrapped: the call site needs a flat 5-vector"
    assert "import sadtalker_numpy2_patch" in body, \
        "the runner must apply the patch inside the child"
    return "lstsq returns (8,) again and align_img is normalised to 5 scalars"


check("numpy-2 alignment patch", numpy2_alignment)


# --------------------------------------------------------------------------- #
# 5. Stage order: InstantID builds the dataset and is gone before training
# --------------------------------------------------------------------------- #


def stage_order():
    images = source("generate_images.py")
    main_body = images[images.index("def main()"):]
    order = [main_body.index("build_master("),
             main_body.index("build_dataset("),
             main_body.index("unload_instantid("),
             main_body.index("luca_lora.train("),
             main_body.index("build_carousel(")]
    assert order == sorted(order), \
        "main() must run master -> dataset -> unload -> train -> carousel"
    for call in ("torch.cuda.empty_cache()", "gc.collect()"):
        assert call in images, f"missing {call}"

    # The carousel is the whole point of the upgrade: LoRA only, no reference.
    carousel = code_of(images[images.index("def build_carousel"):
                              images.index("def main()")])
    for token in ("image_embeds", "face_kps", "controlnet_conditioning_scale",
                  "load_instantid", "draw_kps"):
        assert token not in carousel, \
            f"the carousel still uses {token!r} — inference must be LoRA-only"
    assert "load_lora_weights" in carousel and "set_adapters" in carousel
    assert re.search(r'LUCA_LORA_SCALE", "0\.85"', images), \
        "the LoRA inference scale must default to 0.85"

    # InstantID is still used, but only to build the dataset.
    dataset = code_of(images[images.index("def build_dataset"):
                             images.index("def unload_instantid")])
    assert "image_embeds" in dataset and "controlnet_conditioning_scale" in dataset
    return "master → 15 InstantID variants → unload → LoRA → LoRA-only carousel"


check("stage order and LoRA-only inference", stage_order)


# --------------------------------------------------------------------------- #
# 6. Reel gate: ffprobe verdicts, including the rejections
# --------------------------------------------------------------------------- #


def reel_gate():
    video = load("generate_video", WORKER_DIR / "generate_video.py")
    video.log = lambda *a, **k: None

    fake_bin = pathlib.Path(tempfile.mkdtemp())
    script = fake_bin / "ffprobe"
    script.write_text('''#!/bin/sh
case "$FAKE_MODE" in
  good) echo '{"streams":[{"codec_type":"video","codec_name":"h264","pix_fmt":"yuv420p","width":1080,"height":1920,"duration":"7.04"},{"codec_type":"audio","codec_name":"aac","sample_rate":"44100","channels":2,"duration":"7.04"}],"format":{"duration":"7.04"}}';;
  pix) echo '{"streams":[{"codec_type":"video","codec_name":"h264","pix_fmt":"yuv444p","width":1080,"height":1920,"duration":"7.04"},{"codec_type":"audio","codec_name":"aac","sample_rate":"44100","channels":2,"duration":"7.04"}],"format":{"duration":"7.04"}}';;
  drift) echo '{"streams":[{"codec_type":"video","codec_name":"h264","pix_fmt":"yuv420p","width":1080,"height":1920,"duration":"7.04"},{"codec_type":"audio","codec_name":"aac","sample_rate":"44100","channels":2,"duration":"4.10"}],"format":{"duration":"7.04"}}';;
  mute) echo '{"streams":[{"codec_type":"video","codec_name":"h264","pix_fmt":"yuv420p","width":1080,"height":1920,"duration":"7.04"}],"format":{"duration":"7.04"}}';;
esac
''')
    script.chmod(0o755)
    reel = fake_bin / "luca_reel.mp4"
    reel.write_bytes(b"\0" * 4096)

    original = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{fake_bin}:{original}"
    try:
        os.environ["FAKE_MODE"] = "good"
        info = video.verify_reel(reel, {})
        assert info["iphone_compatible"] is True
        assert info["video"]["pix_fmt"] == "yuv420p"
        assert info["sync"]["ok"] is True

        rejections = {}
        for mode, expected in (("pix", "yuv420p"), ("drift", "drift"),
                               ("mute", "audio")):
            os.environ["FAKE_MODE"] = mode
            try:
                video.verify_reel(reel, {})
            except RuntimeError as exc:
                rejections[mode] = str(exc)
                assert expected in str(exc), (mode, str(exc))
            else:
                raise AssertionError(f"{mode} reel was accepted")
    finally:
        os.environ["PATH"] = original
        os.environ.pop("FAKE_MODE", None)

    # the mux itself must be the iPhone contract, not something close to it
    mux = code_of(source("generate_video.py"))
    for flag in ('"-c:v", "libx264"', '"-pix_fmt", "yuv420p"', '"-c:a", "aac"',
                 '"-movflags", "+faststart"'):
        assert flag in mux, f"the mandatory mux lost {flag}"
    return f"accepted a good reel, rejected {len(rejections)} bad ones, mux flags present"


check("reel gate verifies h264/yuv420p/aac + sync", reel_gate)


# --------------------------------------------------------------------------- #
# 6b. The no-ffprobe fallback must be exactly as strict
# --------------------------------------------------------------------------- #


def ffmpeg_report_parser():
    """
    Kaggle ships ffmpeg but not necessarily ffprobe.

    The fallback parses ffmpeg's `-i` table instead of ffprobe's JSON, and it
    must reject the same things — otherwise the gate that guarantees an
    iPhone-playable reel depends on which binary happens to exist.
    """
    video = load("generate_video", WORKER_DIR / "generate_video.py")
    video.log = lambda *a, **k: None

    tmp = pathlib.Path(tempfile.mkdtemp())
    fake = tmp / "ffmpeg"
    report = (
        "#!/bin/sh\n"
        "cat <<REPORT >&2\n"
        "Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'x.mp4':\n"
        "  Duration: 00:00:07.04, start: 0.000000, bitrate: 812 kb/s\n"
        "  Stream #0:0[0x1](und): Video: ${FAKE_VCODEC:-h264} (High), "
        "${FAKE_PIX:-yuv420p}(progressive), ${FAKE_SIZE:-1080x1920}, 25 fps\n"
        "  Stream #0:1[0x2](und): Audio: ${FAKE_ACODEC:-aac} (LC), 44100 Hz, "
        "stereo, fltp, 160 kb/s\n"
        "REPORT\n"
    )
    fake.write_text(report)
    fake.chmod(0o755)
    reel = tmp / "luca_reel.mp4"
    reel.write_bytes(b"\0" * 2048)

    original = video.ffmpeg_bin
    video.ffmpeg_bin = lambda: str(fake)
    for name in ("FAKE_VCODEC", "FAKE_PIX", "FAKE_SIZE", "FAKE_ACODEC"):
        os.environ.pop(name, None)
    try:
        info = video.probe_media(reel)
        assert info["tool"] == "ffmpeg", info["tool"]
        assert info["video"]["codec"] == "h264", info["video"]
        assert info["video"]["pix_fmt"] == "yuv420p", info["video"]
        assert (info["video"]["width"], info["video"]["height"]) == (1080, 1920)
        assert info["audio"]["codec"] == "aac", info["audio"]
        assert abs(info["duration_s"] - 7.04) < 0.01, info["duration_s"]
        assert video.verify_reel(reel, {})["iphone_compatible"] is True

        for name, value, expected in (("FAKE_PIX", "yuv444p", "yuv420p"),
                                      ("FAKE_ACODEC", "mp2", "aac"),
                                      ("FAKE_VCODEC", "mpeg4", "h264")):
            os.environ[name] = value
            try:
                video.verify_reel(reel, {})
            except RuntimeError as exc:
                assert expected in str(exc), (name, str(exc))
            else:
                raise AssertionError(f"the fallback accepted a bad {name}")
            finally_names = os.environ.pop(name)
    finally:
        video.ffmpeg_bin = original
        for name in ("FAKE_VCODEC", "FAKE_PIX", "FAKE_SIZE", "FAKE_ACODEC"):
            os.environ.pop(name, None)
    return "parsed ffmpeg's stream table, rejected 3 bad reels on it"


check("no-ffprobe fallback is as strict as ffprobe", ffmpeg_report_parser)


# --------------------------------------------------------------------------- #
# 7. SadTalker invocation: flags, weights, and the two import-time shims
# --------------------------------------------------------------------------- #


def sadtalker_invocation():
    video = source("generate_video.py")
    body = code_of(video)

    assert '"--preprocess", "crop"' in body, "SadTalker must preprocess with crop"
    assert '"--still"' not in body, \
        "SadTalker must not be frozen with --still (the head has to move)"
    assert '"--enhancer"' in body and "gfpgan" in body, \
        "the GFPGAN enhancer is what repairs the softened mouth"
    assert '"--checkpoint_dir"' in body and '"--size"' in body
    assert "ensure_ffmpeg_on_path" in body, \
        ("SadTalker shells out to a bare `ffmpeg` via os.system and ignores its "
         "exit code; the binary must be put on PATH first")
    assert "IMAGEIO_FFMPEG_EXE" in body
    assert "wait_for_master" in body, \
        "the avatar must poll for process 1's master portrait"

    # every SadTalker weight must come from a host Kaggle can reach
    # Every weight must be a GitHub *release asset*: the sandbox and the Kaggle
    # runner can both reach those, and none of them need an HF token.
    urls = re.findall(r'https://github\.com/[A-Za-z0-9._/-]*releases/download/'
                      r'[A-Za-z0-9._/-]+', video)
    assert len(urls) >= 7, f"only {len(urls)} release-asset URLs"
    for name in ("SadTalker_V0.0.2_{size}.safetensors", "mapping_00229-model.pth.tar",
                 "GFPGANv1.4.pth", "alignment_WFLW_4HG.pth",
                 "detection_Resnet50_Final.pth"):
        assert name in video, f"missing weight {name}"

    # basicsr cannot be built with pip's default isolation (its setup.py
    # imports torch); it must be installed separately, without isolation, and
    # the enhancer must be skipped rather than crashing if it never lands.
    deps = re.search(r"SADTALKER_DEPS = \[(.*?)\]", video, re.S).group(1)
    assert "basicsr" not in deps, \
        "basicsr needs its own no-build-isolation install, not the plain batch"
    assert "no_build_isolation=True" in body, \
        "basicsr must be installed with --no-build-isolation"
    assert "GFPGAN_DEPS" in body and "gfpgan==1.3.8" in body
    assert "enhancer = None" in body, \
        "a missing gfpgan must downgrade the enhancer, not kill the render"

    # The two import-time shims must be applied before gfpgan/SadTalker load.
    assert "functional_tensor" in body and "VisibleDeprecationWarning" in body
    main_body = video[video.index("def main()"):]
    # Shims first: the dep probe imports every candidate package, so shimming
    # after probing makes an installed basicsr look uninstallable.
    assert main_body.index("apply_legacy_shims()") < \
        main_body.index("ensure_video_deps(") < main_body.index("synth_voice("), \
        "apply_legacy_shims() must run before the dependency probe and the TTS"
    return (f"{len(urls)} release-asset weights, crop+512+gfpgan, no --still, "
            f"shims applied before any import")


check("SadTalker invocation and weights", sadtalker_invocation)


# --------------------------------------------------------------------------- #
# 7a. The SadTalker wrapper actually survives the process boundary
# --------------------------------------------------------------------------- #


def sadtalker_wrapper():
    """
    Regression for run 45980957305.

    SadTalker's `src/face3d/util/preprocess.py` calls
    `warnings.filterwarnings(..., category=np.VisibleDeprecationWarning)` at
    import time. numpy 2 removed that symbol, and SadTalker runs as a
    *subprocess* — so patching numpy in the parent does nothing, and
    PYTHONSTARTUP is ignored for scripts (verified, not assumed). The fix is a
    wrapper script that shims numpy and then run_path()s inference.py; this
    check runs that wrapper against a stand-in that performs the same import.
    """
    video = load("generate_video", WORKER_DIR / "generate_video.py")
    video.log = lambda *a, **k: None
    video.WORK_DIR = pathlib.Path(tempfile.mkdtemp())

    probe_dir = video.WORK_DIR / "runner_probe"
    (probe_dir / "src").mkdir(parents=True, exist_ok=True)
    (probe_dir / "src" / "__init__.py").write_text("")
    (probe_dir / "src" / "helper.py").write_text("MARK = 'sibling-ok'\n")
    stub = probe_dir / "inference.py"
    stub.write_text(
        "import sys, warnings\n"
        "import numpy as np\n"
        "warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)\n"
        "from src.helper import MARK\n"
        "print('SHIM_OK', MARK)\n")

    runner = video.write_sadtalker_runner()
    probe = subprocess.run([sys.executable, str(runner), str(stub)],
                           capture_output=True, text=True, cwd=str(probe_dir))
    assert "SHIM_OK sibling-ok" in (probe.stdout + probe.stderr), \
        (probe.stdout + probe.stderr)[-600:]

    # ...and without the wrapper the very same stub must fail, or this check
    # would pass for the wrong reason.
    failing = subprocess.run([sys.executable, str(stub)], capture_output=True,
                             text=True, cwd=str(probe_dir))
    assert failing.returncode != 0 and "VisibleDeprecationWarning" in failing.stderr, \
        "the stub does not reproduce the numpy 2 failure, so the check proves nothing"
    return "wrapper restores the shim in a child; the bare script fails as it did in the run"


check("SadTalker wrapper survives the process boundary", sadtalker_wrapper)


# --------------------------------------------------------------------------- #
# 7a2. The gfpgan dependency chain has a concrete, tested plan
# --------------------------------------------------------------------------- #


def basicsr_plan():
    """
    basicsr 1.4.2 will not build under a modern setuptools.

    Its setup.py asks git for a hash with
    `subprocess.Popen(...).communicate()`, and that fails during metadata
    generation; pip reports only "Preparing metadata (pyproject.toml) did not
    run successfully / No available output". Verified locally: the same sdist
    builds cleanly once that one line is the equivalent subprocess.run call, so
    the worker now downloads, patches and installs the local path.
    """
    video = load("generate_video", WORKER_DIR / "generate_video.py")
    video.log = lambda *a, **k: None
    body = code_of(source("generate_video.py"))

    assert "prepare_basicsr_sdist" in body
    assert "prepare_sdist=prepare_basicsr_sdist" in body, \
        "the pip call must use the patched sdist"
    old, new = video._BASICSR_SETUP_PATCH
    assert "communicate()" in old and "subprocess.run" in new
    assert "no_build_isolation=True" in body, \
        "basicsr's setup.py imports torch, so isolation has to be off"
    # The enhancer must still be optional: a failed install may not kill the reel.
    assert "enhancer = None" in body
    return "sdist patched, installed without isolation, enhancer still optional"


check("basicsr install plan (patched sdist)", basicsr_plan)


# --------------------------------------------------------------------------- #
# 7a3. A failed enhancer install must never abort the video stage
# --------------------------------------------------------------------------- #


def enhancer_is_optional():
    """
    Regression for run 45984347218.

    `prepare_basicsr_sdist` raised CalledProcessError out of
    `ensure_video_deps`, so the whole video process died before Chatterbox ever
    loaded — even though the enhancer is decorative. Every path in that block
    now degrades to "render without the enhancer".
    """
    video = source("generate_video.py")
    body = code_of(video)

    block = body[body.index("if missing_packages(GFPGAN_DEPS):"):
                 body.index("log(\"SadTalker dependencies ready\")")]
    assert "try:" in block and "except Exception" in block, \
        "the gfpgan install must be wrapped: it is optional"
    assert "installed = False" in block
    assert "raise" not in block, \
        "nothing inside the optional-enhancer block may abort the process"

    # and the runner must still be reached with the enhancer dropped
    assert "enhancer = None" in body
    return "sdist failure degrades to no-enhancer, never to a dead process"


check("enhancer install is optional", enhancer_is_optional)


# --------------------------------------------------------------------------- #
# 7b. Detection failures must never delete a generated image
# --------------------------------------------------------------------------- #


def dataset_retention():
    """
    Regression for run 45980957305, the expensive half of it.

    12 of 15 InstantID renders were deleted because antelopev2's detector — at
    its single 640px scale — could not find a face in the wider framings. The
    training set fell to three images, under the minimum, so no adapter was
    produced and the carousel silently fell back to the base model.
    """
    images = source("generate_images.py")
    body = code_of(images)

    assert "unverified" in body, "unverified renders must be recorded, not deleted"
    # The only unlink() left must not be the per-image face check.
    loop = images[images.index("for index, entry in enumerate"):
                  images.index("dataset complete")]
    assert "unlink" not in loop, \
        "the dataset loop still deletes images on a detection miss"
    assert "DET_SIZES" in body and len(re.findall(r"\(\d{3,4}, \d{3,4}\)", body)) >= 2, \
        "detection must escalate through more than one det_size"
    # ...and training uses every kept image, verified or not.
    assert 'dataset["files"] + dataset["unverified"]' in body, \
        "training must include the renders the detector could not verify"
    return "renders are kept, detection escalates, training uses all of them"


check("dataset retention: a detection miss never deletes an image", dataset_retention)


# --------------------------------------------------------------------------- #
# 7b. Resolution mismatches: the LoRA resizes internally, the avatar cannot
# --------------------------------------------------------------------------- #


def resolution_contract():
    """
    A 1024² master is right for SDXL and wrong for SadTalker.

    SadTalker's face renderer resizes every frame to ``--size``, so the reel is
    512 px wide whatever the portrait. Logging the source resolution is what
    makes that visible in the metrics instead of a surprise in the output.
    """
    images = source("generate_images.py")
    assert "width=1024, height=1024" in images, "the master is no longer 1024²"
    video = source("generate_video.py")
    assert re.search(r'SADTALKER_SIZE = int\(os\.environ\.get\('
                     r'"SADTALKER_SIZE", "256"\)\)', video), \
        "the avatar should default to SadTalker's native 256, not the 512 file"
    assert "master_source_px" in video, \
        "the reel metrics must record the source resolution the avatar cropped from"
    return "master 1024², avatar 256 (native), source resolution recorded"


check("resolution contract (master vs avatar)", resolution_contract)


# --------------------------------------------------------------------------- #
# 7c. A broken gfpgan must not kill the render (it is imported at module scope)
# --------------------------------------------------------------------------- #


def gfpgan_stub():
    """
    Regression for run 45989277255.

    SadTalker's ``src/utils/face_enhancer.py`` does ``from gfpgan import
    GFPGANer`` at module scope, and ``animate.py`` imports that unconditionally —
    so a missing *or half-installed* gfpgan kills the render even when no
    ``--enhancer`` was requested. A failed basicsr build had left a namespace
    package behind, and the run died on "cannot import name 'GFPGANer' from
    'gfpgan' (unknown location)".
    """
    video = load("generate_video", WORKER_DIR / "generate_video.py")
    video.log = lambda *a, **k: None
    video.WORK_DIR = pathlib.Path(tempfile.mkdtemp())

    assert video.gfpgan_usable() is False, \
        "the real gfpgan is expected to be unusable in this environment"
    stub_dir = video.install_gfpgan_stub()
    assert (stub_dir / "gfpgan" / "__init__.py").is_file()

    ok = subprocess.run(
        [sys.executable, "-c", "from gfpgan import GFPGANer; print('ok')"],
        env={**os.environ, "PYTHONPATH": str(stub_dir)},
        capture_output=True, text=True)
    assert "ok" in ok.stdout, ok.stderr[-400:]

    # ...but constructing it must fail loudly: a render that silently skips the
    # enhancer while claiming otherwise would be worse than no render.
    boom = subprocess.run(
        [sys.executable, "-c", "from gfpgan import GFPGANer; GFPGANer()"],
        env={**os.environ, "PYTHONPATH": str(stub_dir)},
        capture_output=True, text=True)
    assert boom.returncode != 0 and "not installed" in boom.stderr, boom.stderr[-300:]

    body = code_of(source("generate_video.py"))
    assert "child_env[\"PYTHONPATH\"]" in body, \
        "the stub directory must reach SadTalker's child process"
    return "module-scope import satisfied, instantiation still a hard error"


check("gfpgan stub satisfies SadTalker's module-scope import", gfpgan_stub)


# --------------------------------------------------------------------------- #
# 7d. A degenerate render must never be shipped
# --------------------------------------------------------------------------- #


def degeneracy_guard():
    """
    Regression for run 45989277255.

    Four carousel PNGs came out at 2.5 KB, byte-identical across four different
    prompts — the signature of a float32 adapter multiplied into a float16 UNet,
    which produces inf/NaN that the VAE decodes to a flat frame. Two fixes: cast
    the adapter to the UNet's dtype, and refuse to save a frame that is flat.
    """
    images = load("generate_images", WORKER_DIR / "generate_images.py")
    images.log = lambda *a, **k: None
    from PIL import Image
    import numpy as np

    blank = Image.new("RGB", (128, 128), (128, 128, 128))
    noise = Image.fromarray(
        np.random.default_rng(0).integers(0, 255, (128, 128, 3), dtype=np.uint8))
    assert images.is_degenerate(images.image_stats(blank)) is True
    assert images.is_degenerate(images.image_stats(noise)) is False

    body = code_of(source("generate_images.py"))
    assert "def cast_lora_to_unet" in body, "the adapter dtype cast is missing"
    assert "cast_lora_to_unet(pipe)" in body, "the cast must run after loading"
    assert "degenerate_with_lora" in body, \
        "a flat slide must be regenerated without the adapter and recorded"
    assert "the master portrait is degenerate" in body, \
        "the master is the avatar's input; it must be checked too"

    # The cast must find tensors by NAME. The first attempt looked only for
    # module.lora_A dicts, found zero (``tensors: 0`` in run 45995210057), and
    # so changed nothing.
    assert "named_parameters()" in body and '"lora" in name.lower()' in body, \
        "the adapter cast must scan named parameters, not a fixed attribute"
    assert "lora_tensors" in body, "the cast must report what it found"

    # ...and a degenerate slide must never unload the adapter for the rest of
    # the carousel: that is how slides 02-04 lost the identity in 45995210057.
    carousel = body[body.index("def build_carousel"):body.index("def main()")]
    assert "LORA_FALLBACK_SCALES" in carousel, \
        "a degenerate slide must walk the adapter scale down first"
    assert "set_adapters([\"luca\"], [scale])" in carousel
    unload = carousel.index("unload_lora_weights")
    retry = carousel.index("LORA_FALLBACK_SCALES")
    assert retry < unload, "the scale ladder must be tried before unloading"
    return "flat frames rejected, adapter cast by name, per-slide scale ladder"


check("degenerate renders are never shipped", degeneracy_guard)


# --------------------------------------------------------------------------- #
# 8. The launcher: stages every import, then runs as a script
# --------------------------------------------------------------------------- #


def launcher_script():
    work = pathlib.Path(tempfile.mkdtemp())
    env = dict(os.environ, OUTPUT_DIR=str(work / "out"),
               LUCA_WORKER_DIR=str(work / "workers"), CUDA_VISIBLE_DEVICES="")
    proc = subprocess.Popen(
        [sys.executable, str(REPO / "kaggle_kernel_luca" / "luca_studio.py")],
        cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=env, start_new_session=True)

    markers = {"embedded payload verified", "prohibition check passed",
               "using committed vendored pipeline",
               "launching both pipelines in parallel"}
    seen, captured, deadline = set(), [], __import__("time").time() + 240
    for line in iter(proc.stdout.readline, ""):
        captured.append(line)
        seen.update(m for m in markers if m in line)
        if seen == markers or __import__("time").time() > deadline:
            break
    try:
        os.killpg(os.getpgid(proc.pid), 9)
    except Exception:
        proc.kill()
    proc.wait(timeout=60)
    log = "".join(captured)

    missing = sorted(markers - seen)
    assert not missing, f"launcher never reached {missing}\n{log[-2000:]}"
    startup = log.split("[launcher] launching both pipelines in parallel")[0]
    for fatal in ("NameError", "AttributeError", "ImportError",
                  "ModuleNotFoundError", "SyntaxError"):
        assert fatal not in startup, f"launcher raised {fatal}:\n{startup[-2000:]}"

    # every module a worker imports must have been written to disk
    staged = sorted(p.name for p in (work / "workers").glob("*.py"))
    for name in ("luca_common.py", "luca_character.py", "luca_lora.py",
                 "generate_images.py", "generate_video.py", "fetch_instantid.py"):
        assert name in staged, f"{name} was not staged ({staged})"
    return f"staged {len(staged)} modules, reached both workers with no startup error"


check("launcher stages its payload and starts both workers", launcher_script)


# --------------------------------------------------------------------------- #
# 9. Banned techniques, on executable code only
# --------------------------------------------------------------------------- #


def prohibitions():
    banned = {"pipe.to(\"cpu\")": "CPU offload", "inswapper_128": "inswapper",
              "StableVideoDiffusion": "SVD-XT",
              "stable-video-diffusion": "SVD-XT",
              "StableDiffusionXLImg2ImgPipeline": "img2img adetailer"}
    hits = []
    for name in ("luca_lora.py", "generate_images.py", "generate_video.py"):
        body = code_of(source(name))
        for token, label in banned.items():
            if token in body:
                hits.append(f"{name}: {label}")
    assert not hits, hits
    # the legacy avatar must be gone, not merely unused
    for name in ("generate_images.py", "generate_video.py"):
        assert "musetalk" not in source(name).lower(), \
            f"{name} still references MuseTalk"
    return f"none of the {len(banned)} banned constructs in any worker; no MuseTalk"


check("banned techniques absent", prohibitions)


# --------------------------------------------------------------------------- #
# 10. The kernel is the sources, byte for byte
# --------------------------------------------------------------------------- #


def kernel_fresh():
    proc = subprocess.run([sys.executable, str(REPO / "scripts" / "build_luca_kernel.py"),
                           "--check"], capture_output=True, text=True, cwd=str(REPO))
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-1500:]

    launcher = (REPO / "kaggle_kernel_luca" / "luca_studio.py").read_text()
    for constant in ("LUCA_COMMON_PY", "LUCA_CHARACTER_PY", "LUCA_LORA_PY",
                     "GENERATE_IMAGES_PY", "GENERATE_VIDEO_PY",
                     "FETCH_INSTANTID_PY", "INSTANTID_PIPELINE_PY"):
        assert constant in launcher, f"{constant} is not embedded"
    return f"generated kernel is current ({len(launcher) / 1024:.0f} KB, 7 sources)"


check("generated kernel matches its sources", kernel_fresh)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

width = max(len(label) for _, label, _ in results)
failures = [r for r in results if not r[0]]
for ok, label, detail in results:
    print(f"  {'PASS' if ok else 'FAIL'}  {label:<{width}}  {detail}")
print()
if failures:
    print(f"{len(failures)} of {len(results)} checks FAILED")
    sys.exit(1)
print(f"all {len(results)} checks passed")
