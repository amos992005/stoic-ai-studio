#!/usr/bin/env python3
"""
Verification for the Luca v2 build, against the REAL libraries where possible.

This replaces the earlier hand-stubbed simulations: now that torch, diffusers,
compel, insightface and opencv are installable here, the real classes exercise
the real call sites. Each block prints PASS/FAIL and the suite exits non-zero on
any failure, so it can gate a commit.
"""
import inspect
import pathlib
import subprocess
import sys
import tempfile
import types

REPO = pathlib.Path("/home/user/stoic-ai-studio")
sys.path.insert(0, str(REPO / "scripts" / "luca_v2"))
sys.path.insert(0, str(REPO / "kaggle_kernel_luca"))

import torch

# NOTE: real torch, real CUDA surface (there is simply no GPU here, so
# torch.cuda.is_available() is False). Patching torch.cuda with a stub breaks
# torch's own device plumbing, which is exactly what the earlier sims did.

results = []
def check(label, fn):
    try:
        detail = fn()
        results.append((True, label, detail or ""))
    except Exception as exc:
        results.append((False, label, f"{type(exc).__name__}: {exc}"))

# --------------------------------------------------------------------------- #
# 1. the vendored InstantID pipeline, against real diffusers
# --------------------------------------------------------------------------- #
def vendored_pipeline():
    spec = __import__("importlib.util", fromlist=["x"]).spec_from_file_location(
        "iid", REPO / "scripts/luca_v2/_vendor/pipeline_stable_diffusion_xl_instantid.py")
    m = __import__("importlib.util", fromlist=["x"]).module_from_spec(spec)
    spec.loader.exec_module(m)
    P = m.StableDiffusionXLInstantIDPipeline
    assert not hasattr(P, "enable_vae_slicing"), "enable_vae_slicing exists again"
    assert hasattr(P, "set_ip_adapter_scale")
    assert hasattr(m, "draw_kps")
    params = inspect.signature(P.__call__).parameters
    assert "ip_adapter_scale" not in params, "ip_adapter_scale is a real param now"
    assert "controlnet_conditioning_scale" in params
    return f"{len(params)} __call__ params, no pipeline-level VAE slicing"

check("vendored pipeline imports and has the expected surface", vendored_pipeline)

# --------------------------------------------------------------------------- #
# 2. compel returns the four SDXL tensors our encode_prompts reads
# --------------------------------------------------------------------------- #
def compel_contract():
    from compel.convenience_wrappers import LabelledConditioning
    fields = set(LabelledConditioning.__dataclass_fields__)
    required = {"embeds", "pooled_embeds", "negative_embeds", "negative_pooled_embeds"}
    assert required <= fields, f"compel renamed fields: {required - fields}"
    # the names the worker reads
    src = (REPO / "scripts/luca_v2/generate_images.py").read_text()
    for attr in ("embeds", "pooled_embeds", "negative_embeds", "negative_pooled_embeds"):
        assert f"conditioning.{attr}" in src, f"worker does not read {attr}"
    return "worker reads the attributes compel actually returns"

check("compel LabelledConditioning contract", compel_contract)

# --------------------------------------------------------------------------- #
# 3. VAE slicing: works on a plain VAE, degrades gracefully without one
# --------------------------------------------------------------------------- #
def vae_slicing():
    import importlib.util
    spec = importlib.util.spec_from_file_location("gi", REPO / "scripts/luca_v2/generate_images.py")
    gi = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(gi)
    except Exception:
        pass                      # heavy deps absent here; we only need the helper
    gi.log = lambda *a, **k: None

    from diffusers import AutoencoderKL
    vae = AutoencoderKL(in_channels=4, out_channels=4,
                        down_block_types=("DownBlock2D", "DownBlock2D"),
                        up_block_types=("UpBlock2D", "UpBlock2D"),
                        block_out_channels=(32, 64), layers_per_block=2,
                        latent_channels=4, sample_size=32, norm_num_groups=32)
    assert vae.use_slicing is False, "slicing on by default?"
    # the pipeline object the worker actually holds: a namespace with a .vae
    assert gi.enable_vae_slicing(types.SimpleNamespace(vae=vae)) == "vae.enable_slicing()"
    assert vae.use_slicing is True
    # the fallback, for a VAE whose diffusers build has no enable_slicing()
    class Bare:
        use_slicing = False
    assert gi.enable_vae_slicing(types.SimpleNamespace(vae=Bare())) == "vae.use_slicing=True"
    # and a pipeline with no VAE at all must not raise
    assert gi.enable_vae_slicing(types.SimpleNamespace()) == "off"
    return "slices via the VAE, falls back, tolerates a missing VAE"

check("VAE slicing helper", vae_slicing)

# --------------------------------------------------------------------------- #
# 4. the master pipeline shares the SDXL weights and fetches nothing
# --------------------------------------------------------------------------- #
def master_pipe_shares():
    """
    Regression for run 45952128760: StableDiffusionXLPipeline.from_pipe()
    silently re-downloads optional components the source pipeline lacks
    (image_encoder, feature_extractor) in float32, which doubled VRAM to
    14.4 GiB and OOM'd. Building from the live components must share, not copy.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("gi", REPO / "scripts/luca_v2/generate_images.py")
    gi = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(gi)
    except Exception:
        pass
    gi.log = lambda *a, **k: None
    gi.DEVICE = "cpu"

    from diffusers import AutoencoderKL, UNet2DConditionModel
    from diffusers.schedulers import DPMSolverMultistepScheduler
    vae = AutoencoderKL(in_channels=4, out_channels=4,
                        down_block_types=("DownBlock2D", "DownBlock2D"),
                        up_block_types=("UpBlock2D", "UpBlock2D"),
                        block_out_channels=(32, 64), layers_per_block=2,
                        latent_channels=4, sample_size=32, norm_num_groups=32)
    unet = UNet2DConditionModel(sample_size=32, in_channels=4, out_channels=4,
                                layers_per_block=1, block_out_channels=(32, 64),
                                down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                                up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
                                cross_attention_dim=32, norm_num_groups=32,
                                attention_head_dim=8)
    scheduler = DPMSolverMultistepScheduler()

    class Fake(torch.nn.Module):        # a stand-in for a transformers encoder
        dtype = torch.float32
    Fake.__module__ = "transformers"

    comps = {"vae": vae, "unet": unet, "scheduler": scheduler,
             "text_encoder": Fake(), "text_encoder_2": Fake(),
             "tokenizer": Fake(), "tokenizer_2": Fake(),
             "controlnet": object(), "image_proj_model": object(),
             "ip_adapter": object()}
    master = gi.build_master_pipe(types.SimpleNamespace(components=dict(comps)), {})

    assert master.unet is unet and master.vae is vae, "weights were copied, not shared"
    assert master.text_encoder_2 is comps["text_encoder_2"]
    assert master.scheduler is scheduler
    assert master.image_encoder is None, "an image_encoder was fetched"
    assert master.feature_extractor is None, "a feature_extractor was fetched"
    assert not hasattr(master, "controlnet"), "controlnet leaked into the master pipeline"
    return "shares unet/vae/encoders, fetches no optional component"

check("master pipeline shares weights instead of refetching", master_pipe_shares)

# --------------------------------------------------------------------------- #
# 4b. the memory controls are in place and harmless
# --------------------------------------------------------------------------- #
def memory_controls():
    import importlib.util
    spec = importlib.util.spec_from_file_location("gi", REPO / "scripts/luca_v2/generate_images.py")
    gi = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(gi)
    except Exception:
        pass
    gi.log = lambda *a, **k: None

    # set at import time: the allocator reads it when CUDA first initialises
    src = (REPO / "scripts/luca_v2/generate_images.py").read_text()
    before_models = src.split("def load_pipeline")[0]
    assert "PYTORCH_CUDA_ALLOC_CONF" in before_models, \
        "alloc conf is set too late to take effect"

    limits = gi.configure_memory_limits()
    assert "expandable_segments" in limits["alloc_conf"], limits
    # attention slicing: applied when available, skipped otherwise
    applied = {}

    class WithHelper:
        def enable_attention_slicing(self, *a, **k):
            applied["sliced"] = True
    assert gi.enable_attention_slicing(WithHelper()) is True and applied["sliced"]
    assert gi.enable_attention_slicing(types.SimpleNamespace()) is False
    return f"allocator {limits['alloc_conf']}, tf32={'on' if limits['tf32'] else 'off (no GPU here)'}"

check("memory controls", memory_controls)

# --------------------------------------------------------------------------- #
# 4c. the identity projector is moved with the rest of the pipeline
# --------------------------------------------------------------------------- #
def projector_moved():
    """
    Regression for run 45954696729: load_ip_adapter_instantid() creates
    image_proj_model after __init__, so it is not in pipe.components and
    pipe.to(device) leaves it on CPU. Every scene then dies on a device
    mismatch; only the master (which skips the identity branch) survives.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("gi", REPO / "scripts/luca_v2/generate_images.py")
    gi = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(gi)
    except Exception:
        pass
    gi.log = lambda *a, **k: None
    gi.DEVICE = "cpu"

    import torch
    class Resampler(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.proj = torch.nn.Linear(4, 4)
    class FakeUNet(torch.nn.Module):
        """Real diffusers models expose .dtype; nn.Linear does not."""
        dtype = torch.float16
    class Pipe:
        def __init__(self):
            self.unet = FakeUNet()                     # the dtype source
            self.image_proj_model = Resampler()        # still float32

    pipe = Pipe()
    assert pipe.image_proj_model.proj.weight.dtype is torch.float32
    assert gi.move_identity_projector(pipe) != "absent"
    assert pipe.image_proj_model.proj.weight.dtype is torch.float16, \
        "the projector was not matched to the unet dtype"
    # a pipeline without one must not raise
    assert gi.move_identity_projector(types.SimpleNamespace()) == "absent"
    return "projector follows the unet's device and dtype"

check("identity projector is moved", projector_moved)

# --------------------------------------------------------------------------- #
# 5. chatterbox precision: s3gen stays float32 (its STFT has no fp16 path)
# --------------------------------------------------------------------------- #
def chatterbox_precision():
    """
    Regression for runs 45950802624 and 45952128760: casting s3gen to fp16 makes
    its float32 STFT output collide with fp16 conv weights. Only the
    autoregressive t3 and the voice encoder may be cast.
    """
    import re
    src = (REPO / "scripts/luca_v2/generate_video.py").read_text()
    m = re.search(r"FP16_SUBMODULES = \(([^)]*)\)", src)
    assert m, "FP16_SUBMODULES disappeared"
    names = re.findall(r'"(\w+)"', m.group(1))
    assert "s3gen" not in names, f"s3gen is cast to fp16 again: {names}"
    assert "t3" in names, f"t3 should still be cast: {names}"
    # the s3gen vocoder really is where the problem is: show it concretely
    import torch
    import torch.nn as nn
    class Voc(nn.Module):                       # mimics the conv after the STFT
        def __init__(self):
            super().__init__(); self.conv = nn.Conv1d(4, 4, 3)
    voc = Voc().half().eval()
    stft_out = torch.stft(torch.randn(4, 64), n_fft=8, return_complex=True)  # float32
    try:
        voc.conv(stft_out.real.float())
        raise AssertionError("expected the dtype error")
    except RuntimeError as exc:
        assert "should be the same" in str(exc)
    return f"fp16 set is {names}; s3gen float32 because torch.stft emits float32"

check("chatterbox keeps s3gen in float32", chatterbox_precision)

# --------------------------------------------------------------------------- #
# 5b. the vendored pipeline calls the parent's check_inputs by keyword only
# --------------------------------------------------------------------------- #
def check_inputs_keywords():
    """
    Regression for run 45953039196. diffusers 0.43 inserted ip_adapter_image and
    ip_adapter_image_embeds into the parent's check_inputs signature, so the
    vendored positional call shifted every later argument and
    controlnet_conditioning_scale arrived holding a tensor. Keywords are stable
    across versions; positional arguments are not.
    """
    import ast
    import inspect

    tree = ast.parse((REPO / "scripts/luca_v2/_vendor/pipeline_stable_diffusion_xl_instantid.py").read_text())
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and node.func.attr == "check_inputs"]
    assert len(calls) == 1, f"expected one check_inputs call, found {len(calls)}"
    call = calls[0]
    assert not call.args, f"{len(call.args)} positional argument(s) — these drift across releases"
    passed = {kw.arg for kw in call.keywords}

    # and they must still be acceptable to whatever diffusers is installed
    from diffusers.pipelines.controlnet.pipeline_controlnet_sd_xl import (
        StableDiffusionXLControlNetPipeline as Parent)
    params = set(inspect.signature(Parent.check_inputs).parameters) - {"self"}
    unknown = passed - params
    assert not unknown, f"diffusers dropped {unknown}"
    required = {n for n, p_ in inspect.signature(Parent.check_inputs).parameters.items()
                if p_.default is p_.empty} - {"self"}
    assert required <= passed, f"diffusers added required {required - passed}"
    return f"{len(passed)} keyword args, no positional, bindable on diffusers {__import__('diffusers').__version__}"

check("vendored check_inputs is called by keyword", check_inputs_keywords)

# --------------------------------------------------------------------------- #
# 6. verify_reel rejects bad reels and accepts a good one
# --------------------------------------------------------------------------- #
def reel_gate():
    import importlib.util
    spec = importlib.util.spec_from_file_location("gv", REPO / "scripts/luca_v2/generate_video.py")
    gv = importlib.util.module_from_spec(spec); spec.loader.exec_module(gv)
    gv.log = lambda *a, **k: None
    real_probe = gv.probe_media
    gv.probe_media = lambda path: {
        "tool": "ffprobe",
        "video": {"codec": "h264", "width": 832, "height": 1024,
                  "duration_s": 8.02, "fps": 25.0},
        "audio": {"codec": "aac", "sample_rate": 24000, "channels": 1,
                  "duration_s": 8.01},
        "duration_s": 8.02}
    try:
        f = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False); f.write(b"x"); f.close()
        good = gv.verify_reel(pathlib.Path(f.name), {})
        assert good["sync"]["ok"]
        gv.probe_media = lambda path: {
            "tool": "ffprobe",
            "video": {"codec": "h264", "width": 832, "height": 1024,
                      "duration_s": 8.02, "fps": 25.0},
            "audio": None, "duration_s": 8.02}
        try:
            gv.verify_reel(pathlib.Path(f.name), {}); raise AssertionError("no-audio not rejected")
        except RuntimeError:
            pass
        gv.probe_media = lambda path: {
            "tool": "ffprobe",
            "video": {"codec": "h264", "width": 832, "height": 1024,
                      "duration_s": 8.0, "fps": 25.0},
            "audio": {"codec": "aac", "duration_s": 3.0}, "duration_s": 8.0}
        try:
            gv.verify_reel(pathlib.Path(f.name), {}); raise AssertionError("drift not rejected")
        except RuntimeError:
            pass
    finally:
        gv.probe_media = real_probe
    return "accepts a synced reel, rejects missing audio and 5s drift"

check("ffprobe gate", reel_gate)

# --------------------------------------------------------------------------- #
# 7. the launcher, run as a script exactly like Kaggle
# --------------------------------------------------------------------------- #
def launcher_script():
    work = tempfile.mkdtemp()
    env = {"PATH": "/usr/bin:/bin", "OUTPUT_DIR": work,
           "LUCA_WORKER_DIR": f"{work}/luca_workers", "CUDA_VISIBLE_DEVICES": "",
           "HOME": work}
    proc = subprocess.run(
        [sys.executable, str(REPO / "kaggle_kernel_luca/luca_studio.py")],
        capture_output=True, text=True, env=env, timeout=900)
    log = (proc.stdout or "") + (proc.stderr or "")
    startup = log.split("[launcher] launching both pipelines in parallel")[0]
    for marker in ("embedded payload verified", "prohibition check passed",
                   "using committed vendored pipeline"):
        assert marker in startup, f"missing {marker}\n{startup[-1500:]}"
    for fatal in ("NameError", "ImportError", "AttributeError", "SyntaxError"):
        assert fatal not in startup, f"{fatal} during startup\n{startup[-1500:]}"
    staged = sorted(p.name for p in pathlib.Path(f"{work}/luca_workers").iterdir())
    assert "generate_images.py" in staged and "_vendor" in staged, staged
    return f"staged {staged}"

check("kernel runs as a script and stages its payload", launcher_script)

# --------------------------------------------------------------------------- #
# 8. the generated kernel is current
# --------------------------------------------------------------------------- #
def kernel_fresh():
    proc = subprocess.run([sys.executable, str(REPO / "scripts/build_luca_kernel.py"),
                           "--check"], capture_output=True, text=True)
    assert proc.returncode == 0, (proc.stdout + proc.stderr).strip()
    return proc.stdout.strip()

check("generated kernel matches its sources", kernel_fresh)

# --------------------------------------------------------------------------- #
print()
failed = 0
for ok, label, detail in results:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if detail:
        print(f"        {detail}")
    failed += not ok
print(f"\n{len(results) - failed}/{len(results)} checks passed")
sys.exit(1 if failed else 0)
