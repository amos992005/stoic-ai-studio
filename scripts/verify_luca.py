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
# 4. from_pipe shares components (no second SDXL copy on a 14.5 GB card)
# --------------------------------------------------------------------------- #
def from_pipe_shares():
    from diffusers import StableDiffusionXLPipeline, AutoencoderKL, UNet2DConditionModel
    vae = AutoencoderKL(in_channels=4, out_channels=4, down_block_types=("DownBlock2D",),
                        up_block_types=("UpBlock2D",), block_out_channels=(32,),
                        layers_per_block=1, latent_channels=4, sample_size=32,
                        norm_num_groups=32)
    unet = UNet2DConditionModel(sample_size=32, in_channels=4, out_channels=4,
                                layers_per_block=1, block_out_channels=(32, 64),
                                down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                                up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
                                cross_attention_dim=32, norm_num_groups=32,
                                attention_head_dim=8)
    base = StableDiffusionXLPipeline(vae=vae, text_encoder=None, tokenizer=None,
                                     unet=unet, scheduler=None,
                                     text_encoder_2=None, tokenizer_2=None)
    plain = StableDiffusionXLPipeline.from_pipe(base)
    assert plain.unet is base.unet and plain.vae is base.vae, "components were copied"
    assert "image" not in inspect.signature(plain.__call__).parameters
    return "unet/vae shared, and the master pipeline needs no control image"

check("from_pipe shares weights and drops the image requirement", from_pipe_shares)

# --------------------------------------------------------------------------- #
# 5. the HiFT dtype wrapper is number-correct
# --------------------------------------------------------------------------- #
def hift_wrapper():
    import importlib.util
    spec = importlib.util.spec_from_file_location("gv", REPO / "scripts/luca_v2/generate_video.py")
    gv = importlib.util.module_from_spec(spec); spec.loader.exec_module(gv)
    gv.log = lambda *a, **k: None

    import torch.nn as nn
    class Source(nn.Module):
        def __init__(self):
            super().__init__(); self.l_linear = nn.Linear(9, 1); self.l_tanh = nn.Tanh()
        def forward(self, x):
            return self.l_tanh(self.l_linear(x.float().unsqueeze(-1).expand(-1, -1, -1, 9)))
    class S3(nn.Module):
        def __init__(self):
            super().__init__(); self.m_source = Source()

    gen = S3().half().eval()
    x = torch.randn(2, 16, 1).half()
    try:
        gen.m_source(x)
        raise AssertionError("expected a dtype error before wrapping")
    except RuntimeError as exc:
        assert "same dtype" in str(exc), exc
    assert gv.make_dtype_safe(gen) == 1
    y = gen.m_source(x)
    ref = S3().eval()
    ref.m_source.l_linear.weight.data = gen.m_source.l_linear.weight.data.float()
    ref.m_source.l_linear.bias.data = gen.m_source.l_linear.bias.data.float()
    assert torch.allclose(y.float(), ref.m_source(x.float()), atol=2e-2)
    assert gv.make_dtype_safe(gen) == 0, "wrapper is not idempotent"
    return "fp16 output matches the fp32 reference"

check("HiFT dtype wrapper fixes the crash numerically", hift_wrapper)

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
