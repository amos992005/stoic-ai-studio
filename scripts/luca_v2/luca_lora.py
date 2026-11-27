"""
Fast SDXL LoRA training on a 15-image identity dataset.

Deliberately small and hand-rolled rather than a wrapper around `train_text_to_image_lora_sdxl.py`:
that script assumes a dataset directory, an accelerate launcher and a full
validation pass, none of which apply to 15 images on a 14.5 GB T4.

What this is
------------
A peft adapter (r=16, lora_alpha=16) on the UNet's attention projections,
trained in float16 with gradient checkpointing at 768px. The text encoders are
frozen and pre-encoded once per image, so the only forward pass per step is the
UNet. At batch 1 and 400 steps that is minutes, not hours.

Design notes that matter
------------------------
* ``latents are cached``: the VAE is run once per image up front and then
  released, so it never competes with the UNet for VRAM during training.
* ``text embeds are cached``: SDXL's two encoders are 2.5 GB and would otherwise
  sit resident for a stage that never updates them.
* ``the attention mask is omitted``, matching diffusers' own ``encode_prompt``.
  Passing one is what triggers the classic "all-zero embeddings for one batch"
  bug in every published SDXL LoRA trainer.
* ``no PEFT wrapper on the whole UNet``: only the adapter tensors get optimiser
  state, which is what keeps AdamW's footprint at tens of MB instead of GBs.
"""

from __future__ import annotations

import math
import time

import torch.nn as torch_nn
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from luca_common import log, timed

#: Attention projections every SDXL LoRA targets. These are in the
#: transformer blocks only — the resnet convs carry little identity information
#: and quadrupling the adapter size for them is a bad trade on a T4.
LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]


class _UnwrapDownBlock(torch_nn.Module):
    """
    Make an encoder down block return only its hidden states.

    diffusers' ``DownBlock2D.forward`` returns ``(hidden_states, output_states)``,
    and ``Encoder.forward`` assigns that tuple straight to ``sample``:

        for down_block in self.down_blocks:
            sample = down_block(sample)

    so the next block receives a tuple and ``group_norm`` raises
    "'tuple' object has no attribute 'dim'". This only bites the *encode* path —
    a pipeline that generates never calls it, which is why the previous pipeline
    on this repo never hit it, and why a LoRA trainer that caches latents hits it
    immediately. Unwrapping here is version-agnostic: a block that returns a
    plain tensor passes through untouched.
    """

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, hidden_states, *args, **kwargs):
        out = self.inner(hidden_states, *args, **kwargs)
        return out[0] if isinstance(out, tuple) else out


def patch_vae_encoder(vae) -> int:
    """Wrap every encoder down block; returns how many were wrapped."""
    count = 0
    for index, block in enumerate(list(vae.encoder.down_blocks)):
        if not isinstance(block, _UnwrapDownBlock):
            vae.encoder.down_blocks[index] = _UnwrapDownBlock(block)
            count += 1
    if count:
        log(f"encoder down blocks unwrapped for the encode path ({count})")
    return count


def build_adapter(unet, rank: int = 16, alpha: int = 16):
    """Attach a peft LoRA to the UNet's attention projections."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights="gaussian",     # non-zero B: the adapter moves from step 0
        target_modules=LORA_TARGETS,
    )
    model = get_peft_model(unet, config)
    trainable = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in trainable)
    log(f"LoRA attached: r={rank} alpha={alpha} · {total:,} trainable "
        f"parameters · targets={LORA_TARGETS}")
    return model, trainable


def save_adapter(model, path: Path) -> Path:
    """
    Write ONE safetensors file that diffusers can load with
    ``load_lora_weights()``.

    peft emits keys like
    ``base_model.model.down_blocks.1....to_q.lora_A.weight``; diffusers wants
    them under an ``unet.`` prefix. The rename is done explicitly rather than
    through ``convert_state_dict_to_peft``, which cannot infer the direction
    from a peft-shaped dict.

    Do NOT add ``.default.`` (i.e. ``lora_A.default.weight``). diffusers
    rewrites the adapter-name segment itself, so ``lora_A.default.weight``
    becomes ``lora_A.<adapter>.default.weight`` and matches nothing: the layers
    are created but stay empty (zero-init ``lora_B``, and with
    ``low_cpu_mem_usage`` — the default — a meta tensor, which is the
    "Cannot copy out of meta tensor" from run 46004724589). Verified: the
    ``.default.`` form lands 0/64 tensors, the plain form 64/64.
    """
    import torch
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import save_file

    state = get_peft_model_state_dict(model)
    converted: Dict[str, Any] = {}
    for key, value in state.items():
        key = key.replace("base_model.model.", "")
        key = key.replace(".lora_A.default.weight", ".lora_A.weight")
        key = key.replace(".lora_B.default.weight", ".lora_B.weight")
        if not key.startswith("unet."):
            key = "unet." + key
        # Keep the tensors in float32: the LoRA is applied to fp16 weights at
        # inference, and storing fp16 here loses precision for no size benefit.
        converted[key] = value.detach().to(device="cpu", dtype=torch.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(converted, str(path))
    log(f"wrote {path.name} ({path.stat().st_size / 1e6:.1f} MB, "
        f"{len(converted)} tensors)")
    return path


def _ceil_to_8(value: int) -> int:
    return int(math.ceil(value / 8.0) * 8)


def train(
    *,
    image_paths: List[Path],
    prompts: List[str],
    base_model: str,
    lora_path: Path,
    device: str = "cuda:0",
    resolution: int = 768,
    rank: int = 16,
    alpha: int = 16,
    learning_rate: float = 1e-4,
    max_train_steps: int = 400,
    batch_size: int = 1,
    seed: int = 1234,
    tag: str = "img",
) -> Dict[str, Any]:
    """
    Train the identity LoRA and return a summary for luca_metrics.json.

    Raises on any real failure — the caller treats a half-trained adapter as no
    adapter at all, because a carousel generated from one is worse than a clear
    error.
    """
    import torch
    import torch.nn.functional as F
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    if len(image_paths) != len(prompts):
        raise ValueError("every training image needs exactly one caption")
    if not image_paths:
        raise ValueError("no training images")

    timings: Dict[str, float] = {}
    started = time.perf_counter()
    torch.manual_seed(seed)

    # ---- tokeniser + encoders -------------------------------------------- #
    with timed("lora_load_text_encoders", timings, tag):
        tokenizer = CLIPTokenizer.from_pretrained(
            base_model, subfolder="tokenizer")
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            base_model, subfolder="tokenizer_2")
        text_encoder = CLIPTextModel.from_pretrained(
            base_model, subfolder="text_encoder", torch_dtype=torch.float16)
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            base_model, subfolder="text_encoder_2", torch_dtype=torch.float16)
        for encoder in (text_encoder, text_encoder_2):
            encoder.to(device, dtype=torch.float16).requires_grad_(False).eval()

    # ---- pre-encode prompt + pooled embeddings ---------------------------- #
    with timed("lora_encode_prompts", timings, tag):
        prompt_embeds, pooled_embeds = [], []
        with torch.no_grad():
            for caption in prompts:
                # `padding="max_length"` and no attention mask: this is exactly
                # what diffusers' own encode_prompt does, and omitting the mask
                # is what avoids the all-zero-embedding bug.
                tokens = tokenizer(
                    caption, padding="max_length", max_length=77,
                    truncation=True, return_tensors="pt").input_ids.to(device)
                tokens_2 = tokenizer_2(
                    caption, padding="max_length", max_length=77,
                    truncation=True, return_tensors="pt").input_ids.to(device)

                out_1 = text_encoder(tokens, output_hidden_states=True)
                out_2 = text_encoder_2(tokens_2, output_hidden_states=True)
                # SDXL concatenates the penultimate hidden states of both
                # encoders along the channel axis: 768 + 1280 = 2048.
                prompt_embeds.append(
                    torch.cat([out_1.hidden_states[-2], out_2.hidden_states[-2]],
                              dim=-1))
                # only the final encoder's pooled output is used
                pooled_embeds.append(out_2[0])
        prompt_embeds = torch.stack(prompt_embeds).squeeze(1)   # [N, 77, 2048]
        pooled_embeds = torch.stack(pooled_embeds).squeeze(1)   # [N, 1280]
        log(f"encoded {len(prompts)} captions → "
            f"{tuple(prompt_embeds.shape)} + pooled {tuple(pooled_embeds.shape)}",
            tag=tag)

    # The encoders are 2.5 GB and never updated again. Release them before the
    # UNet and its activations need the card.
    del text_encoder, text_encoder_2
    torch.cuda.empty_cache()

    # ---- cache latents once, then drop the VAE ---------------------------- #
    with timed("lora_cache_latents", timings, tag):
        from PIL import Image
        import numpy as np

        vae = AutoencoderKL.from_pretrained(
            base_model, subfolder="vae", torch_dtype=torch.float16)
        vae.to(device, dtype=torch.float16).requires_grad_(False).eval()
        # encode() is the broken path on diffusers 0.42; see _UnwrapDownBlock.
        patch_vae_encoder(vae)

        height = _ceil_to_8(resolution)
        width = _ceil_to_8(resolution)
        latents: List[Any] = []
        with torch.no_grad():
            for path in image_paths:
                with Image.open(path) as handle:
                    image = handle.convert("RGB")
                    # centre-crop to the training aspect, then resize: keeps the
                    # subject centred without distorting his face
                    side = min(image.size)
                    left = (image.size[0] - side) // 2
                    top = (image.size[1] - side) // 2
                    image = image.crop((left, top, left + side, top + side))
                    image = image.resize((width, height), Image.LANCZOS)
                    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
                tensor = torch.from_numpy(array).permute(2, 0, 1)[None]
                tensor = tensor.to(device, dtype=torch.float16)
                latent = vae.encode(tensor).latent_dist.sample()
                latents.append(latent * vae.config.scaling_factor)
        del vae
        torch.cuda.empty_cache()
        latents = torch.cat(latents, dim=0)                    # [N, 4, H/8, W/8]
        log(f"cached {latents.shape[0]} latents at "
            f"{latents.shape[-1] * 8}px → {tuple(latents.shape)}", tag=tag)

    # ---- UNet + adapter ---------------------------------------------------- #
    with timed("lora_load_unet", timings, tag):
        unet = UNet2DConditionModel.from_pretrained(
            base_model, subfolder="unet", torch_dtype=torch.float16)
        unet.to(device, dtype=torch.float16)
        unet.requires_grad_(False)
        unet.enable_gradient_checkpointing()   # trades compute for VRAM
        unet.train()
        model, trainable = build_adapter(unet, rank=rank, alpha=alpha)
        # The adapter stays in float32 on purpose. Its forward folds the LoRA
        # delta into the fp16 weights, so a fp16 *parameter* is not required —
        # and casting the trainable tensors to fp16 is what makes AdamW's
        # second-moment estimate underflow and the loss go NaN.
        for parameter in trainable:
            parameter.data = parameter.data.float()
        model.to(device)

    noise_scheduler = DDPMScheduler.from_pretrained(
        base_model, subfolder="scheduler")
    # SDXL's micro-conditioning: original size, crop offset, target size.
    add_time_ids = torch.tensor(
        [[height, width, 0, 0, height, width]], device=device,
        dtype=torch.float16)

    optimizer = torch.optim.AdamW(trainable, lr=learning_rate,
                                  betas=(0.9, 0.999), weight_decay=1e-2,
                                  eps=1e-8)

    log(f"training: {max_train_steps} steps · batch {batch_size} · "
        f"lr {learning_rate} · {height}px · fp16 + gradient checkpointing", tag=tag)

    losses: List[float] = []
    step_started = time.perf_counter()
    for step in range(max_train_steps):
        index = torch.randint(0, latents.shape[0], (batch_size,)).tolist()
        batch_latents = latents[index].to(device, dtype=torch.float16)
        batch_prompts = prompt_embeds[index].to(device, dtype=torch.float16)
        batch_pooled = pooled_embeds[index].to(device, dtype=torch.float16)
        batch_ids = add_time_ids.repeat(len(index), 1)

        noise = torch.randn_like(batch_latents)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (len(index),),
            device=device).long()
        noisy = noise_scheduler.add_noise(batch_latents, noise, timesteps)

        prediction = model(
            noisy, timesteps, encoder_hidden_states=batch_prompts,
            added_cond_kwargs={"text_embeds": batch_pooled,
                               "time_ids": batch_ids},
        ).sample

        # v-prediction models need a different target; SDXL/RealVisXL is
        # epsilon-prediction, but ask the scheduler rather than assume.
        if getattr(noise_scheduler.config, "prediction_type", "epsilon") == "v_prediction":
            target = noise_scheduler.get_velocity(batch_latents, noise, timesteps)
        else:
            target = noise

        loss = F.mse_loss(prediction.float(), target.float(),
                          reduction="mean")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        loss_value = float(loss.detach())
        losses.append(loss_value)
        if not math.isfinite(loss_value):
            # fp16 master weights + AdamW can overflow on the first few steps:
            # the loss goes NaN and every later step is garbage, but nothing
            # raises. This is the failure mode that silently produces an adapter
            # that changes nothing, so bail with a diagnosable message instead.
            log(f"loss became non-finite at step {step + 1} "
                f"({loss_value}) — fp16 overflow, restarting in float32 would be "
                f"needed", "ERROR", tag)
            raise RuntimeError(
                f"LoRA training diverged at step {step + 1}: loss={loss_value}. "
                f"Lower the learning rate or train the adapter in float32.")
        if step % 25 == 0 or step == max_train_steps - 1:
            elapsed = time.perf_counter() - step_started
            rate = elapsed / (step + 1)
            log(f"  step {step + 1}/{max_train_steps} · "
                f"loss {losses[-1]:.4f} · {rate:.2f}s/step · "
                f"eta {(max_train_steps - step - 1) * rate / 60:.1f} min",
                tag=tag)

    with timed("lora_save", timings, tag):
        save_adapter(model, lora_path)

    total_seconds = round(time.perf_counter() - started, 1)
    first = sum(losses[:25]) / max(1, len(losses[:25]))
    last = sum(losses[-25:]) / max(1, len(losses[-25:]))
    log(f"training done in {total_seconds / 60:.1f} min · "
        f"loss {first:.4f} → {last:.4f} · "
        f"{'decreasing' if last < first else 'NOT decreasing'}", tag=tag)

    # Free everything so the inference pipeline starts from a clean card.
    del model, unet, optimizer, latents, prompt_embeds, pooled_embeds
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    return {
        "base_model": base_model,
        "adapter": str(lora_path.name),
        "rank": rank,
        "alpha": alpha,
        "learning_rate": learning_rate,
        "max_train_steps": max_train_steps,
        "batch_size": batch_size,
        "resolution": height,
        "images": len(image_paths),
        "precision": "float16 with gradient checkpointing",
        "loss_first_25": round(first, 4),
        "loss_last_25": round(last, 4),
        "loss_decreasing": last < first,
        "adapter_mb": round(lora_path.stat().st_size / 1e6, 2),
        "train_seconds": total_seconds,
        "timings_seconds": timings,
    }
