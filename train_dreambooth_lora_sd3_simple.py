#!/usr/bin/env python
# coding=utf-8
"""
Simplified DreamBooth + LoRA training for SD3.
Text encoders are always frozen. Only the transformer LoRA layers are trained.

Supports two training paradigms:
  - Single-stage: baseline (random init) or full finetune.
  - Two-stage LoRA-One (default for gradient/SVD init):
      Stage 1: gradient estimation + LoRA init + training on extreme timesteps
               [0, stage1_t_low] ∪ [stage1_t_high, 999]
      Stage 2: fuse stage-1 LoRA into base weights, then gradient estimation +
               LoRA init + training on middle timesteps [stage1_t_low, stage1_t_high].
"""
import argparse
import copy
import itertools
import json
import logging
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint
import transformers
import yaml
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import (
    convert_unet_state_dict_to_peft,
)
from diffusers.utils.torch_utils import is_compiled_module

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Timestep helpers
# ---------------------------------------------------------------------------

def get_sigmas(noise_scheduler, timesteps, device, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def sample_timesteps_in_ranges(bsz, valid_index_ranges, noise_scheduler, device):
    """Sample `bsz` timesteps uniformly from the union of valid index ranges."""
    num_train = noise_scheduler.config.num_train_timesteps
    valid_pool = []
    for lo, hi in valid_index_ranges:
        valid_pool.extend(range(max(0, lo), min(hi + 1, num_train)))
    if not valid_pool:
        raise ValueError(f"Empty timestep pool for ranges {valid_index_ranges}")
    idx = [valid_pool[random.randrange(len(valid_pool))] for _ in range(bsz)]
    t_indices = torch.tensor(idx, dtype=torch.long)
    return noise_scheduler.timesteps[t_indices].to(device)


# ---------------------------------------------------------------------------
# Gradient estimation with timestep range restriction
# ---------------------------------------------------------------------------

def estimate_gradient_in_range(
    transformer,
    vae,
    dataloader,
    noise_scheduler,
    prompt_embeds,
    pooled_prompt_embeds,
    device,
    weight_dtype,
    valid_index_ranges,
    vae_shift_factor,
    vae_scale_factor,
    num_iters=4,
    latents_cache=None,
):
    """
    Accumulate parameter gradients for `transformer` over `num_iters` batches,
    restricting timestep sampling to the union of `valid_index_ranges`.

    Returns a dict {param_name: averaged_gradient_tensor}.
    """
    transformer.eval()

    # Temporarily enable grad on ALL parameters
    saved_grad = {n: p.requires_grad for n, p in transformer.named_parameters()}
    for p in transformer.parameters():
        p.requires_grad_(True)
    transformer.zero_grad()

    num_train = noise_scheduler.config.num_train_timesteps
    valid_pool = []
    for lo, hi in valid_index_ranges:
        valid_pool.extend(range(max(0, lo), min(hi + 1, num_train)))
    if not valid_pool:
        raise ValueError(f"Empty timestep pool for ranges {valid_index_ranges}")

    data_iter = iter(dataloader)
    from tqdm import tqdm
    for it in tqdm(range(num_iters), desc="Training", unit="iter"):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        if latents_cache is not None:
            model_input = latents_cache[it % len(latents_cache)].sample()
        else:
            with torch.no_grad():
                pv = batch["pixel_values"].to(device, dtype=torch.float32)
                model_input = vae.encode(pv).latent_dist.sample()

        model_input = (model_input - vae_shift_factor) * vae_scale_factor
        model_input = model_input.to(dtype=weight_dtype)
        noise = torch.randn_like(model_input)
        bsz = model_input.shape[0]

        raw_idx = [valid_pool[random.randrange(len(valid_pool))] for _ in range(bsz)]
        timesteps = noise_scheduler.timesteps[torch.tensor(raw_idx)].to(device)
        sigmas = get_sigmas(noise_scheduler, timesteps, device,
                            n_dim=model_input.ndim, dtype=model_input.dtype)
        noisy = (1.0 - sigmas) * model_input + sigmas * noise

        model_pred = transformer(
            hidden_states=noisy,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]

        loss = torch.mean((model_pred.float() - (noise - model_input).float()) ** 2)
        loss.backward()

    named_grads = {}
    for name, param in transformer.named_parameters():
        if param.grad is not None:
            named_grads[name] = param.grad.detach().clone() / num_iters

    # Restore grad state
    for name, param in transformer.named_parameters():
        param.requires_grad_(saved_grad[name])
    transformer.zero_grad()
    transformer.train()

    return named_grads


# ---------------------------------------------------------------------------
# LoRA merge helper
# ---------------------------------------------------------------------------

def merge_and_remove_lora(transformer):
    """
    Fuse the active "default" LoRA adapter into the base weights in-place,
    then remove the adapter so add_adapter("default") can be called again.

    Does not rely on merge_adapter() / delete_adapter() availability.
    """
    from peft.tuners.lora import LoraLayer

    # Step 1: W += scale * B @ A  for every LoRA layer
    for module in transformer.modules():
        if not isinstance(module, LoraLayer):
            continue
        lora_A = getattr(module, "lora_A", {})
        lora_B = getattr(module, "lora_B", {})
        scaling = getattr(module, "scaling", {})
        for adapter_name in list(lora_A.keys()):
            if adapter_name not in lora_B:
                continue
            A = lora_A[adapter_name].weight       # (rank, in_features)
            B = lora_B[adapter_name].weight       # (out_features, rank)
            scale = scaling.get(adapter_name, 1.0)
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data += (scale * B @ A).to(module.weight.data.dtype)

    # Step 2: Remove adapter-specific state from every LoRA layer so the
    #         injection code can re-add "default" cleanly.
    for module in transformer.modules():
        if not isinstance(module, LoraLayer):
            continue
        for attr in ("lora_A", "lora_B", "lora_dropout", "lora_embedding_A",
                     "lora_embedding_B", "scaling", "lora_magnitude_vector"):
            obj = getattr(module, attr, None)
            if isinstance(obj, torch.nn.ModuleDict):
                obj.clear()
            elif isinstance(obj, dict):
                obj.clear()

    # Step 3: Remove "default" from the top-level peft_config so
    #         add_adapter("default") passes the duplicate-name check.
    peft_config = getattr(transformer, "peft_config", {})
    peft_config.pop("default", None)


# ---------------------------------------------------------------------------
# Text encoder helpers
# ---------------------------------------------------------------------------

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path, revision, subfolder="text_encoder"):
    config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
    else:
        raise ValueError(f"Unsupported text encoder class: {model_class}")


def _encode_prompt_with_t5(text_encoder, tokenizer, max_sequence_length, prompt,
                            num_images_per_prompt=1, device=None):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    text_inputs = tokenizer(prompt, padding="max_length", max_length=max_sequence_length,
                            truncation=True, add_special_tokens=True, return_tensors="pt")
    prompt_embeds = text_encoder(text_inputs.input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    return prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)


def _encode_prompt_with_clip(text_encoder, tokenizer, prompt, device=None, num_images_per_prompt=1):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    text_inputs = tokenizer(prompt, padding="max_length", max_length=77,
                            truncation=True, return_tensors="pt")
    out = text_encoder(text_inputs.input_ids.to(device), output_hidden_states=True)
    pooled = out[0]
    embeds = out.hidden_states[-2].to(dtype=text_encoder.dtype, device=device)
    _, seq_len, _ = embeds.shape
    embeds = embeds.repeat(1, num_images_per_prompt, 1)
    return embeds.view(batch_size * num_images_per_prompt, seq_len, -1), pooled


def encode_prompt(text_encoders, tokenizers, prompt, max_sequence_length,
                  device=None, num_images_per_prompt=1):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    clip_embeds_list, pooled_list = [], []
    for tokenizer, text_encoder in zip(tokenizers[:2], text_encoders[:2]):
        embeds, pooled = _encode_prompt_with_clip(
            text_encoder, tokenizer, prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
        )
        clip_embeds_list.append(embeds)
        pooled_list.append(pooled)
    clip_prompt_embeds = torch.cat(clip_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(pooled_list, dim=-1)
    t5_embed = _encode_prompt_with_t5(
        text_encoders[-1], tokenizers[-1], max_sequence_length, prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[-1].device,
    )
    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    return torch.cat([clip_prompt_embeds, t5_embed], dim=-2), pooled_prompt_embeds


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DreamBoothDataset(Dataset):
    def __init__(self, instance_data_root, instance_prompt, class_prompt=None,
                 class_data_root=None, class_num=None, size=512, repeats=1,
                 center_crop=False, random_flip=False, resolution=512,
                 dataset_name=None, dataset_config_name=None, cache_dir=None,
                 image_column="image", caption_column=None):
        self.size = size
        self.center_crop = center_crop
        self.instance_prompt = instance_prompt
        self.custom_instance_prompts = None
        self.class_prompt = class_prompt

        if dataset_name is not None:
            try:
                from datasets import load_dataset
            except ImportError:
                raise ImportError("Install `datasets` to use --dataset_name.")
            dataset = load_dataset(dataset_name, dataset_config_name, cache_dir=cache_dir)
            column_names = dataset["train"].column_names
            image_column = image_column if image_column in column_names else column_names[0]
            instance_images = dataset["train"][image_column]
            if caption_column is not None and caption_column in column_names:
                custom_prompts = dataset["train"][caption_column]
                self.custom_instance_prompts = []
                for cap in custom_prompts:
                    self.custom_instance_prompts.extend(itertools.repeat(cap, repeats))
        else:
            self.instance_data_root = Path(instance_data_root)
            if not self.instance_data_root.exists():
                raise ValueError(f"Instance data directory does not exist: {instance_data_root}")
            instance_images = [Image.open(p) for p in sorted(self.instance_data_root.iterdir())]

        self.instance_images = []
        for img in instance_images:
            self.instance_images.extend(itertools.repeat(img, repeats))

        train_resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        train_crop = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        train_flip_fn = transforms.RandomHorizontalFlip(p=1.0)
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.pixel_values = []
        for image in self.instance_images:
            image = exif_transpose(image)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            image = train_resize(image)
            if random_flip and random.random() < 0.5:
                image = train_flip_fn(image)
            if center_crop:
                image = train_crop(image)
            else:
                y1, x1, h, w = train_crop.get_params(image, (resolution, resolution))
                image = crop(image, y1, x1, h, w)
            self.pixel_values.append(train_transforms(image))

        self.num_instance_images = len(self.instance_images)
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = (
                min(len(self.class_images_path), class_num) if class_num
                else len(self.class_images_path)
            )
            self._length = max(self.num_class_images, self.num_instance_images)
            self.image_transforms = transforms.Compose([
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
        else:
            self.class_data_root = None

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        example["instance_images"] = self.pixel_values[index % self.num_instance_images]
        if self.custom_instance_prompts:
            caption = self.custom_instance_prompts[index % self.num_instance_images]
            example["instance_prompt"] = caption if caption else self.instance_prompt
        else:
            example["instance_prompt"] = self.instance_prompt
        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt"] = self.class_prompt
        return example


def collate_fn(examples, with_prior_preservation=False):
    pixel_values = [e["instance_images"] for e in examples]
    prompts = [e["instance_prompt"] for e in examples]
    if with_prior_preservation:
        pixel_values += [e["class_images"] for e in examples]
        prompts += [e["class_prompt"] for e in examples]
    pixel_values = torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float()
    return {"pixel_values": pixel_values, "prompts": prompts}


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Two-stage SD3 DreamBooth + LoRA training (text encoders always frozen)."
    )

    # Model
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)

    # Data
    parser.add_argument("--instance_data_dir", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--image_column", type=str, default="image")
    parser.add_argument("--caption_column", type=str, default=None)
    parser.add_argument("--instance_prompt", type=str, required=True)
    parser.add_argument("--class_prompt", type=str, default=None)
    parser.add_argument("--class_data_dir", type=str, default=None)
    parser.add_argument("--with_prior_preservation", action="store_true", default=False)
    parser.add_argument("--prior_loss_weight", type=float, default=1.0)
    parser.add_argument("--num_class_images", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--center_crop", action="store_true", default=False)
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument("--max_sequence_length", type=int, default=77)

    # Output
    parser.add_argument("--output_dir", type=str, default="sd3-dreambooth-lora")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="tensorboard")

    # Training
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total stage-2 training steps. Overrides num_train_epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--cache_latents", action="store_true", default=False)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--upcast_before_saving", action="store_true", default=False)

    # Loss / timestep
    parser.add_argument("--weighting_scheme", type=str, default="logit_normal",
                        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"])
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument("--precondition_outputs", type=int, default=0)

    # Two-stage timestep split
    parser.add_argument("--stage1_t_low", type=int, default=200,
                        help="Stage 1 uses timestep indices [0, stage1_t_low] ∪ [stage1_t_high, 999]. "
                             "Stage 2 uses [stage1_t_low, stage1_t_high].")
    parser.add_argument("--stage1_t_high", type=int, default=800,
                        help="See --stage1_t_low.")
    parser.add_argument("--stage1_steps", type=int, default=100,
                        help="Number of training steps for stage 1.")
    parser.add_argument("--stage1_iters", type=int, default=None,
                        help="Number of gradient accumulation iters for stage-1 reinit. "
                             "Defaults to the 'iters' field in the init config YAML.")
    parser.add_argument("--stage2_iters", type=int, default=None,
                        help="Number of gradient accumulation iters for stage-2 reinit. "
                             "Defaults to the 'iters' field in the init config YAML.")

    # Optimizer
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-4)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # LoRA
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--lora_layers", type=str, default=None,
                        help="Comma-separated transformer layer names for LoRA.")
    parser.add_argument("--lora_blocks", type=str, default=None,
                        help="Comma-separated block indices to restrict LoRA to.")

    # LoRA init / reinit
    parser.add_argument("--baseline", action="store_true", default=False,
                        help="Single-stage training with random (Gaussian) LoRA init.")
    parser.add_argument("--full_finetune", action="store_true", default=False,
                        help="Single-stage full finetune (no LoRA). Saves trainable_weights.pt.")
    parser.add_argument("--init_config", type=str, default="config/gradient.yaml",
                        help="Path to LoRA init config YAML (gradient.yaml / svd.yaml / default.yaml).")
    parser.add_argument("--reinit_strategy", type=str, default=None,
                        choices=["medium", "crossAtt", "low", "high", "all"])
    parser.add_argument("--stable_gamma", type=int, default=1)
    parser.add_argument("--direction", type=str, default="LoRA-One")
    parser.add_argument("--lr_scale", type=float, default=1.0,
                        help="LR divisor for reinit'd LoRA layers.")
    parser.add_argument("--gradient_path", type=str, default=None,
                        help="Precomputed gradient .pt for stage 1 (skips live estimation).")
    parser.add_argument("--reinit_data_dir", type=str, default=None,
                        help="Data dir for gradient estimation (defaults to instance_data_dir).")
    parser.add_argument("--reinit_only", action="store_true", default=False,
                        help="Save checkpoint-0 after stage-1 reinit then exit.")
    parser.add_argument("--svd_algo", type=str, default="svd", choices=["svd", "pmd"],
                        help="SVD algorithm for LoRA-One reinitialization: 'svd' (standard) or 'pmd' (sparse PMD).")
    parser.add_argument("--pmd_c_u", type=float, default=1.0,
                        help="L1 sparsity bound on left vectors for PMD (c_u). Only used when --svd_algo pmd.")
    parser.add_argument("--pmd_c_v", type=float, default=1.0,
                        help="L1 sparsity bound on right vectors for PMD (c_v). Only used when --svd_algo pmd.")

    # Checkpointing
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument("--local_rank", type=int, default=-1)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.instance_data_dir is None:
        raise ValueError("Specify either --dataset_name or --instance_data_dir")
    if args.dataset_name is not None and args.instance_data_dir is not None:
        raise ValueError("Specify only one of --dataset_name or --instance_data_dir")

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


# ---------------------------------------------------------------------------
# Helpers shared across stages
# ---------------------------------------------------------------------------

def build_lora_config(args):
    return LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=_resolve_target_modules(args),
    )


def _resolve_target_modules(args):
    if args.lora_layers is not None:
        mods = [l.strip() for l in args.lora_layers.split(",")]
    else:
        mods = [
            "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
            "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
            "ff.net.0.proj", "ff.net.2",
            "norm_out.linear", "proj_out",
        ]
    if args.lora_blocks is not None:
        blocks = [int(b.strip()) for b in args.lora_blocks.split(",")]
        mods = [f"transformer_blocks.{blk}.{mod}" for blk in blocks for mod in mods]
    return mods


def build_reinit_conf(args, init_conf_base):
    """Apply reinit_strategy and CLI overrides to a loaded init_conf dict."""
    conf = dict(init_conf_base)
    conf["direction"] = args.direction
    conf["stable_gamma"] = args.stable_gamma
    conf["svd_algo"] = args.svd_algo
    conf["pmd_c_u"] = args.pmd_c_u
    conf["pmd_c_v"] = args.pmd_c_v
    if args.reinit_strategy:
        s = args.reinit_strategy.lower()
        if s == "medium":
            conf["reinit_pos_start"] = 10
            conf["reinit_pos_end"] = 13
        elif s == "crossAtt":
            conf["reinit_pos_start"] = 0
            conf["reinit_pos_end"] = 23
            conf["lora_module"] = "crossAtt"
        elif s == "all":
            conf["reinit_pos_start"] = 0
            conf["reinit_pos_end"] = 23
            conf["lora_module"] = "all"
    return conf


def split_lora_params(transformer, inited_modules):
    """Split trainable params into reinit'd and non-reinit'd groups."""
    reinit_params, other_params = [], []
    for name, param in transformer.named_parameters():
        if not param.requires_grad:
            continue
        if ".".join(name.split(".")[:-3]) in inited_modules:
            reinit_params.append(param)
        else:
            other_params.append(param)
    return other_params, reinit_params


def make_optimizer(args, params_to_optimize):
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            cls = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError("Install bitsandbytes for 8-bit Adam.")
    else:
        cls = torch.optim.AdamW
    return cls(
        params_to_optimize,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )


def make_lr_scheduler(args, optimizer, total_steps, accelerator):
    return get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=total_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )


def run_training_loop(
    transformer, optimizer, lr_scheduler, train_dataloader,
    accelerator, args, noise_scheduler_copy,
    prompt_embeds, pooled_prompt_embeds,
    weight_dtype, vae, vae_shift_factor, vae_scale_factor,
    all_lora_params, latents_cache,
    global_step, max_steps, stage_name,
    valid_index_ranges,          # None → sample all timesteps normally
    save_ckpt_list,
    first_epoch=0,
):
    """
    Generic training loop shared by both stages.

    `valid_index_ranges`: list of (lo, hi) integer index pairs to restrict
    timestep sampling. If None, uses the standard weighting-scheme sampling.

    Returns updated global_step.
    """
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_epochs = math.ceil(max_steps / num_update_steps_per_epoch)

    loss_log_path = os.path.join(args.output_dir, f"loss_curve_{stage_name}.jsonl")
    loss_log_file = open(loss_log_path, "w") if accelerator.is_main_process else None

    progress_bar = tqdm(
        range(global_step, global_step + max_steps),
        desc=f"[{stage_name}] Steps",
        disable=not accelerator.is_local_main_process,
    )

    steps_done_in_stage = 0

    for epoch in range(first_epoch, num_epochs):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                # Prompt embeddings (custom per-image captions not supported here for simplicity)
                p_embeds = prompt_embeds
                pp_embeds = pooled_prompt_embeds

                # Latents
                if latents_cache is not None:
                    model_input = latents_cache[step % len(latents_cache)].sample()
                else:
                    pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                    model_input = vae.encode(pixel_values).latent_dist.sample()

                model_input = (model_input - vae_shift_factor) * vae_scale_factor
                model_input = model_input.to(dtype=weight_dtype)
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                # Timestep sampling
                if valid_index_ranges is not None:
                    timesteps = sample_timesteps_in_ranges(
                        bsz, valid_index_ranges, noise_scheduler_copy, model_input.device
                    )
                else:
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=args.weighting_scheme,
                        batch_size=bsz,
                        logit_mean=args.logit_mean,
                        logit_std=args.logit_std,
                        mode_scale=args.mode_scale,
                    )
                    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                    timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                sigmas = get_sigmas(noise_scheduler_copy, timesteps, model_input.device,
                                    n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                model_pred = transformer(
                    hidden_states=noisy_model_input,
                    timestep=timesteps,
                    encoder_hidden_states=p_embeds,
                    pooled_projections=pp_embeds,
                    return_dict=False,
                )[0]

                if args.precondition_outputs:
                    model_pred = model_pred * (-sigmas) + noisy_model_input

                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                target = model_input if args.precondition_outputs else noise - model_input

                if args.with_prior_preservation:
                    model_pred, pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)
                    w, w_prior = torch.chunk(weighting, 2, dim=0)
                    prior_loss = torch.mean(
                        (w_prior.float() * (pred_prior.float() - target_prior.float()) ** 2
                         ).reshape(target_prior.shape[0], -1), 1).mean()
                    weighting = w

                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2
                     ).reshape(target.shape[0], -1), 1).mean()

                if args.with_prior_preservation:
                    loss = loss + args.prior_loss_weight * prior_loss

                loss_val = loss.item()
                print(f"[{stage_name}] Step {global_step}: Loss = {loss_val:.6f}")
                if loss_log_file is not None:
                    loss_log_file.write(json.dumps({"step": global_step, "loss": loss_val}) + "\n")
                    loss_log_file.flush()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(all_lora_params, args.max_grad_norm)
                    if global_step in save_ckpt_list:
                        step_gradients = {
                            f"transformer_{n}": p.grad.detach().clone()
                            for n, p in transformer.named_parameters() if p.grad is not None
                        }
                        # torch.save(step_gradients,
                        #            f"{args.output_dir}/grads_{stage_name}_step_{global_step}.pt")

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                steps_done_in_stage += 1

                if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                    if global_step in save_ckpt_list:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = sorted(
                                [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")],
                                key=lambda x: int(x.split("-")[1]),
                            )
                            while len(checkpoints) >= args.checkpoints_total_limit:
                                shutil.rmtree(os.path.join(args.output_dir, checkpoints.pop(0)))
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"[{stage_name}] Saved checkpoint to {save_path}")

            accelerator.log(
                {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]},
                step=global_step,
            )
            progress_bar.set_postfix(loss=loss.detach().item())

            if steps_done_in_stage >= max_steps:
                break
        if steps_done_in_stage >= max_steps:
            break

    if loss_log_file is not None:
        loss_log_file.close()

    return global_step


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        config_path = os.path.join(args.output_dir, "run_config.json")
        with open(config_path, "w") as f:
            json.dump(vars(args), f, indent=2)

    # ------------------------------------------------------------------
    # Load frozen text encoders (encode once then free)
    # ------------------------------------------------------------------
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision)
    tokenizer_two = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision)
    tokenizer_three = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_3", revision=args.revision)

    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision)
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2")

    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder",
        revision=args.revision, variant=args.variant)
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2",
        revision=args.revision, variant=args.variant)
    from transformers import T5EncoderModel
    text_encoder_three = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_3",
        revision=args.revision, variant=args.variant)

    # ------------------------------------------------------------------
    # Load scheduler, VAE, transformer
    # ------------------------------------------------------------------
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae",
        revision=args.revision, variant=args.variant)
    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer",
        revision=args.revision, variant=args.variant)

    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=torch.float32)
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    text_encoder_three.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate *= (
            args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    def unwrap_model(model):
        m = accelerator.unwrap_model(model)
        return m._orig_mod if is_compiled_module(m) else m

    # ------------------------------------------------------------------
    # Dataset & dataloader
    # ------------------------------------------------------------------
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_prompt=args.class_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_num=args.num_class_images,
        size=args.resolution,
        repeats=args.repeats,
        center_crop=args.center_crop,
        random_flip=args.random_flip,
        resolution=args.resolution,
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        cache_dir=args.cache_dir,
        image_column=args.image_column,
        caption_column=args.caption_column,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
    )

    # ------------------------------------------------------------------
    # Pre-compute text embeddings (frozen encoders, encode once)
    # ------------------------------------------------------------------
    tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
    text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, args.instance_prompt, args.max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(accelerator.device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)

    del tokenizers, text_encoders, text_encoder_one, text_encoder_two, text_encoder_three
    free_memory()

    # ------------------------------------------------------------------
    # Cache latents (optional)
    # ------------------------------------------------------------------
    vae_shift_factor = vae.config.shift_factor
    vae_scale_factor = vae.config.scaling_factor
    latents_cache = None
    if args.cache_latents:
        latents_cache = []
        for batch in tqdm(train_dataloader, desc="Caching latents"):
            with torch.no_grad():
                pv = batch["pixel_values"].to(accelerator.device, dtype=torch.float32)
                latents_cache.append(vae.encode(pv).latent_dist)
        # Keep VAE alive for gradient estimation if needed; freed after stage 2 grad estimation.

    # ------------------------------------------------------------------
    # Checkpoint save / load hooks
    # ------------------------------------------------------------------
    def save_model_hook(models, weights, output_dir):
        if not accelerator.is_main_process:
            return
        if args.full_finetune:
            for model in models:
                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    m = unwrap_model(model)
                    trainable = {n: p.data.clone() for n, p in m.named_parameters() if p.requires_grad}
                    torch.save(trainable, os.path.join(output_dir, "trainable_weights.pt"))
                if weights:
                    weights.pop()
        else:
            for model in models:
                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    m = unwrap_model(model)
                    if args.upcast_before_saving:
                        m = m.to(torch.float32)
                    lora_layers = get_peft_model_state_dict(m)
                    StableDiffusion3Pipeline.save_lora_weights(
                        output_dir, transformer_lora_layers=lora_layers)
                if weights:
                    weights.pop()

    def load_model_hook(models, input_dir):
        if args.full_finetune:
            while models:
                model = models.pop()
                weights_path = os.path.join(input_dir, "trainable_weights.pt")
                if os.path.exists(weights_path):
                    unwrap_model(model).load_state_dict(
                        torch.load(weights_path, map_location="cpu"), strict=False)
            return
        transformer_ = None
        while models:
            model = models.pop()
            if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                transformer_ = unwrap_model(model)
        lora_sd = StableDiffusion3Pipeline.lora_state_dict(input_dir)
        t_sd = {k.replace("transformer.", ""): v
                for k, v in lora_sd.items() if k.startswith("transformer.")}
        t_sd = convert_unet_state_dict_to_peft(t_sd)
        incompatible = set_peft_model_state_dict(transformer_, t_sd, adapter_name="default")
        if incompatible and getattr(incompatible, "unexpected_keys", None):
            logger.warning(f"Unexpected LoRA keys: {incompatible.unexpected_keys}")
        if args.mixed_precision == "fp16":
            cast_training_params([transformer_])

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # ------------------------------------------------------------------
    # Gradient estimation dataloader (may differ from train_dataloader)
    # ------------------------------------------------------------------
    reinit_data_dir = args.reinit_data_dir if args.reinit_data_dir else args.instance_data_dir
    grad_dataset = DreamBoothDataset(
        instance_data_root=reinit_data_dir,
        instance_prompt=args.instance_prompt,
        size=args.resolution,
        repeats=args.repeats,
        center_crop=args.center_crop,
        resolution=args.resolution,
    )
    grad_dataloader = torch.utils.data.DataLoader(
        grad_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, False),
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    # ------------------------------------------------------------------
    # Timestep ranges
    # ------------------------------------------------------------------
    t_low, t_high = args.stage1_t_low, args.stage1_t_high
    stage1_ranges = [(0, t_low), (t_high, 999)]       # extreme timesteps
    stage2_ranges = [(t_low, t_high)]                  # middle timesteps
    # stage2_ranges = [(0, 999)]

    logger.info(f"Stage 1 timestep ranges: {stage1_ranges} ({args.stage1_steps} steps)")
    logger.info(f"Stage 2 timestep ranges: {stage2_ranges}")

    # ------------------------------------------------------------------
    # Load init config
    # ------------------------------------------------------------------
    with open(args.init_config, "r") as f:
        init_conf_base = yaml.safe_load(f)

    from lora_one_utils import reinit_lora

    save_ckpt_list = [0, 1, 10, 20, 50, 100, 150, 200, 300, 400, 500]
    global_step = 0

    # ==========================================
    # SINGLE-STAGE PATH (baseline / full_finetune)
    # ==========================================
    if args.baseline or args.full_finetune:
        if args.full_finetune:
            args.adam_weight_decay = 1e-4
            args.learning_rate = 1e-5
            target_modules = _resolve_target_modules(args)
            for name, param in transformer.named_parameters():
                if any(t in name for t in target_modules):
                    param.requires_grad = True
                    param.data = param.data.to(torch.float32)
            lora_params = [p for p in transformer.parameters() if p.requires_grad]
            reinit_params = []
        else:
            transformer.add_adapter(build_lora_config(args))
            if args.mixed_precision == "fp16":
                cast_training_params([transformer], dtype=torch.float32)
            lora_params = [p for p in transformer.parameters() if p.requires_grad]
            reinit_params = []

        optimizer = make_optimizer(args, [{"params": lora_params, "lr": args.learning_rate}])
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        max_train_steps = args.max_train_steps or args.num_train_epochs * num_update_steps_per_epoch
        lr_scheduler = make_lr_scheduler(args, optimizer, max_train_steps, accelerator)
        transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, train_dataloader, lr_scheduler)

        accelerator.init_trackers("dreambooth-sd3-lora", config=vars(args))
        accelerator.save_state(os.path.join(args.output_dir, "checkpoint-0"))

        global_step = run_training_loop(
            transformer, optimizer, lr_scheduler, train_dataloader,
            accelerator, args, noise_scheduler_copy,
            prompt_embeds, pooled_prompt_embeds,
            weight_dtype, vae, vae_shift_factor, vae_scale_factor,
            lora_params, latents_cache,
            global_step=0, max_steps=max_train_steps,
            stage_name="main",
            valid_index_ranges=None,   # standard sampling
            save_ckpt_list=save_ckpt_list,
        )

        # Save final
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            tr = unwrap_model(transformer)
            if args.full_finetune:
                trainable = {n: p.data.clone() for n, p in tr.named_parameters() if p.requires_grad}
                torch.save(trainable, os.path.join(args.output_dir, "trainable_weights.pt"))
            else:
                if args.upcast_before_saving:
                    tr = tr.to(torch.float32)
                StableDiffusion3Pipeline.save_lora_weights(
                    save_directory=args.output_dir,
                    transformer_lora_layers=get_peft_model_state_dict(tr),
                )
        accelerator.end_training()
        return

    # ==========================================
    # TWO-STAGE PATH (gradient-based / SVD init)
    # ==========================================
    accelerator.init_trackers("dreambooth-sd3-lora-twostage", config=vars(args))

    # ---- STAGE 1: extreme timesteps ----------------------------------------

    logger.info("=" * 60)
    logger.info("STAGE 1: gradient estimation on extreme timesteps")
    logger.info("=" * 60)

    if args.gradient_path is not None:
        logger.info(f"Loading precomputed stage-1 gradients from {args.gradient_path}")
        named_grads_s1 = torch.load(args.gradient_path, map_location="cpu")
    else:
        named_grads_s1 = estimate_gradient_in_range(
            transformer, vae, grad_dataloader, noise_scheduler_copy,
            prompt_embeds, pooled_prompt_embeds,
            accelerator.device, weight_dtype,
            valid_index_ranges=stage1_ranges,
            vae_shift_factor=vae_shift_factor,
            vae_scale_factor=vae_scale_factor,
            num_iters=args.stage1_iters if args.stage1_iters is not None else init_conf_base.get("iters", 4),
            latents_cache=latents_cache,
        )

    # torch.save(named_grads_s1, os.path.join(args.output_dir, "named_grads_stage1.pt"))

    # Add LoRA and reinit with stage-1 gradients
    transformer.add_adapter(build_lora_config(args))
    init_conf_s1 = build_reinit_conf(args, init_conf_base)
    _, inited_modules_s1 = reinit_lora(transformer, init_conf_s1, {"named_grads": named_grads_s1})

    if args.mixed_precision == "fp16":
        cast_training_params([transformer], dtype=torch.float32)

    lora_params_s1, reinit_params_s1 = split_lora_params(transformer, inited_modules_s1)
    all_params_s1 = lora_params_s1 + reinit_params_s1

    params_to_opt_s1 = [{"params": lora_params_s1, "lr": args.learning_rate}]
    if reinit_params_s1:
        params_to_opt_s1.append({"params": reinit_params_s1, "lr": args.learning_rate / args.lr_scale})

    optimizer_s1 = make_optimizer(args, params_to_opt_s1)
    lr_scheduler_s1 = make_lr_scheduler(args, optimizer_s1, args.stage1_steps, accelerator)

    transformer, optimizer_s1, train_dataloader, lr_scheduler_s1 = accelerator.prepare(
        transformer, optimizer_s1, train_dataloader, lr_scheduler_s1)

    # Save checkpoint-0 (after stage-1 reinit, before any training)
    accelerator.save_state(os.path.join(args.output_dir, "checkpoint-0"))
    if args.reinit_only:
        accelerator.end_training()
        return

    if args.seed is not None:
        set_seed(args.seed)

    logger.info(f"Stage 1 training for {args.stage1_steps} steps "
                f"(timestep ranges {stage1_ranges})")

    global_step = run_training_loop(
        transformer, optimizer_s1, lr_scheduler_s1, train_dataloader,
        accelerator, args, noise_scheduler_copy,
        prompt_embeds, pooled_prompt_embeds,
        weight_dtype, vae, vae_shift_factor, vae_scale_factor,
        all_params_s1, latents_cache,
        global_step=0, max_steps=args.stage1_steps,
        stage_name="stage1",
        valid_index_ranges=None,
        save_ckpt_list=save_ckpt_list,
    )

    
    # Save stage-1 LoRA checkpoint
    if accelerator.is_main_process:
        s1_ckpt_dir = os.path.join(args.output_dir, "checkpoint-stage1-final")
        os.makedirs(s1_ckpt_dir, exist_ok=True)
        tr_raw = unwrap_model(transformer)
        StableDiffusion3Pipeline.save_lora_weights(
            s1_ckpt_dir,
            transformer_lora_layers=get_peft_model_state_dict(tr_raw),
        )
        logger.info(f"Stage-1 LoRA saved to {s1_ckpt_dir}")

    # ---- STAGE 2: fuse stage-1 LoRA, then middle timesteps ----------------

    logger.info("=" * 60)
    logger.info("STAGE 2: fusing stage-1 LoRA, then gradient estimation on middle timesteps")
    logger.info("=" * 60)

    # Unwrap from accelerator to get the raw PEFT model
    transformer_raw = unwrap_model(transformer)

    # Fuse stage-1 LoRA into base weights and remove the adapter
    merge_and_remove_lora(transformer_raw)
    transformer_raw.requires_grad_(False)

    logger.info("Stage-1 LoRA fused into base weights.")

    # Estimate stage-2 gradients on the fused model
    
    named_grads_s2 = estimate_gradient_in_range(
        transformer_raw, vae, grad_dataloader, noise_scheduler_copy,
        prompt_embeds, pooled_prompt_embeds,
        accelerator.device, weight_dtype,
        valid_index_ranges=stage2_ranges,
        vae_shift_factor=vae_shift_factor,
        vae_scale_factor=vae_scale_factor,
        num_iters=args.stage2_iters if args.stage2_iters is not None else init_conf_base.get("iters", 4),
        latents_cache=latents_cache,
    )
    # Strip ".base_layer." from keys before saving (same cleanup applied before reinit_lora)
    named_grads_s2 = {k.replace(".base_layer.", "."): v for k, v in named_grads_s2.items()}
    # torch.save(named_grads_s2, os.path.join(args.output_dir, "named_grads_stage2.pt"))

    # VAE no longer needed after gradient estimation
    if latents_cache is not None:
        del vae
        free_memory()

    # Add fresh LoRA adapter for stage 2 (same "default" name, now safe after merge+remove)
    transformer_raw.add_adapter(build_lora_config(args))
    init_conf_s2 = build_reinit_conf(args, init_conf_base)

    # After merging, LoRA layer wrappers remain so gradient keys contain ".base_layer."
    # (e.g. "transformer_blocks.0.attn.to_q.base_layer.weight").
    # Strip it so keys match the plain-weight naming expected by reinit_lora.
    named_grads_s2 = {k.replace(".base_layer.", "."): v for k, v in named_grads_s2.items()}

    _, inited_modules_s2 = reinit_lora(transformer_raw, init_conf_s2, {"named_grads": named_grads_s2})

    if args.mixed_precision == "fp16":
        cast_training_params([transformer_raw], dtype=torch.float32)

    lora_params_s2, reinit_params_s2 = split_lora_params(transformer_raw, inited_modules_s2)
    all_params_s2 = lora_params_s2 + reinit_params_s2

    params_to_opt_s2 = [{"params": lora_params_s2, "lr": args.learning_rate}]
    if reinit_params_s2:
        params_to_opt_s2.append({"params": reinit_params_s2, "lr": args.learning_rate / args.lr_scale})

    # Resolve stage-2 total steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps_s2 = args.max_train_steps or args.num_train_epochs * num_update_steps_per_epoch
    max_train_steps_s2 = max_train_steps_s2 - args.stage1_steps

    optimizer_s2 = make_optimizer(args, params_to_opt_s2)
    lr_scheduler_s2 = make_lr_scheduler(args, optimizer_s2, max_train_steps_s2, accelerator)

    # Re-prepare for stage 2 (transformer_raw is now the fused+new-LoRA model)
    transformer_raw, optimizer_s2, train_dataloader, lr_scheduler_s2 = accelerator.prepare(
        transformer_raw, optimizer_s2, train_dataloader, lr_scheduler_s2)

    # Checkpoint after stage-2 reinit
    accelerator.save_state(os.path.join(args.output_dir, f"checkpoint-{global_step}-stage2-init"))

    if args.seed is not None:
        set_seed(args.seed)

    logger.info(f"Stage 2 training for {max_train_steps_s2} steps "
                f"(timestep ranges {stage2_ranges})")
    # There's differences between None and [(0, 999)] in valid_index_ranges.
    # None indicates using shifting scheduling default by sd3, 0, 999 indicates using uniform scheduling.
    global_step = run_training_loop(
        transformer_raw, optimizer_s2, lr_scheduler_s2, train_dataloader,
        accelerator, args, noise_scheduler_copy,
        prompt_embeds, pooled_prompt_embeds,
        weight_dtype,
        vae if latents_cache is None else None,
        vae_shift_factor, vae_scale_factor,
        all_params_s2, latents_cache,
        global_step=global_step, max_steps=max_train_steps_s2,
        stage_name="stage2",
        valid_index_ranges=None,
        save_ckpt_list=save_ckpt_list,
    )

    # ------------------------------------------------------------------
    # Save final stage-2 LoRA weights
    # ------------------------------------------------------------------
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        tr_final = unwrap_model(transformer_raw)
        if args.upcast_before_saving:
            tr_final = tr_final.to(torch.float32)
        else:
            tr_final = tr_final.to(weight_dtype)
        StableDiffusion3Pipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=get_peft_model_state_dict(tr_final),
        )
        logger.info(f"Final stage-2 LoRA saved to {args.output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
