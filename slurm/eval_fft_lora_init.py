#!/usr/bin/env python
"""
Evaluate LoRA initialization from full-finetune (FFT) weight deltas.

For each FFT checkpoint step, applies three initialization methods:
  - LoRA-One: SVD of -(fft - pretrained) delta
  - LoRA-GA:  SVD of delta, secondary singular vectors
  - SVD:      direct low-rank SVD of fft weights

No training is performed.  Images are generated for evaluation only.
"""

import argparse
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
from diffusers.training_utils import free_memory
from lora_one_utils import reinit_lora_from_fft

# ---------------------------------------------------------------------------
# Configuration  (edit these as needed)
# ---------------------------------------------------------------------------
MODEL_NAME = "stabilityai/stable-diffusion-3-medium-diffusers"
FFT_DIR = "/home/j/jiayang_gu/workspace/diffusers/slurm/dog-weight-fft-lowerlr"                 # directory that contains checkpoint-{step}/
OUTPUT_BASE = "./eval_fft_lora_init"  # results go here
RANK = 32
SEED = 42
NUM_IMAGES = 5

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=RANK, help="LoRA rank")
    parser.add_argument("--scale_factor", type=float, default=1.0, help="Scale factor for delta in gradient init mode")
    parser.add_argument("--rank_threshold", type=float, default=None,
                        help="Energy threshold (0-1) for rank-95 filtering. "
                             "Only reinit modules whose rank_t at this threshold <= LoRA rank.")
    return parser.parse_args()

# Steps to evaluate.  500 = final output dir (no checkpoint- prefix).
CHECKPOINT_STEPS = [1, 10, 100, 200, 400]
# CHECKPOINT_STEPS = [0, 1, 10, 20, 50]
MAX_STEP = 500

unique_token = "sks"
class_token = "dog"
PROMPT_LIST = [
    f"a {unique_token} {class_token} in the jungle",
    f"a {unique_token} {class_token} in the snow",
    f"a {unique_token} {class_token} on the beach",
    f"a {unique_token} {class_token} on a cobblestone street",
    f"a {unique_token} {class_token} on top of pink fabric",
    f"a {unique_token} {class_token} on top of a wooden floor",
]

TARGET_MODULES = [
    "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
    "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
    "ff.net.0.proj", "ff.net.2", "norm_out.linear", "proj_out",
]

INIT_CONFIGS = {
    "lora_one": {
        "mode": "gradient",
        "direction": "LoRA-One",
        "scale": "stable",
        "stable_gamma": 36,
        "dtype": "fp32",
        "lora_module": "all",
    },
    # "lora_ga": {
    #     "mode": "gradient",
    #     "direction": "LoRA-GA",
    #     "scale": "stable",
    #     "stable_gamma": 36,
    #     "dtype": "fp32",
    #     "lora_module": "all",
    # },
    # "svd": {
    #     "mode": "svd",
    #     "scale": "default",
    #     "dtype": "fp32",
    #     "lora_module": "all",
    # },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_fft_weights_path(fft_dir, step, max_step=500):
    if step == max_step:
        return os.path.join(fft_dir, "trainable_weights.pt")
    return os.path.join(fft_dir, f"checkpoint-{step}", "trainable_weights.pt")


def generate_images(pipeline, output_dir, seed, num_images):
    os.makedirs(output_dir, exist_ok=True)
    device = pipeline.device
    generator = (
        torch.Generator(device=device).manual_seed(seed)
        if seed is not None
        else None
    )
    for prompt_idx, prompt in enumerate(PROMPT_LIST):
        save_dir = os.path.join(output_dir, str(prompt_idx))
        os.makedirs(save_dir, exist_ok=True)
        print(f"    [{prompt_idx}] {prompt}")

        with torch.autocast(device_type="cuda"):
            for img_idx in range(num_images):
                image = pipeline(prompt=prompt, generator=generator).images[0]
                image.save(os.path.join(save_dir, f"{img_idx}.png"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    rank = args.rank
    scale_factor = args.scale_factor
    rank_threshold = args.rank_threshold
    output_base = OUTPUT_BASE
    if rank != RANK:
        output_base = f"{output_base}_rank{rank}"
    if scale_factor != 1.0:
        output_base = f"{output_base}_scale{scale_factor}"
    if rank_threshold is not None:
        output_base = f"{output_base}_rt{rank_threshold}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16

    print(f"Running with rank={rank}, scale_factor={scale_factor}, rank_threshold={rank_threshold}, output -> {output_base}")

    # ------------------------------------------------------------------
    # Load pipeline once (VAE + text encoders stay in memory throughout)
    # ------------------------------------------------------------------
    print("Loading base pipeline...")
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        MODEL_NAME, torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    # Cache pretrained transformer weights on CPU for delta computation
    print("Caching pretrained transformer weights...")
    pretrained_state = {}
    for n, p in pipeline.transformer.named_parameters():
        pretrained_state[n] = p.data.clone().cpu()

    total = len(INIT_CONFIGS) * len(CHECKPOINT_STEPS)
    current = 0

    for method_name, init_config_template in INIT_CONFIGS.items():
        for step in CHECKPOINT_STEPS:
            current += 1
            print(f"\n{'=' * 60}")
            print(f"[{current}/{total}] Method: {method_name}  Step: {step}  Rank: {rank}")
            print("=" * 60)

            # Check FFT weights exist
            fft_path = get_fft_weights_path(FFT_DIR, step, MAX_STEP)
            if not os.path.exists(fft_path):
                print(f"  FFT weights not found: {fft_path}, skipping.")
                continue

            # Reload fresh transformer (avoids residual LoRA / weight-offset
            # contamination from previous iteration)
            print("  Loading fresh transformer...")
            transformer = SD3Transformer2DModel.from_pretrained(
                MODEL_NAME, subfolder="transformer", torch_dtype=weight_dtype,
            )

            # Add LoRA adapter with random init (will be overwritten by reinit)
            lora_config = LoraConfig(
                r=rank,
                lora_alpha=rank,
                init_lora_weights="gaussian",
                target_modules=TARGET_MODULES,
            )
            transformer.add_adapter(lora_config)

            # Load FFT weights
            print(f"  Loading FFT weights from {fft_path}")
            fft_state = torch.load(fft_path, map_location="cpu")
            # Print key in fft_state
            # for key in fft_state.keys():
                # print(key)
            # Reinitialize LoRA A/B from FFT weight deltas
            init_config = copy.deepcopy(init_config_template)
            if scale_factor != 1.0:
                init_config["scale_factor"] = scale_factor
            print(f"  Reinitializing LoRA ({method_name})...")
            output_dir = os.path.join(output_base, method_name, f"step-{step}")
            reinit_lora_from_fft(
                transformer, init_config, fft_state, pretrained_state, output_dir,
                rank=rank, rank_threshold=rank_threshold
            )

            # Swap transformer into pipeline
            old_transformer = pipeline.transformer
            pipeline.transformer = transformer.to(device, dtype=weight_dtype)
            del old_transformer
            torch.cuda.empty_cache()

            # Generate evaluation images
            
            # print(f"  Generating images -> {output_dir}")
            # generate_images(pipeline, output_dir, SEED, NUM_IMAGES)

            # Save LoRA weights for debugging
            # lora_state = get_peft_model_state_dict(pipeline.transformer)
            # lora_save_path = os.path.join(output_dir, "pytorch_lora_weights.pt")
            # torch.save(lora_state, lora_save_path)
            # print(f"  Saved LoRA weights -> {lora_save_path}")

            print(f"  Done: {method_name} step {step}")

    del pipeline
    free_memory()
    print("\nAll evaluations complete.")


if __name__ == "__main__":
    main()
