#!/usr/bin/env python
"""
Evaluate LoRA initialization from full-finetune (FFT) weight deltas for Flux.

For each FFT checkpoint step, applies three initialization methods:
  - LoRA-One: SVD of -(fft - pretrained) delta
  - LoRA-GA:  SVD of delta, secondary singular vectors
  - SVD:      direct low-rank SVD of fft weights

No training is performed.  Images are generated for evaluation only.
"""

import copy
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from peft import LoraConfig
from diffusers import FluxTransformer2DModel, FluxPipeline
from diffusers.training_utils import free_memory
from lora_one_utils import reinit_lora_from_fft

# ---------------------------------------------------------------------------
# Configuration  (edit these as needed)
# ---------------------------------------------------------------------------
MODEL_NAME = "black-forest-labs/FLUX.1-dev"
FFT_DIR = "./dog-flux-fft"                 # directory that contains checkpoint-{step}/
OUTPUT_BASE = "./eval_fft_lora_init_flux"  # results go here
RANK = 32
SEED = 42
NUM_IMAGES = 5

# Steps to evaluate.  500 = final output dir (no checkpoint- prefix).
CHECKPOINT_STEPS = [100, 200, 300, 400, 500]
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
    "ff.net.0.proj", "ff.net.2",
    "ff_context.net.0.proj", "ff_context.net.2",
]

INIT_CONFIGS = {
    "lora_one": {
        "mode": "gradient",
        "direction": "LoRA-One",
        "scale": "stable",
        "stable_gamma": 64,
        "dtype": "fp32",
        "lora_module": "all",
    },
    "lora_ga": {
        "mode": "gradient",
        "direction": "LoRA-GA",
        "scale": "stable",
        "stable_gamma": 64,
        "dtype": "fp32",
        "lora_module": "all",
    },
    "svd": {
        "mode": "svd",
        "scale": "default",
        "dtype": "fp32",
        "lora_module": "all",
    },
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

        # Pre-compute prompt embeddings outside autocast (T5 does not support fp16)
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
                prompt, prompt_2=prompt
            )

        with torch.autocast(device_type="cuda"):
            for img_idx in range(num_images):
                image = pipeline(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    generator=generator,
                ).images[0]
                image.save(os.path.join(save_dir, f"{img_idx}.png"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16

    # ------------------------------------------------------------------
    # Load pipeline once (VAE + text encoders stay in memory throughout)
    # ------------------------------------------------------------------
    print("Loading base pipeline...")
    pipeline = FluxPipeline.from_pretrained(
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
            print(f"[{current}/{total}] Method: {method_name}  Step: {step}")
            print("=" * 60)

            # Check FFT weights exist
            fft_path = get_fft_weights_path(FFT_DIR, step, MAX_STEP)
            if not os.path.exists(fft_path):
                print(f"  FFT weights not found: {fft_path}, skipping.")
                continue

            # Reload fresh transformer (avoids residual LoRA / weight-offset
            # contamination from previous iteration)
            print("  Loading fresh transformer...")
            transformer = FluxTransformer2DModel.from_pretrained(
                MODEL_NAME, subfolder="transformer", torch_dtype=weight_dtype,
            )

            # Add LoRA adapter with random init (will be overwritten by reinit)
            lora_config = LoraConfig(
                r=RANK,
                lora_alpha=RANK,
                init_lora_weights="gaussian",
                target_modules=TARGET_MODULES,
            )
            transformer.add_adapter(lora_config)

            # Load FFT weights
            print(f"  Loading FFT weights from {fft_path}")
            fft_state = torch.load(fft_path, map_location="cpu")

            # Reinitialize LoRA A/B from FFT weight deltas
            init_config = copy.deepcopy(init_config_template)
            print(f"  Reinitializing LoRA ({method_name})...")
            reinit_lora_from_fft(
                transformer, init_config, fft_state, pretrained_state,
            )

            # Swap transformer into pipeline
            old_transformer = pipeline.transformer
            pipeline.transformer = transformer.to(device, dtype=weight_dtype)
            del old_transformer
            torch.cuda.empty_cache()

            # Generate evaluation images
            output_dir = os.path.join(OUTPUT_BASE, method_name, f"step-{step}")
            print(f"  Generating images -> {output_dir}")
            generate_images(pipeline, output_dir, SEED, NUM_IMAGES)

            print(f"  Done: {method_name} step {step}")

    del pipeline
    free_memory()
    print("\nAll evaluations complete.")


if __name__ == "__main__":
    main()
