#!/usr/bin/env python
# coding=utf-8
"""
Evaluation script for Z-Image LoRA DreamBooth weights.
Loads LoRA weights from checkpoint directories and generates images for a prompt list.

Usage:
    python eval_dreambooth_zimage.py \
        --pretrained_model_name_or_path Tongyi-MAI/Z-Image \
        --output_dir ./trained-z-image-lora-prodigy \
        --seed 42 --num_validation_images 5 --mixed_precision bf16
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from diffusers import ZImagePipeline
from diffusers.training_utils import free_memory

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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Z-Image LoRA DreamBooth weights.")
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, required=True,
        help="Path to pretrained Z-Image model or HuggingFace model id.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Base output directory containing checkpoint-{step}/ subdirs. "
             "Generated images are saved under each checkpoint dir.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_validation_images", type=int, default=5)
    parser.add_argument(
        "--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"],
    )
    parser.add_argument(
        "--checkpoint_steps", type=str, default=None,
        help="Comma-separated list of checkpoint steps to evaluate. "
             "E.g. '0,10,100,200,300,400,500'. If not set, evaluates all "
             "checkpoint-* dirs plus the final output dir.",
    )
    parser.add_argument(
        "--max_step", type=int, default=500,
        help="The final training step (weights saved in output_dir root, not checkpoint-N).",
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=5.0,
    )
    return parser.parse_args()


def discover_checkpoints(output_dir, max_step):
    """Auto-discover checkpoint steps from the output directory."""
    steps = []
    if os.path.exists(output_dir):
        for name in os.listdir(output_dir):
            if name.startswith("checkpoint-"):
                try:
                    step = int(name.split("-")[1])
                    steps.append(step)
                except ValueError:
                    pass
    # Check if final weights exist
    if os.path.exists(os.path.join(output_dir, "pytorch_lora_weights.safetensors")):
        steps.append(max_step)
    return sorted(set(steps))


def get_lora_path(output_dir, step, max_step):
    """Return the directory containing LoRA weights for a given step."""
    if step == max_step:
        return output_dir
    return os.path.join(output_dir, f"checkpoint-{step}")


def main():
    args = parse_args()

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine which steps to evaluate
    if args.checkpoint_steps is not None:
        checkpoint_steps = [int(s.strip()) for s in args.checkpoint_steps.split(",")]
    else:
        checkpoint_steps = discover_checkpoints(args.output_dir, args.max_step)

    if not checkpoint_steps:
        print("No checkpoints found to evaluate.")
        return

    print(f"Will evaluate steps: {checkpoint_steps}")

    for step in checkpoint_steps:
        lora_dir = get_lora_path(args.output_dir, step, args.max_step)
        lora_file = os.path.join(lora_dir, "pytorch_lora_weights.safetensors")
        if not os.path.exists(lora_file):
            print(f"Skipping step {step}: {lora_file} not found")
            continue

        print(f"\n{'=' * 50}")
        print(f"Evaluating step {step}  ({lora_dir})")
        print("=" * 50)

        # Load fresh pipeline for each checkpoint to avoid contamination
        pipeline = ZImagePipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            torch_dtype=weight_dtype,
        )
        pipeline.load_lora_weights(lora_dir)
        pipeline = pipeline.to(device)
        pipeline.set_progress_bar_config(disable=True)

        generator = (
            torch.Generator(device=device).manual_seed(args.seed)
            if args.seed is not None
            else None
        )

        eval_dir = os.path.join(lora_dir, "eval_images")
        os.makedirs(eval_dir, exist_ok=True)

        for prompt_idx, prompt in enumerate(PROMPT_LIST):
            save_dir = os.path.join(eval_dir, str(prompt_idx))
            os.makedirs(save_dir, exist_ok=True)
            print(f"  [{prompt_idx}] {prompt}")

            for img_idx in range(args.num_validation_images):
                with torch.autocast(device_type="cuda"):
                    image = pipeline(
                        prompt=prompt,
                        guidance_scale=args.guidance_scale,
                        generator=generator,
                    ).images[0]
                image.save(os.path.join(save_dir, f"{img_idx}.png"))

        del pipeline
        free_memory()

    print("\nAll evaluations complete.")


if __name__ == "__main__":
    main()
