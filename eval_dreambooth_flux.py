#!/usr/bin/env python
# coding=utf-8
"""
Evaluation script for Flux LoRA DreamBooth weights.
Loads LoRA weights from output_dir and generates images for a prompt list.

Usage:
    python eval_dreambooth_flux.py \
        --pretrained_model_name_or_path black-forest-labs/FLUX.1-dev \
        --output_dir ./flux-dog/checkpoint-100 \
        --seed 42 --num_validation_images 5 --mixed_precision fp16
"""

import argparse
import os

import torch

from diffusers import FluxPipeline
from diffusers.training_utils import free_memory

unique_token = "sks"
class_token = "dog"
prompt_list = [
    'a {0} {1} in the jungle'.format(unique_token, class_token),
    'a {0} {1} in the snow'.format(unique_token, class_token),
    'a {0} {1} on the beach'.format(unique_token, class_token),
    'a {0} {1} on a cobblestone street'.format(unique_token, class_token),
    'a {0} {1} on top of pink fabric'.format(unique_token, class_token),
    'a {0} {1} on top of a wooden floor'.format(unique_token, class_token),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Flux LoRA DreamBooth weights.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained Flux model or HuggingFace model id.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory containing pytorch_lora_weights.safetensors. "
             "Generated images are also saved under this directory.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=5,
        help="Number of images to generate per prompt.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument(
        "--lora_scale",
        type=float,
        default=1.0,
        help="Scale factor applied to LoRA weights during inference.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine weight dtype
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the base pipeline
    print("Loading Flux pipeline...")
    pipeline = FluxPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )

    # Load LoRA weights
    print(f"Loading LoRA weights from {args.output_dir}")
    pipeline.load_lora_weights(args.output_dir)

    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    generator = (
        torch.Generator(device=device).manual_seed(args.seed)
        if args.seed is not None
        else None
    )

    eval_dir = os.path.join(args.output_dir, f"output_scale{args.lora_scale}")
    os.makedirs(eval_dir, exist_ok=True)

    for prompt_idx, prompt in enumerate(prompt_list):
        save_dir = os.path.join(eval_dir, str(prompt_idx))
        os.makedirs(save_dir, exist_ok=True)
        print(f"[{prompt_idx}] {prompt}")

        # Pre-compute prompt embeddings outside autocast (T5 does not support fp16 autocast)
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
                prompt, prompt_2=prompt
            )

        for img_idx in range(args.num_validation_images):
            with torch.autocast(device_type="cuda"):
                image = pipeline(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    generator=generator,
                    joint_attention_kwargs={"scale": args.lora_scale},
                ).images[0]
            image.save(os.path.join(save_dir, f"{img_idx}.png"))

    del pipeline
    free_memory()
    print("Done.")


if __name__ == "__main__":
    main()
