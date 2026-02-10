#!/usr/bin/env python
# coding=utf-8
# Evaluation script for full-finetune SD3 DreamBooth weights.
# Loads only the trained modules from trainable_weights.pt and runs inference.

import argparse
import os

import torch
from PIL import Image

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from diffusers.training_utils import free_memory
from diffusers.utils import check_min_version

check_min_version("0.33.0.dev0")

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
    parser = argparse.ArgumentParser(description="Evaluate full-finetune SD3 DreamBooth weights.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained SD3 model.",
    )
    parser.add_argument(
        "--fft_weights_path",
        type=str,
        required=True,
        help="Path to trainable_weights.pt from full finetune training.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save generated images.",
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
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )

    # Load trained modules into the transformer
    trainable_state = torch.load(args.fft_weights_path, map_location="cpu")
    # Cast loaded weights to match pipeline dtype
    trainable_state = {k: v.to(weight_dtype) for k, v in trainable_state.items()}
    missing, unexpected = pipeline.transformer.load_state_dict(trainable_state, strict=False)
    print(f"Loaded {len(trainable_state)} trained parameter tensors from {args.fft_weights_path}")
    if unexpected:
        print(f"WARNING: unexpected keys: {unexpected}")

    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed is not None else None

    os.makedirs(args.output_dir, exist_ok=True)

    for prompt_idx, prompt in enumerate(prompt_list):
        save_dir = os.path.join(args.output_dir, str(prompt_idx))
        os.makedirs(save_dir, exist_ok=True)
        print(f"[{prompt_idx}] {prompt}")

        with torch.autocast(device_type="cuda"):
            for img_idx in range(args.num_validation_images):
                image = pipeline(prompt=prompt, generator=generator).images[0]
                image.save(os.path.join(save_dir, f"{img_idx}.png"))

    del pipeline
    free_memory()
    print("Done.")


if __name__ == "__main__":
    main()
