#!/usr/bin/env python
"""
Compute principal angles between pretrained SD3 transformer weights
and fine-tuned (FFT) weights at each checkpoint step.

For every target module, measures how much the column-space of the
weight matrix has rotated during fine-tuning.
"""

import gc
import json
import os
import sys

import numpy as np
import torch
from scipy.linalg import subspace_angles

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from diffusers import SD3Transformer2DModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "stabilityai/stable-diffusion-3-medium-diffusers"
FFT_DIR = "/home/j/jiayang_gu/workspace/diffusers/slurm/dog-weight-fft-lowerlr"

CHECKPOINT_STEPS = [0, 1, 10, 20, 50, 100, 150, 200, 300, 400]

TARGET_MODULES = [
    "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
    "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
    "ff.net.0.proj", "ff.net.2", "norm_out.linear", "proj_out",
]

OUTPUT_FILE = os.path.join(
    os.path.dirname(__file__), "fft_principal_angle_results.json"
)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def is_target_weight(name):
    """Check if a parameter name matches a target module weight."""
    if not name.endswith(".weight"):
        return False
    return any(target in name for target in TARGET_MODULES)


def load_pretrained_weights(model_name):
    """Load pretrained SD3 transformer weights for target modules only."""
    print("Loading pretrained transformer weights...")
    transformer = SD3Transformer2DModel.from_pretrained(
        model_name, subfolder="transformer"
    )
    weights = {}
    for name, param in transformer.named_parameters():
        if is_target_weight(name):
            weights[name] = param.data.clone().cpu().float()
    del transformer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Loaded {len(weights)} weight tensors.")
    return weights


def load_fft_weights(step):
    """Load trainable_weights.pt for a given FFT checkpoint step."""
    path = os.path.join(FFT_DIR, f"checkpoint-{step}", "trainable_weights.pt")
    if not os.path.exists(path):
        print(f"  FFT weights not found: {path}")
        return None
    print(f"  Loading FFT weights: {path}")
    raw = torch.load(path, map_location="cpu", weights_only=False)
    filtered = {}
    for key, tensor in raw.items():
        if is_target_weight(key):
            filtered[key] = tensor.float()
    del raw
    gc.collect()
    return filtered


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def principal_angles_deg(mat_a, mat_b):
    """Principal angles (degrees) between column-spaces of two matrices."""
    angles = subspace_angles(mat_a.numpy(), mat_b.numpy())
    if len(angles) == 0:
        return {"max": 0.0, "mean": 0.0, "min": 0.0}
    return {
        "max": float(np.degrees(angles[0])),
        "mean": float(np.degrees(np.mean(angles))),
        "min": float(np.degrees(angles[-1])),
    }


def compare_per_module(weights_a, weights_b, label=""):
    """Compute principal angles for every module present in both dicts.

    Returns (per_module_dict, aggregate_dict).
    """
    common = sorted(set(weights_a) & set(weights_b))
    per_module = {}
    for i, key in enumerate(common):
        if (i + 1) % 20 == 0 or i == 0 or (i + 1) == len(common):
            print(f"    {label} module {i+1}/{len(common)}: {key}")
        per_module[key] = principal_angles_deg(weights_a[key], weights_b[key])

    if not per_module:
        return {}, {"mean_max_deg": 0.0, "mean_mean_deg": 0.0, "num_modules": 0}

    agg = {
        "mean_max_deg": float(np.mean([v["max"] for v in per_module.values()])),
        "mean_mean_deg": float(np.mean([v["mean"] for v in per_module.values()])),
        "num_modules": len(per_module),
    }
    return per_module, agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pretrained = load_pretrained_weights(MODEL_NAME)

    results = {}

    print("\n" + "=" * 70)
    print("Principal angles: pretrained vs FFT weights")
    print("=" * 70)

    for step in CHECKPOINT_STEPS:
        print(f"\n--- Checkpoint step {step} ---")
        fft = load_fft_weights(step)
        if fft is None:
            continue

        per_mod, agg = compare_per_module(pretrained, fft, label=f"step{step}")
        results[str(step)] = {
            "aggregate": agg,
            "per_module": per_mod,
        }
        print(f"  => {agg['num_modules']} modules  "
              f"MeanMaxAngle={agg['mean_max_deg']:.4f} deg  "
              f"MeanMeanAngle={agg['mean_mean_deg']:.4f} deg")

        del fft
        gc.collect()

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY — pretrained vs FFT weights")
    print("=" * 70)
    for step_s, data in sorted(results.items(), key=lambda x: int(x[0])):
        a = data["aggregate"]
        print(f"  Step {step_s:>4s}: "
              f"MeanMaxAngle={a['mean_max_deg']:.4f} deg  "
              f"MeanMeanAngle={a['mean_mean_deg']:.4f} deg  "
              f"({a['num_modules']} modules)")

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
