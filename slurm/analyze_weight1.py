#!/usr/bin/env python
"""
Analyze principal angles between weight-matrix column-spaces.

Comparisons:
  Part 1 — (pretrain + lora_init_from_fft) vs fft_weights
      For each init method (lora_one, lora_ga, svd) at each checkpoint step.
      Measures how close the LoRA-reconstructed weight is to the true FFT weight.

  Part 2 — (pretrain + lora_init_from_fft) vs (pretrain + lora_trained)
      For each init method at each step.
      Measures how close the zero-shot LoRA init is to a separately trained LoRA.
      (LORA_TRAINED_PATH must be set; skipped if None.)

Uses scipy.linalg.subspace_angles on each per-module weight matrix.
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
LORA_INIT_BASE = "/home/j/jiayang_gu/workspace/diffusers/slurm/eval_fft_lora_init"

# Path to a separately-trained LoRA checkpoint (.pt or .safetensors).
# Set to None to skip Part 2.
LORA_TRAINED_PATH = "/home/j/jiayang_gu/workspace/diffusers/slurm/dog-base"  # TODO: fill in when available

CHECKPOINT_STEPS = [0,10, 20, 50, 100, 200, 300, 400]
INIT_METHODS = ["lora_one", "lora_ga", "svd"]

TARGET_MODULES = [
    "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
    "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
    "ff.net.0.proj", "ff.net.2", "norm_out.linear", "proj_out",
]

OUTPUT_FILE = os.path.join(
    os.path.dirname(__file__), "principal_angle_results2.json"
)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_pretrained_weights(model_name):
    """Load pretrained SD3 transformer weights for target modules only."""
    print("Loading pretrained transformer weights...")
    transformer = SD3Transformer2DModel.from_pretrained(
        model_name, subfolder="transformer"
    )
    weights = {}
    for name, param in transformer.named_parameters():
        for target in TARGET_MODULES:
            if target in name and name.endswith(".weight"):
                weights[name] = param.data.clone().cpu().float()
                break
    del transformer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Loaded {len(weights)} weight tensors.")
    return weights


def load_fft_weights(step):
    """Load trainable_weights.pt for a given FFT checkpoint step.

    Returns only target-module keys, already on CPU float32.
    """
    path = os.path.join(FFT_DIR, f"checkpoint-{step}", "trainable_weights.pt")
    if not os.path.exists(path):
        print(f"  FFT weights not found: {path}")
        return None
    print(f"  Loading FFT weights: {path}")
    raw = torch.load(path, map_location="cpu", weights_only=False)
    filtered = {}
    for key, tensor in raw.items():
        for target in TARGET_MODULES:
            if target in key and key.endswith(".weight"):
                filtered[key] = tensor.float()
                break
    del raw
    gc.collect()
    return filtered


def load_lora_init_weights(method_name, step):
    """Load LoRA A/B from eval_fft_lora_init/{method}/step-{step}/pytorch_lora_weights.pt.

    Returns {base_key: {"A": tensor, "B": tensor}}.
    """
    path = os.path.join(
        LORA_INIT_BASE, method_name, f"step-{step}", "pytorch_lora_weights.pt"
    )
    if not os.path.exists(path):
        print(f"  LoRA init weights not found: {path}")
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)
    return _parse_lora_state(state)


def load_lora_trained_weights(path):
    """Load a separately-trained LoRA checkpoint (.pt or .safetensors).

    Returns {base_key: {"A": tensor, "B": tensor}} or None.
    """
    if path is None or not os.path.exists(path):
        return None
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(path)
    else:
        state = torch.load(path, map_location="cpu", weights_only=False)
    return _parse_lora_state(state)


def _parse_lora_state(state_dict):
    """Parse a LoRA state dict into {base_key: {"A": tensor, "B": tensor}}.

    Handles key formats:
      - transformer_blocks.0.attn.to_q.lora_A.weight          (get_peft_model_state_dict)
      - base_model.model.transformer_blocks.0.attn.to_q.lora_A.default.weight  (raw peft)
      - transformer.transformer_blocks.0.attn.to_q.lora_A.weight               (safetensors)
    """
    pairs = {}
    for key, tensor in state_dict.items():
        clean = key
        # strip common prefixes
        for prefix in ("base_model.model.", "transformer."):
            if clean.startswith(prefix):
                clean = clean[len(prefix):]

        if ".lora_A." in clean:
            base_key = clean.split(".lora_A.")[0] + ".weight"
            pairs.setdefault(base_key, {})["A"] = tensor.cpu().float()
        elif ".lora_B." in clean:
            base_key = clean.split(".lora_B.")[0] + ".weight"
            pairs.setdefault(base_key, {})["B"] = tensor.cpu().float()
    return pairs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def effective_weight(base_w, lora_A, lora_B):
    """pretrain + B @ A  (alpha == rank → scaling = 1)."""
    return base_w + lora_B @ lora_A


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
    """Compute principal angles for every module that exists in both dicts.

    weights_a, weights_b: {module_key: float32 tensor}
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

    results = {"lora_init_vs_fft": {}, "lora_init_vs_lora_trained": {}}

    # ---------------------------------------------------------------
    # Part 1: (pretrain + lora_init) vs fft_weights
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Part 1: principal angles — (pretrain + lora_init) vs fft_weights")
    print("=" * 70)

    for step in CHECKPOINT_STEPS:
        fft = load_fft_weights(step)
        if fft is None:
            continue

        for method in INIT_METHODS:
            print(f"\n  Method={method}  Step={step}")
            lora_pairs = load_lora_init_weights(method, step)
            if lora_pairs is None:
                continue

            # Build effective weights for every matched module
            eff_lora = {}
            for key, base_w in pretrained.items():
                if key in lora_pairs and "A" in lora_pairs[key] and "B" in lora_pairs[key]:
                    eff_lora[key] = effective_weight(
                        base_w, lora_pairs[key]["A"], lora_pairs[key]["B"]
                    )

            per_mod, agg = compare_per_module(eff_lora, fft, label=f"{method}/step{step}")
            results["lora_init_vs_fft"].setdefault(method, {})[str(step)] = {
                "aggregate": agg,
                "per_module": per_mod,
            }
            print(f"    => {agg['num_modules']} modules  "
                  f"MeanMaxAngle={agg['mean_max_deg']:.4f}°  "
                  f"MeanMeanAngle={agg['mean_mean_deg']:.4f}°")

        del fft
        gc.collect()

    # ---------------------------------------------------------------
    # Part 2: (pretrain + lora_init) vs (pretrain + lora_trained)
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Part 2: principal angles — (pretrain + lora_init) vs (pretrain + lora_trained)")
    print("=" * 70)

    trained_pairs = load_lora_trained_weights(LORA_TRAINED_PATH)
    if trained_pairs is None:
        print("  LORA_TRAINED_PATH not set or file missing — skipping Part 2.")
    else:
        # Build effective trained-LoRA weights once
        eff_trained = {}
        for key, base_w in pretrained.items():
            if key in trained_pairs and "A" in trained_pairs[key] and "B" in trained_pairs[key]:
                eff_trained[key] = effective_weight(
                    base_w, trained_pairs[key]["A"], trained_pairs[key]["B"]
                )

        for step in CHECKPOINT_STEPS:
            for method in INIT_METHODS:
                print(f"\n  Method={method}  Step={step}")
                lora_pairs = load_lora_init_weights(method, step)
                if lora_pairs is None:
                    continue

                eff_init = {}
                for key, base_w in pretrained.items():
                    if key in lora_pairs and "A" in lora_pairs[key] and "B" in lora_pairs[key]:
                        eff_init[key] = effective_weight(
                            base_w, lora_pairs[key]["A"], lora_pairs[key]["B"]
                        )

                per_mod, agg = compare_per_module(
                    eff_init, eff_trained, label=f"{method}/step{step}"
                )
                results["lora_init_vs_lora_trained"].setdefault(method, {})[str(step)] = {
                    "aggregate": agg,
                    "per_module": per_mod,
                }
                print(f"    => {agg['num_modules']} modules  "
                      f"MeanMaxAngle={agg['mean_max_deg']:.4f}°  "
                      f"MeanMeanAngle={agg['mean_mean_deg']:.4f}°")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY — Part 1: (pretrain + lora_init) vs fft_weights")
    print("=" * 70)
    for method in INIT_METHODS:
        print(f"\n  --- {method} ---")
        for step_s, data in sorted(
            results["lora_init_vs_fft"].get(method, {}).items(),
            key=lambda x: int(x[0]),
        ):
            a = data["aggregate"]
            print(f"    Step {step_s:>4s}: "
                  f"MeanMaxAngle={a['mean_max_deg']:.4f}°  "
                  f"MeanMeanAngle={a['mean_mean_deg']:.4f}°  "
                  f"({a['num_modules']} modules)")

    if any(results["lora_init_vs_lora_trained"].values()):
        print("\n" + "=" * 70)
        print("SUMMARY — Part 2: (pretrain + lora_init) vs (pretrain + lora_trained)")
        print("=" * 70)
        for method in INIT_METHODS:
            print(f"\n  --- {method} ---")
            for step_s, data in sorted(
                results["lora_init_vs_lora_trained"].get(method, {}).items(),
                key=lambda x: int(x[0]),
            ):
                a = data["aggregate"]
                print(f"    Step {step_s:>4s}: "
                      f"MeanMaxAngle={a['mean_max_deg']:.4f}°  "
                      f"MeanMeanAngle={a['mean_mean_deg']:.4f}°  "
                      f"({a['num_modules']} modules)")

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
