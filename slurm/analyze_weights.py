"""
Post-training analysis: compare weight evolution across LoRA baseline,
LoRA-One, and full finetuning.

Loads checkpoints at specified steps, computes:
  1. Per-method metrics vs pretrained base (L2, principal angle, relative L2)
  2. Cross-method comparison: full finetune vs (pretrain + LoRA) at each step
Saves results to JSON.
"""

import json
import os
import random
import sys

import torch
import numpy as np
from safetensors.torch import load_file as load_safetensors

# Add parent dir so we can import diffusers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from diffusers import SD3Transformer2DModel


BASE_OUTPUT_DIR = "./"
MODEL_NAME = "stabilityai/stable-diffusion-3-medium-diffusers"

CHECKPOINT_STEPS = [0, 1, 10, 20, 50, 100, 150, 200, 300, 400, 500]

MAX_MODULES = 100  # Sample at most this many modules to speed up analysis

METHODS = {
    "lora_baseline": {"type": "lora", "dir": os.path.join(BASE_OUTPUT_DIR, "dog-base")},
    "lora_one": {"type": "lora", "dir": os.path.join(BASE_OUTPUT_DIR, "dog-lora-one-gamma64")},
    "full_finetune": {"type": "full_ft", "dir": os.path.join(BASE_OUTPUT_DIR, "dog-fft")},
}

# Target modules to analyze
TARGET_MODULES = [
    "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
    "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
    "ff.net.0.proj", "ff.net.2", "norm_out.linear", "proj_out",
]


def load_base_weights(model_name):
    """Load base SD3 transformer weights."""
    print("Loading base SD3 transformer weights...")
    transformer = SD3Transformer2DModel.from_pretrained(
        model_name, subfolder="transformer"
    )
    base_weights = {}
    for name, param in transformer.named_parameters():
        for target in TARGET_MODULES:
            if target in name and "weight" in name:
                base_weights[name] = param.data.clone()
                break
    del transformer
    torch.cuda.empty_cache()
    print(f"  Loaded {len(base_weights)} base weight tensors.")
    return base_weights


def load_lora_checkpoint(checkpoint_dir):
    """Load LoRA weights from a checkpoint directory and return lora_A/lora_B pairs."""
    # Try the checkpoint subdir first, then the main dir
    lora_path = os.path.join(checkpoint_dir, "pytorch_lora_weights.safetensors")
    if not os.path.exists(lora_path):
        return None

    lora_state = load_safetensors(lora_path)

    # Organize by module: collect lora_A and lora_B for each base weight
    lora_pairs = {}
    for key, tensor in lora_state.items():
        # Keys look like: transformer.transformer_blocks.0.attn.to_q.lora_A.weight
        # Strip "transformer." prefix
        clean_key = key.replace("transformer.", "", 1)
        if ".lora_A." in clean_key:
            base_key = clean_key.replace(".lora_A.weight", ".weight")
            lora_pairs.setdefault(base_key, {})["A"] = tensor
        elif ".lora_B." in clean_key:
            base_key = clean_key.replace(".lora_B.weight", ".weight")
            lora_pairs.setdefault(base_key, {})["B"] = tensor

    return lora_pairs


def load_full_ft_checkpoint(checkpoint_dir):
    """Load full finetune trainable weights."""
    weights_path = os.path.join(checkpoint_dir, "trainable_weights.pt")
    if not os.path.exists(weights_path):
        return None
    return torch.load(weights_path, map_location="cpu")


def compute_effective_weight_lora(base_weight, lora_A, lora_B, alpha=None, rank=None):
    """Compute W_base + B @ A (effective weight after LoRA)."""
    # lora_A: (rank, in_features), lora_B: (out_features, rank)
    # For LoRA with alpha=rank, scaling = alpha/rank = 1
    delta = lora_B.float() @ lora_A.float()
    return base_weight.float() + delta


def compute_metrics(base_weight, trained_weight):
    """Compute L2 norm of delta, principal angle (degrees), and relative L2."""
    base_flat = base_weight.float().flatten()
    trained_flat = trained_weight.float().flatten()
    delta = trained_flat - base_flat

    l2_norm = torch.norm(delta).item()
    base_norm = torch.norm(base_flat).item()
    relative_l2 = l2_norm / base_norm if base_norm > 0 else 0.0

    cos_sim = torch.nn.functional.cosine_similarity(
        base_flat.unsqueeze(0), trained_flat.unsqueeze(0)
    ).item()
    # Clamp to [-1, 1] to avoid numerical issues with arccos

    import numpy as np
    from scipy.linalg import subspace_angles, orth

    # basis_A = orth(base_weight)
    # basis_B = orth(trained_weight)
    angles_rad = subspace_angles(base_weight, trained_weight)
    principal_angle = np.degrees(angles_rad[0]) if len(angles_rad) > 0 else 0.0

    return {
        "l2_norm": l2_norm,
        "base_norm": base_norm,
        "relative_l2": relative_l2,
        "principal_angle_deg": principal_angle,
    }


def load_loss_curve(method_dir):
    """Load loss_curve.jsonl from a method directory."""
    loss_path = os.path.join(method_dir, "loss_curve.jsonl")
    if not os.path.exists(loss_path):
        return []
    losses = []
    with open(loss_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                losses.append(json.loads(line))
    return losses


def get_checkpoint_dir(method_info, step, max_step=500):
    """Return the checkpoint directory for a given method and step."""
    if step == max_step:
        return method_info["dir"]
    return os.path.join(method_info["dir"], f"checkpoint-{step}")


def get_effective_weights(base_weights, method_info, step, max_step=500):
    """Load a checkpoint and return effective weights (base + delta) for all modules.

    For LoRA: effective = base + B @ A
    For full finetune: effective = trained weight directly
    Returns dict {module_key: effective_weight_tensor} or None if checkpoint missing.
    """
    checkpoint_dir = get_checkpoint_dir(method_info, step, max_step)

    if not os.path.exists(checkpoint_dir):
        print(f"  Checkpoint dir not found: {checkpoint_dir}")
        return None

    effective = {}

    if method_info["type"] == "lora":
        lora_pairs = load_lora_checkpoint(checkpoint_dir)
        if lora_pairs is None:
            print(f"  No LoRA weights found in {checkpoint_dir}")
            return None

        for base_key, base_w in base_weights.items():
            if base_key in lora_pairs:
                pair = lora_pairs[base_key]
                if "A" in pair and "B" in pair:
                    effective[base_key] = compute_effective_weight_lora(
                        base_w, pair["A"], pair["B"]
                    )

    elif method_info["type"] == "full_ft":
        trained_state = load_full_ft_checkpoint(checkpoint_dir)
        if trained_state is None:
            print(f"  No trainable_weights.pt found in {checkpoint_dir}")
            return None

        for base_key, base_w in base_weights.items():
            if base_key in trained_state:
                effective[base_key] = trained_state[base_key].float()

    return effective if effective else None


def aggregate_metrics(per_module):
    """Compute aggregate metrics from per-module metrics dict."""
    total_l2 = sum(m["l2_norm"] ** 2 for m in per_module.values())
    total_base_norm = sum(m["base_norm"] ** 2 for m in per_module.values())
    angles = [m["principal_angle_deg"] for m in per_module.values()]

    return {
        "total_l2_norm": float(np.sqrt(total_l2)),
        "total_base_norm": float(np.sqrt(total_base_norm)),
        "total_relative_l2": float(np.sqrt(total_l2) / np.sqrt(total_base_norm)) if total_base_norm > 0 else 0.0,
        "mean_principal_angle_deg": float(np.mean(angles)) if angles else 0.0,
        "num_modules": len(per_module),
    }


def analyze_checkpoint(base_weights, method_name, method_info, step):
    """Analyze a single checkpoint for a given method (vs pretrained base)."""
    effective = get_effective_weights(base_weights, method_info, step)
    if effective is None:
        return None

    per_module = {}
    matched = [(k, v) for k, v in base_weights.items() if k in effective]
    total = len(matched)
    for i, (base_key, base_w) in enumerate(matched):
        if (i + 1) % 10 == 0 or i == 0 or (i + 1) == total:
            print(f"    [{method_name} step {step}] module {i+1}/{total}: {base_key}")
        per_module[base_key] = compute_metrics(base_w, effective[base_key])

    return {"per_module": per_module, "aggregate": aggregate_metrics(per_module)}


def compare_methods_at_step(base_weights, method_a_info, method_b_info, step):
    """Compare effective weights of two methods at the same step.

    Computes metrics treating method_a's effective weight as the 'reference'
    and method_b's effective weight as the 'trained' weight.
    Returns aggregate and per-module metrics, or None if either checkpoint is missing.
    """
    eff_a = get_effective_weights(base_weights, method_a_info, step)
    eff_b = get_effective_weights(base_weights, method_b_info, step)

    if eff_a is None or eff_b is None:
        return None

    per_module = {}
    common_keys = sorted(set(eff_a.keys()) & set(eff_b.keys()))
    total = len(common_keys)
    for i, key in enumerate(common_keys):
        if (i + 1) % 10 == 0 or i == 0 or (i + 1) == total:
            print(f"    [cross-compare step {step}] module {i+1}/{total}: {key}")
        per_module[key] = compute_metrics(eff_a[key], eff_b[key])

    if not per_module:
        return None

    return {"per_module": per_module, "aggregate": aggregate_metrics(per_module)}


def main():
    # Load base weights
    base_weights = load_base_weights(MODEL_NAME)
    print(1)
    # Subsample modules if there are too many
    all_keys = sorted(base_weights.keys())
    if len(all_keys) > MAX_MODULES:
        print(f"Subsampling {MAX_MODULES} / {len(all_keys)} modules for speed...")
        random.seed(42)
        sampled_keys = sorted(random.sample(all_keys, MAX_MODULES))
        base_weights = {k: base_weights[k] for k in sampled_keys}
    else:
        print(f"Using all {len(all_keys)} modules.")

    results = {}

    # # --- Part 1: Per-method analysis (each method vs pretrained base) ---
    # for method_name, method_info in METHODS.items():
    #     print(f"\nAnalyzing method: {method_name}")
    #     method_results = {"checkpoints": {}, "loss_curve": []}

    #     for step in CHECKPOINT_STEPS:
    #         print(f"  Step {step}...")
    #         checkpoint_result = analyze_checkpoint(base_weights, method_name, method_info, step)
    #         if checkpoint_result is not None:
    #             # Store only aggregate for the main results (per_module can be huge)
    #             method_results["checkpoints"][str(step)] = checkpoint_result["aggregate"]

    #     # Load loss curve
    #     method_results["loss_curve"] = load_loss_curve(method_info["dir"])
    #     print(f"  Loaded {len(method_results['loss_curve'])} loss entries.")

    #     results[method_name] = method_results

    # --- Part 2: Cross-method comparison at each step ---
    # Compare full_finetune vs lora_baseline and full_finetune vs lora_one
    cross_comparisons = {
        "full_finetune_vs_lora_baseline": ("full_finetune", "lora_baseline"),
        "full_finetune_vs_lora_one": ("full_finetune", "lora_one"),
        "lora_baseline_vs_lora_one": ("lora_baseline", "lora_one"),
    }

    results["cross_method"] = {}
    for comp_name, (method_a_name, method_b_name) in cross_comparisons.items():
        print(f"\nCross-method comparison: {comp_name}")
        comp_results = {}

        for step in CHECKPOINT_STEPS:
            print(f"  Step {step}...")
            cmp = compare_methods_at_step(
                base_weights,
                METHODS[method_a_name],
                METHODS[method_b_name],
                step,
            )
            if cmp is not None:
                comp_results[str(step)] = cmp["aggregate"]

        results["cross_method"][comp_name] = comp_results

    # --- Print summary ---
    print("\n" + "=" * 80)
    print("SUMMARY: Per-method metrics vs pretrained base")
    print("=" * 80)

    for method_name in METHODS:
        print(f"\n--- {method_name} ---")
        if method_name in results:
            for step_str, agg in sorted(results[method_name]["checkpoints"].items(), key=lambda x: int(x[0])):
                print(f"  Step {step_str:>4s}: L2={agg['total_l2_norm']:.6f}  "
                      f"RelL2={agg['total_relative_l2']:.6f}  "
                      f"Angle={agg['mean_principal_angle_deg']:.4f}deg")

    print("\n" + "=" * 80)
    print("SUMMARY: Cross-method weight comparison at each step")
    print("=" * 80)

    for comp_name, comp_results in results["cross_method"].items():
        print(f"\n--- {comp_name} ---")
        for step_str, agg in sorted(comp_results.items(), key=lambda x: int(x[0])):
            print(f"  Step {step_str:>4s}: L2={agg['total_l2_norm']:.6f}  "
                  f"RelL2={agg['total_relative_l2']:.6f}  "
                  f"Angle={agg['mean_principal_angle_deg']:.4f}deg")

    # Print loss comparison at step 0
    print("\n" + "=" * 80)
    print("LOSS AT STEP 0 (should be identical for LoRA baseline & full finetune):")
    print("=" * 80)
    for method_name in METHODS:
        if method_name in results and results[method_name]["loss_curve"]:
            step0_losses = [e for e in results[method_name]["loss_curve"] if e["step"] == 0]
            if step0_losses:
                print(f"  {method_name}: {step0_losses[0]['loss']:.8f}")

    # Save results
    output_path = os.path.join(BASE_OUTPUT_DIR, "weight_comparison_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
