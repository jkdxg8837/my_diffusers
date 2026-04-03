import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
os.environ["MODEL_NAME"] = "stabilityai/stable-diffusion-3-medium-diffusers"
os.environ["INSTANCE_DIR"] = "./dog"
os.environ["OUTPUT_DIR"] = "./dog-lora-pmd-reranked"

# ---------------------------------------------------------------------------
# Two-stage hyperparameters
# ---------------------------------------------------------------------------
stage1_t_low   = 200     # stage 1 uses timestep indices [0, stage1_t_low] ∪ [stage1_t_high, 999]
stage1_t_high  = 800     # stage 2 uses [stage1_t_low, stage1_t_high]
stage1_steps   = 500     # training steps for stage 1
max_train_steps = 500    # training steps for stage 2

# LoRA / init
rank         = 32
stable_gamma = 49
direction    = "LoRA-One"
svd_algo     = "pmd"   # "svd" (standard) or "pmd" (sparse PMD)
pmd_c_u      = 1.0     # PMD L1 bound on left vectors  (only used when svd_algo="pmd")
pmd_c_v      = 1.0     # PMD L1 bound on right vectors (only used when svd_algo="pmd")

# ---------------------------------------------------------------------------
# Training command
# ---------------------------------------------------------------------------
cmd = [
    "accelerate", "launch",
    "--config_file", "../config/accelerate_config.yaml",
    "../train_dreambooth_lora_sd3_simple.py",
    "--pretrained_model_name_or_path", os.environ["MODEL_NAME"],
    "--instance_data_dir",             os.environ["INSTANCE_DIR"],
    "--output_dir",                    os.environ["OUTPUT_DIR"],
    "--instance_prompt",               "a photo of sks dog",
    "--resolution",                    "512",
    "--rank",                          str(rank),
    "--mixed_precision",               "fp16",
    "--train_batch_size",              "4",
    "--gradient_accumulation_steps",   "1",
    "--learning_rate",                 "4e-4",
    "--lr_scheduler",                  "constant",
    "--lr_warmup_steps",               "0",
    "--repeats",                       "15",
    "--seed",                          "0",
    # LoRA-One init
    "--init_config",                   "../config/gradient.yaml",
    "--reinit_strategy",               "all",
    "--stable_gamma",                  str(stable_gamma),
    "--direction",                     direction,
    "--svd_algo",                      svd_algo,
    "--pmd_c_u",                       str(pmd_c_u),
    "--pmd_c_v",                       str(pmd_c_v),
    # Two-stage split
    "--stage1_t_low",                  str(stage1_t_low),
    "--stage1_t_high",                 str(stage1_t_high),
    "--stage1_steps",                  str(stage1_steps),
    "--max_train_steps",               str(max_train_steps),
    "--stage1_iters",                  "500",
    "--stage2_iters",                  "50",
    # Logging
    "--report_to",                     "wandb",
]

# ---------------------------------------------------------------------------
# Eval command template
# ---------------------------------------------------------------------------
eval_cmd = [
    "accelerate", "launch",
    "--config_file", "../config/accelerate_config.yaml",
    "../eval_dreambooth_sd3.py",
    "--pretrained_model_name_or_path", os.environ["MODEL_NAME"],
    "--instance_data_dir",             os.environ["INSTANCE_DIR"],
    "--instance_prompt",               "a photo of sks dog",
    "--resolution",                    "512",
    "--rank",                          str(rank),
    "--train_batch_size",              "1",
    "--gradient_accumulation_steps",   "1",
    "--max_train_steps",               "2",
    "--learning_rate",                 "1e-4",
    "--lr_scheduler",                  "constant",
    "--lr_warmup_steps",               "0",
    "--lora_scale",                    "1.0",
    "--output_dir",                    os.environ["OUTPUT_DIR"],   # placeholder, overwritten below
]

out_dir = os.environ["OUTPUT_DIR"]

# ---------------------------------------------------------------------------
# Run training
# ---------------------------------------------------------------------------
subprocess.run(cmd, check=True)

# ---------------------------------------------------------------------------
# Remove named_grads files (large, not needed after reinit)
# ---------------------------------------------------------------------------
for grads_file in ["named_grads_stage1.pt", "named_grads_stage2.pt"]:
    grads_path = os.path.join(out_dir, grads_file)
    if os.path.exists(grads_path):
        os.remove(grads_path)
        print(f"Removed {grads_path}")

# ---------------------------------------------------------------------------
# Evaluate checkpoints saved during stage 2
# ---------------------------------------------------------------------------
eval_checkpoints = [
    "checkpoint-1",
    "checkpoint-10",
    "checkpoint-20",
    "checkpoint-50",
    # f"checkpoint-{stage1_steps}",          # first stage-2 step (= global_step at stage-2 start)
    "checkpoint-100",
    "checkpoint-200",
    "checkpoint-300",
    "checkpoint-400",
    # f"checkpoint-{stage1_steps + max_train_steps}",  # final
]

lora_scale = float(eval_cmd[eval_cmd.index("--lora_scale") + 1])

for ckpt in eval_checkpoints:
    ckpt_path = os.path.join(out_dir, ckpt)
    if os.path.isdir(ckpt_path):
        eval_cmd[eval_cmd.index("--output_dir") + 1] = ckpt_path
        subprocess.run(eval_cmd)
    else:
        print(f"Checkpoint not found, skipping: {ckpt_path}")

# ---------------------------------------------------------------------------
# Compute CLIP-T, CLIP-I, FID for every evaluated checkpoint
# ---------------------------------------------------------------------------
# Add workspace root to path so eval_metrics.py can be imported directly
workspace = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, workspace)

from eval_metrics import evaluate  # noqa: E402

metrics_rows = []
for ckpt in eval_checkpoints:
    ckpt_path = os.path.join(out_dir, ckpt)
    gen_dir = os.path.join(ckpt_path, f"output_scale{lora_scale}")
    if not os.path.isdir(gen_dir):
        print(f"No generated images for {ckpt}, skipping metrics.")
        continue
    results = evaluate(
        eval_dir=ckpt_path,
        instance_data_dir=os.environ["INSTANCE_DIR"],
        lora_scale=lora_scale,
        device="cuda",
        clip_model="openai/clip-vit-large-patch14",
    )
    metrics_rows.append(results)

# Print summary table
if metrics_rows:
    print(f"\n{'='*70}")
    print(f"{'Checkpoint':<40} {'CLIP-T':>8} {'CLIP-I':>8} {'FID':>10}")
    print("-" * 70)
    for r in metrics_rows:
        name = os.path.basename(r["eval_dir"])
        print(f"{name:<40} {r['clip_t']:>8.4f} {r['clip_i']:>8.4f} {r['fid']:>10.2f}")

    # Write CSV next to the output dir
    import csv
    csv_path = os.path.join(out_dir, "eval_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics_rows[0].keys())
        writer.writeheader()
        writer.writerows(metrics_rows)
    print(f"\nMetrics saved to {csv_path}")
