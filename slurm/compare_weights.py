import os
import subprocess
import sys

# Shared configuration
MODEL_NAME = "stabilityai/stable-diffusion-3-medium-diffusers"
INSTANCE_DIR = "./dog"
BASE_OUTPUT_DIR = "./weight_comparison"

SHARED_ARGS = [
    "--pretrained_model_name_or_path", MODEL_NAME,
    "--instance_data_dir", INSTANCE_DIR,
    "--mixed_precision", "fp16",
    "--instance_prompt", "a photo of sks dog",
    "--resolution", "512",
    "--rank", "32",
    "--train_batch_size", "4",
    "--gradient_accumulation_steps", "1",
    "--learning_rate", "4e-4",
    "--report_to", "wandb",
    "--lr_scheduler", "constant",
    "--lr_warmup_steps", "0",
    "--max_train_steps", "500",
    "--validation_prompt", "A photo of sks dog in front of a building",
    "--validation_epochs", "25",
    "--seed", "0",
    "--time_step", "0.2",
    "--repeats", "8",
    "--reinit_strategy", "all",
]


def run_experiment(name, extra_args, output_dir):
    """Run a single training experiment."""
    print(f"\n{'='*60}")
    print(f"Starting experiment: {name}")
    print(f"Output dir: {output_dir}")
    print(f"{'='*60}\n")

    cmd = [
        "accelerate", "launch",
        "--config_file", "../config/accelerate_config.yaml",
        "../train_dreambooth_lora_one_sd3.py",
    ] + SHARED_ARGS + [
        "--output_dir", output_dir,
    ] + extra_args

    print("Command:", " ".join(cmd))
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"WARNING: {name} exited with return code {result.returncode}")
    else:
        print(f"{name} completed successfully.")

    return result.returncode


def main():
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    # 1. LoRA baseline (random init)
    rc1 = run_experiment(
        "LoRA Baseline",
        ["--baseline"],
        os.path.join(BASE_OUTPUT_DIR, "lora_baseline"),
    )

    # 2. LoRA-One (gradient init)
    rc2 = run_experiment(
        "LoRA-One",
        ["--stable_gamma", "1"],
        os.path.join(BASE_OUTPUT_DIR, "lora_one"),
    )

    # 3. Full finetune
    rc3 = run_experiment(
        "Full Finetune",
        ["--full_finetune"],
        os.path.join(BASE_OUTPUT_DIR, "full_finetune"),
    )

    print(f"\n{'='*60}")
    print("All experiments finished.")
    print(f"  LoRA Baseline:  return code {rc1}")
    print(f"  LoRA-One:       return code {rc2}")
    print(f"  Full Finetune:  return code {rc3}")
    print(f"{'='*60}")

    if any(rc != 0 for rc in [rc1, rc2, rc3]):
        sys.exit(1)


if __name__ == "__main__":
    main()
