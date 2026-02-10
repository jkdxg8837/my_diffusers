import subprocess
import sys
from pathlib import Path

script_dir = Path("/home/j/jiayang_gu/workspace/diffusers")

MODEL_NAME = "black-forest-labs/FLUX.1-dev"
OUTPUT_BASE = "./flux-dog"          # directory that contains checkpoint-{step}/
CHECKPOINTS = [100, 200, 300, 400, 500]
MAX_STEP = 500

for step in CHECKPOINTS:
    if step == MAX_STEP:
        output_path = OUTPUT_BASE
    else:
        output_path = f"{OUTPUT_BASE}/checkpoint-{step}"

    lora_file = Path(output_path) / "pytorch_lora_weights.safetensors"
    if not lora_file.exists():
        print(f"Skipping step {step}: {lora_file} not found")
        continue

    print(f"\n{'=' * 50}")
    print(f"Evaluating step {step}  ({output_path})")
    print("=" * 50)

    cmd = [
        sys.executable,
        str(script_dir / "eval_dreambooth_flux.py"),
        "--pretrained_model_name_or_path", MODEL_NAME,
        "--output_dir", output_path,
        "--seed", "42",
        "--num_validation_images", "5",
        "--mixed_precision", "fp16",
        "--lora_scale", "1.0",
    ]

    subprocess.run(cmd, check=True)

print("\nAll evaluations complete.")
