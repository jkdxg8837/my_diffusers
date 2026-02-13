import subprocess
import sys
from pathlib import Path

script_dir = Path("/home/j/jiayang_gu/workspace/diffusers")
output_path = "/home/j/jiayang_gu/workspace/diffusers/slurm/dog-weight-fft-lowerlr/checkpoint-50/"
cmd = [
    sys.executable,
    str(script_dir / "eval_dreambooth_sd3_fft.py"),
    "--pretrained_model_name_or_path", "stabilityai/stable-diffusion-3-medium-diffusers",
    "--fft_weights_path", output_path + "trainable_weights.pt",
    "--output_dir", output_path+"eval_output",
    "--seed", "42",
    "--num_validation_images", "5",
    "--mixed_precision", "fp16",
]

subprocess.run(cmd, check=True)