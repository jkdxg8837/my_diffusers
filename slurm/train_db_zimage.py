
import subprocess
import sys

MODEL_NAME = "Tongyi-MAI/Z-Image"
INSTANCE_DIR = "dog"
OUTPUT_DIR = "trained-z-image-lora-prodigy"
SAVE_AT_STEPS = "0,10,100,200,300,400,500"
MAX_STEP = 500

# --- Training ---
subprocess.run([
    "accelerate", "launch", "../train_dreambooth_lora_zimage.py",
    "--pretrained_model_name_or_path", MODEL_NAME,
    "--instance_data_dir", INSTANCE_DIR,
    "--output_dir", OUTPUT_DIR,
    "--mixed_precision", "bf16",
    "--gradient_checkpointing",
    "--cache_latents",
    "--instance_prompt", "a photo of sks dog",
    "--resolution", "512",
    "--train_batch_size", "1",
    "--guidance_scale", "5.0",
    "--gradient_accumulation_steps", "4",
    "--optimizer", "prodigy",
    "--learning_rate", "1e-4",
    "--report_to", "wandb",
    "--lr_scheduler", "constant_with_warmup",
    "--lr_warmup_steps", "100",
    "--max_train_steps", str(MAX_STEP),
    "--repeats", "8",
    "--rank", "32",
    "--lora_alpha", "32",
    "--save_at_steps", SAVE_AT_STEPS,
    "--seed", "42",
], check=True)

# --- Evaluation ---
subprocess.run([
    sys.executable, "eval_dreambooth_zimage.py",
    "--pretrained_model_name_or_path", MODEL_NAME,
    "--output_dir", OUTPUT_DIR,
    "--checkpoint_steps", SAVE_AT_STEPS,
    "--max_step", str(MAX_STEP),
    "--seed", "42",
    "--num_validation_images", "5",
    "--mixed_precision", "bf16",
    "--guidance_scale", "5.0",
], check=True)
