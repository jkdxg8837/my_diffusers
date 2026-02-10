import subprocess
MODEL_NAME = "black-forest-labs/FLUX.1-dev"
INSTANCE_DIR = "dog"
OUTPUT_DIR = "trained-flux-lora"

cmd = [
    "accelerate", "launch", "../train_dreambooth_lora_flux.py",
    "--pretrained_model_name_or_path", MODEL_NAME,
    "--instance_data_dir", INSTANCE_DIR,
    "--output_dir", OUTPUT_DIR,
    "--mixed_precision", "bf16",
    "--instance_prompt", "a photo of sks dog",
    "--resolution", "512",
    "--train_batch_size", "1",
    "--guidance_scale", "1",
    "--gradient_accumulation_steps", "4",
    "--optimizer", "prodigy",
    "--learning_rate", "1.",
    "--report_to", "wandb",
    "--lr_scheduler", "constant",
    "--lr_warmup_steps", "0",
    "--max_train_steps", "500",
    "--validation_prompt", "A photo of sks dog in a bucket",
    "--validation_epochs", "25",
    "--seed", "0",
    "--push_to_hub",
    "--repeats","8",
]

subprocess.run(cmd)
