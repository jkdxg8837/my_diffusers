import subprocess

model_name = "black-forest-labs/FLUX.1-dev"
instance_dir = "dog"
output_dir = "trained-flux"

cmd = [
    "accelerate", "launch", "train_dreambooth_flux.py",
    "--pretrained_model_name_or_path", model_name,
    "--instance_data_dir", instance_dir,
    "--output_dir", output_dir,
    "--mixed_precision", "bf16",
    "--instance_prompt", "a photo of sks dog",
    "--resolution", "512",
    "--train_batch_size", "1",
    "--guidance_scale", "1",
    "--gradient_accumulation_steps", "4",
    "--optimizer", "prodigy",
    "--learning_rate", "1e-5",
    "--report_to", "wandb",
    "--lr_scheduler", "constant",
    "--lr_warmup_steps", "0",
    "--max_train_steps", "500",
    "--validation_prompt", "A photo of sks dog in a bucket",
    "--validation_epochs", "25",
    "--seed", "0",
    "--push_to_hub"
]

subprocess.run(cmd, check=True)
