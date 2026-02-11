import subprocess

model_name = "black-forest-labs/FLUX.2-klein-4B"
instance_dir = "dog"
output_dir = "trained-flux2-klein"

cmd = [
    "accelerate", "launch", 
    "--config_file", "./accelerate_zero2.yaml",
    "../train_dreambooth_lora_flux2_klein.py",
    "--do_fp8_training",
    "--pretrained_model_name_or_path", model_name,
    "--instance_data_dir", instance_dir,
    "--output_dir", output_dir,
    # "--mixed_precision", "bf16",
    "--instance_prompt", "a photo of sks dog",
    "--resolution", "512",
    "--train_batch_size", "1",
    "--guidance_scale", "1",
    "--gradient_accumulation_steps", "4",
    "--optimizer", "adamW",
    "--use_8bit_adam",
    "--learning_rate", "1e-5",
    "--report_to", "wandb",
    "--lr_scheduler", "constant",
    "--lr_warmup_steps", "0",
    "--max_train_steps", "500",
    "--seed", "0",
    "--cache_latents",
    "--full_finetune"
    "--repeats","8",
]

subprocess.run(cmd, check=True)
