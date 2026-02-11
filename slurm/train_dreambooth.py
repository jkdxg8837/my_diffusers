import os
import subprocess
import sys

# 设置环境变量
os.environ["MODEL_NAME"] = "stabilityai/stable-diffusion-3-medium-diffusers"
os.environ["INSTANCE_DIR"] = "./dog"
# os.environ["OUTPUT_DIR"] = "sd3-dog-singlecard-reinit80-randomseed-woprecondition-POS-crossAtt-scaleLR"
os.environ["OUTPUT_DIR"] = "./dog-weight-svd-121"
time_step = 0.2
re_init_schedule = "multi"
re_init_bsz = 1
re_init_samples = 32
noise_samples = 1
stable_gamma = 64
stable_gamma_list = [64]
# 构造命令

cmd = [
    "accelerate", "launch",
    "--config_file", "../config/accelerate_config.yaml",
    "../train_dreambooth_lora_one_sd3.py",
    "--pretrained_model_name_or_path", os.environ.get("MODEL_NAME"), # 使用 os.environ.get 提供默认值以防环境变量未设置
    "--instance_data_dir", os.environ.get("INSTANCE_DIR"),   # 使用 os.environ.get 提供默认值
    "--output_dir", os.environ.get("OUTPUT_DIR"),       # 使用 os.environ.get 提供默认值
    "--stable_gamma", str(stable_gamma),
    "--mixed_precision", "fp16",
    "--instance_prompt", "a photo of sks dog",
    "--resolution", "512",
    "--rank", "32",
    "--train_batch_size", "4",
    "--gradient_accumulation_steps", "1",
    "--learning_rate", "1e-4",
    "--report_to", "wandb",
    "--lr_scheduler", "constant",
    "--lr_warmup_steps", "0",
    "--max_train_steps", "500",
    "--validation_prompt", "A photo of sks dog in front of a building",
    "--validation_epochs", "25",
    "--seed", "0",
    "--time_step", str(time_step),
    # "--re_init_bsz", str(re_init_bsz),
    "--repeats","8",
    "--reinit_strategy", "all",
    # "--baseline",
    "--max_train_steps", "500",
    "--init_config", "/home/j/jiayang_gu/workspace/diffusers/config/svd.yaml",
    # "--full_finetune",
    # "--with_prior_preservation",
    # "--class_data_dir",
    # os.environ.get("INSTANCE_DIR"),
    # "--class_prompt",
    # "a photo of dog",
    # "--fixed_noise",
    # "--noise_samples", str(noise_samples),
    # "--reinit_only"
            # "--push_to_hub"
]

is_full_finetune = "--full_finetune" in cmd

# Build eval command based on training mode
output_dir = os.environ.get("OUTPUT_DIR")
if is_full_finetune:
    # FFT eval: use eval_dreambooth_sd3_fft.py with --fft_weights_path
    def make_fft_eval_cmd(checkpoint_dir):
        return [
            sys.executable,
            "../eval_dreambooth_sd3_fft.py",
            "--pretrained_model_name_or_path", os.environ.get("MODEL_NAME"),
            "--fft_weights_path", os.path.join(checkpoint_dir, "trainable_weights.pt"),
            "--output_dir", os.path.join(checkpoint_dir, "eval_output"),
            "--seed", "42",
            "--num_validation_images", "5",
            "--mixed_precision", "fp16",
        ]
else:
    # LoRA eval: use eval_dreambooth_sd3.py via accelerate
    eval_cmd = [
        "accelerate", "launch",
        "--config_file", "../config/accelerate_config.yaml",
        "../eval_dreambooth_sd3.py",
        "--pretrained_model_name_or_path", os.environ.get("MODEL_NAME"),
        "--instance_data_dir", os.environ.get("INSTANCE_DIR"),
        "--instance_prompt", "a photo of sks dog",
        "--resolution", "512",
        "--rank", "32",
        "--train_batch_size", "1",
        "--gradient_accumulation_steps", "1",
        "--max_train_steps", "2",
        "--learning_rate", "1e-4",
        "--lr_scheduler", "constant",
        "--lr_warmup_steps", "0",
        "--output_dir", output_dir,
        "--lora_scale", "1.0",
    ]

# Run training
subprocess.run(cmd)

# Run evaluation at each checkpoint
eval_checkpoints = [1, 100, 200, 300, 400]
for step in eval_checkpoints:
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    if is_full_finetune:
        subprocess.run(make_fft_eval_cmd(checkpoint_dir))
    else:
        eval_cmd[eval_cmd.index("--output_dir") + 1] = checkpoint_dir
        subprocess.run(eval_cmd)



# New method model commands

# cmd[8] = os.environ.get("OUTPUT_DIR") + "_" + "stable_gamma" + str(stable_gamma)
# cmd[cmd.index("--stable_gamma") + 1] = str(stable_gamma)
# subprocess.run(cmd)

# clip_score_cmd = [
#     "python",
#     "../eval_from_dir.py",
#     "--img_path", "./"
# ]
# # Define the checkpoints you want to evaluate
# checkpoints = [0, 10, 20, 50, 100, 150, 200, 250, 300]  # Add or remove steps as needed
# # checkpoints = [0, 1, 10, 20]  # Add or remove steps as needed
# for stable_gamma in stable_gamma_list:
#     # 先不加learning rate scaling
#     # cmd[8] = os.environ.get("OUTPUT_DIR") + "_" + "lr_scale" + str(stable_gamma)
#     # cmd[cmd.index("--lr_scale") + 1] = str(stable_gamma)
#     cmd[8] = os.environ.get("OUTPUT_DIR") + "_" + "stable_gamma" + str(stable_gamma)
#     cmd[cmd.index("--stable_gamma") + 1] = str(stable_gamma)
#     subprocess.run(cmd)

#     for step in checkpoints:
#         checkpoint_path = f"{cmd[8]}/checkpoint-{step}"
#         eval_cmd[eval_cmd.index("--output_dir") + 1] = checkpoint_path
#         subprocess.run(eval_cmd)

#     # Also evaluate the final output dir if needed
#     eval_cmd[eval_cmd.index("--output_dir") + 1] = cmd[8]
#     subprocess.run(eval_cmd)

    # Eval 3 outputs CLIP results
    # lora_scale = eval_cmd[eval_cmd.index("--lora_scale") + 1]
    # for step in checkpoints:
    #     clip_score_cmd[clip_score_cmd.index("--img_path") + 1] = f"{cmd[8]}/checkpoint-{step}/output_scale{lora_scale}"
    #     subprocess.run(clip_score_cmd)
    # clip_score_cmd[clip_score_cmd.index("--img_path") + 1] = f"{cmd[8]}/output_scale{lora_scale}"
    # subprocess.run(clip_score_cmd)