import torch
import pickle as pkl

TARGET_MODULES = [
    "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
    "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
    "ff.net.0.proj", "ff.net.2", "norm_out.linear", "proj_out",
]
def load_pretrained_weights(model_name):
    """Load pretrained SD3 transformer weights for target modules only."""
    print("Loading pretrained transformer weights...")
    transformer = SD3Transformer2DModel.from_pretrained(
        model_name, subfolder="transformer"
    )
    weights = {}
    for name, param in transformer.named_parameters():
        for target in TARGET_MODULES:
            if target in name and name.endswith(".weight"):
                weights[name] = param.data.clone().cpu().float()
                break
    del transformer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Loaded {len(weights)} weight tensors.")
    return weights

def load_lora_init_weights(method_name, step):
    """Load LoRA A/B from eval_fft_lora_init/{method}/step-{step}/pytorch_lora_weights.pt.

    Returns {base_key: {"A": tensor, "B": tensor}}.
    """
    path = os.path.join(
        LORA_INIT_BASE, method_name, f"step-{step}", "pytorch_lora_weights.pt"
    )
    if not os.path.exists(path):
        print(f"  LoRA init weights not found: {path}")
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)
    return _parse_lora_state(state)

def load_lora_trained_weights(path):
    """Load a separately-trained LoRA checkpoint (.pt or .safetensors).

    Returns {base_key: {"A": tensor, "B": tensor}} or None.
    """
    if path is None or not os.path.exists(path):
        return None
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(path)
    else:
        state = torch.load(path, map_location="cpu", weights_only=False)
    return _parse_lora_state(state)
def load_fft_weights(step):
    """Load trainable_weights.pt for a given FFT checkpoint step.

    Returns only target-module keys, already on CPU float32.
    """
    path = os.path.join(FFT_DIR, f"checkpoint-{step}", "trainable_weights.pt")
    if not os.path.exists(path):
        print(f"  FFT weights not found: {path}")
        return None
    print(f"  Loading FFT weights: {path}")
    raw = torch.load(path, map_location="cpu", weights_only=False)
    filtered = {}
    for key, tensor in raw.items():
        for target in TARGET_MODULES:
            if target in key and key.endswith(".weight"):
                filtered[key] = tensor.float()
                break
    del raw
    gc.collect()
    return filtered


def _parse_lora_state(state_dict):
    """Parse a LoRA state dict into {base_key: {"A": tensor, "B": tensor}}.

    Handles key formats:
      - transformer_blocks.0.attn.to_q.lora_A.weight          (get_peft_model_state_dict)
      - base_model.model.transformer_blocks.0.attn.to_q.lora_A.default.weight  (raw peft)
      - transformer.transformer_blocks.0.attn.to_q.lora_A.weight               (safetensors)
    """
    pairs = {}
    for key, tensor in state_dict.items():
        clean = key
        # strip common prefixes
        for prefix in ("base_model.model.", "transformer."):
            if clean.startswith(prefix):
                clean = clean[len(prefix):]

        if ".lora_A." in clean:
            base_key = clean.split(".lora_A.")[0] + ".weight"
            pairs.setdefault(base_key, {})["A"] = tensor.cpu().float()
        elif ".lora_B." in clean:
            base_key = clean.split(".lora_B.")[0] + ".weight"
            pairs.setdefault(base_key, {})["B"] = tensor.cpu().float()
    return pairs
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)


def main():
    eta = 5.656854249492381
    zeta_ga = ??
    lora_one_value = 0.0
    lora_ga_value = 0.0
    lora_value = 0.0
    pretrained = load_pretrained_weights(MODEL_NAME)
    for step in CHECKPOINT_STEPS:
        fft = load_fft_weights(step)
        lora_init = load_lora_init_weights(METHOD_NAME, step)
        lora_trained = load_lora_trained_weights(PATH)
        lora_one_weight = load_lora_init_weights("lora-one", step)
        lora_ga_weight = load_lora_init_weights("lora-ga", step)
        for key in fft.keys():
            gradient = fft[key] - pretrained[key]
            lora_one_A_weight = lora_one_weight[key]["A"]
            lora_one_B_weight = lora_one_weight[key]["B"]
            lora_ga_A_weight = lora_ga_weight[key]["A"]
            lora_ga_B_weight = lora_ga_weight[key]["B"]
            lora_A_weight = lora_init[key]["A"]
            lora_B_weight = lora_init[key]["B"]

            A_t_A_one = torch.matmul(lora_one_A_weight, lora_one_A_weight.T)
            A_t_A_ga = torch.matmul(lora_ga_A_weight, lora_ga_A_weight.T)
            A_t_A = torch.matmul(lora_A_weight, lora_A_weight.T)
            B_t_B_one = torch.matmul(lora_one_B_weight, lora_one_B_weight.T)
            B_t_B_ga = torch.matmul(lora_ga_B_weight, lora_ga_B_weight.T)
            B_t_B = torch.matmul(lora_B_weight, lora_B_weight.T)
            
            modified_grad_one = eta * eta * torch.matmul(grad, A_t_A_one) + eta * eta *torch.matmul(B_B_t_one, grad) - grad * zeta_ga
            modified_grad_ga = eta * eta * torch.matmul(grad, A_t_A_ga) + eta * eta *torch.matmul(B_B_t_ga, grad) - grad * zeta_ga
            modified_grad = eta * eta * torch.matmul(grad, A_t_A) + eta * eta *torch.matmul(B_B_t, grad) - grad * zeta_ga
            lora_one_grad_result = torch.norm(modified_grad_one, p='fro').item()
            lora_ga_grad_result = torch.norm(modified_grad_ga, p='fro').item()
            lora_grad_result = torch.norm(modified_grad, p='fro').item()
            lora_one_value += lora_one_grad_result
            lora_ga_value += lora_ga_grad_result
            lora_value += lora_grad_result

    print(f"Lora one value: {lora_one_value}")
    print(f"Lora ga value: {lora_ga_value}")
    print(f"Lora value: {lora_value}")