import torch
from tqdm import tqdm
import math
import utils
import itertools
from peft.tuners.lora.layer import Linear as LoraLinear
import logging
log = logging.getLogger(__name__)
from typing import Tuple, List, Dict
from diffusers.training_utils import (
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
def get_record_gradient_hook(model, record_dict):
    def record_gradient_hook(grad):
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                if n not in record_dict:
                    record_dict[n] = [p.grad.cpu()]
                else:
                    record_dict[n].append(p.grad.cpu())
                p.grad = None
        return grad

    return record_gradient_hook

import torch
import numpy as np
from scipy.stats import norm

def kde_from_params(means, stds, weights, num_points=1000):
    """Create a KDE-like density from mixture of Gaussians."""
    x = np.linspace(0, 1, num_points)
    pdf = np.zeros_like(x)
    for mu, std, w in zip(means, stds, weights):
        pdf += w * np.exp(-(x - mu)**2 / (2 * std**2)) / (std * np.sqrt(2 * np.pi))
    pdf /= np.trapz(pdf, x)  # Normalize to make it a proper density
    return x, pdf

def sample_from_kde(x, pdf, n_samples=32):
    shift=2.0
    n_samples = n_samples+2
    cdf = np.cumsum(pdf)
    cdf = cdf / cdf[-1]
    # inv_cdf = np.interp(np.linspace(0, 1, n_samples), cdf, x)
    # return torch.tensor(inv_cdf, dtype=torch.float32)[1:-1]  # Exclude the first and last points to avoid 0 and 1
        # Draw random uniform samples and invert CDF
    uniform_samples = np.random.rand(n_samples)
    inv_cdf = np.interp(uniform_samples, cdf, x)
    samples = torch.tensor(inv_cdf, dtype=torch.float32)[1:-1]

    samples = (shift*samples)/(1+(shift-1) * samples)

    return samples


def sample_with_matched_distribution(n=32, mean=0.0, std=1.0):
    shift=2.0
    # Get evenly spaced quantiles (excluding 0 and 1)
    quantiles = np.linspace(1 / (n + 1), n / (n + 1), n)
    # Compute quantile-matched normal values
    samples = norm.ppf(quantiles, loc=mean, scale=std)
    # Convert to torch tensor
    samples = torch.tensor(samples, dtype=torch.float32)
    samples = torch.nn.functional.sigmoid(samples)
    # samples = (shift*samples)/(1+(shift-1) * samples)
    # Center to mean=0 and std=1
    # samples = (samples - samples.mean()) / samples.std()
    # samples = 0.1 + 0.85 * samples 
    samples = (shift*samples)/(1+(shift-1) * samples)
    return samples

def evenly_sample_0_2_to_0_8(n=32):
    """Evenly samples n values from 0.2 to 0.8 (inclusive)."""
    return torch.linspace(0.2, 0.8, steps=n)

def generate_u2_like_samples_v2(n=32):
    # torch.manual_seed(seed)

    # Define weights and centers for mixture components
    weights = torch.tensor([0.3, 0.5, 0.2])
    means = torch.tensor([-0.5, 0.5, 1.5])  # in normal space
    stds = torch.tensor([0.3, 0.2, 0.15])

    samples = []
    for _ in range(n):
        # Choose a component
        idx = torch.multinomial(weights, num_samples=1).item()
        sample = torch.normal(means[idx], stds[idx], size=(1,))
        samples.append(sample)

    samples = torch.stack(samples).squeeze()
    samples = torch.sigmoid(samples)  # squash to (0, 1)

    # Rescale to [0.2, 0.75]
    samples = 0.15 + (0.75 - 0.15) * samples

    return samples
    
def generate_u2_like_samples_v3(n=32):
    # torch.manual_seed(seed)
    
    # Define clusters (observed empirically)
    components = [
        {"mean": 0.26, "std": 0.025, "weight": 0.2},  # cluster 1
        {"mean": 0.45, "std": 0.06,  "weight": 0.5},  # cluster 2
        {"mean": 0.66, "std": 0.04,  "weight": 0.3},  # cluster 3
    ]

    weights = torch.tensor([c["weight"] for c in components])
    weights = weights / weights.sum()

    samples = []
    for _ in range(n):
        # Choose component
        idx = torch.multinomial(weights, num_samples=1).item()
        c = components[idx]

        # Truncated normal to ensure values in [0.2, 0.8]
        for _ in range(10):  # retry loop
            s = torch.normal(mean=c["mean"], std=c["std"], size=(1,))
            if 0.2 <= s <= 0.8:
                samples.append(s)
                break

    return torch.cat(samples)


def sample_like_u2(n_samples=32, seed=None):
    """
    Generate a tensor of samples mimicking the multimodal distribution of u2.
    Peaks and weights are adjusted based on visual distribution analysis.
    """
    if seed is not None:
        np.random.seed(seed)

    # Approximate peak centers, standard deviations, and their weights
    peak_centers = [0.30, 0.45, 0.63, 0.75]
    peak_stds    = [0.03, 0.035, 0.03, 0.02]
    peak_weights = [0.2, 0.45, 0.25, 0.1]  # Should sum to 1

    # Sample count per peak based on weights
    counts = np.random.multinomial(n_samples, peak_weights)

    # Generate and clip samples
    samples = []
    for center, std, count in zip(peak_centers, peak_stds, counts):
        s = np.random.normal(loc=center, scale=std, size=count)
        samples.extend(s)

    samples = np.clip(samples, 0.0, 1.0)
    samples = np.sort(samples)

    return torch.tensor(samples, dtype=torch.float32)

def print_gpu_memory_usage(device_id=0):
    allocated = torch.cuda.memory_allocated(device_id)
    total = torch.cuda.get_device_properties(device_id).total_memory
    ratio = allocated / total
    print(f"显存占用：{ratio:.2%} （{allocated / (1024 ** 2):.2f} MB / {total / (1024 ** 2):.2f} MB）")

def estimate_gradient(
    models, dataloader, args, noise_scheduler_copy, accelerator, text_encoders, tokenizers, batch_size: int = 4
) -> Dict[str, List[torch.Tensor]]:
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    r"""
    Estimate the gradient of the model on the given dataset
    """
    transformer, vae = models[0], models[1]
    log.info("Estimating gradient")
    transformer.train()

    for param in transformer.parameters():
        param.requires_grad = True

    # Debug: Check if transformer parameters require gradients
    grad_params = [p for p in transformer.parameters() if p.requires_grad]
    log.info(f"Number of parameters requiring gradients: {len(grad_params)}")
    if len(grad_params) == 0:
        log.warning("No parameters require gradients! This will cause the backward pass to fail.")
    
    named_grads = {}
    hooks = []
    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    for name, param in transformer.named_parameters():
        if param.requires_grad == True:
            hook = param.register_hook(get_record_gradient_hook(transformer, named_grads))
            hooks.append(hook)
    num = 0
    weight_dtype = torch.float16

    epochs = 5
    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    from tqdm import tqdm
    for epoch in range(epochs):
        for batch in tqdm(dataloader, desc="Estimating gradient"):
            # Save batch
            import pickle as pkl
            # load fixed batch data
            if args.all_fixed:
                # 15 * 3 * 512 *512 
                with open(f"./fixed/batch.pkl", "rb") as f:
                    batch = pkl.load(f)
            # print(batch)
            num += 1
            
            # batch = {k: v.to(transformer.device) for k, v in batch.items()}
            # Calculate diffusion model loss
            pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
            pixel_values = pixel_values.to(vae.device)
            with torch.no_grad():
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
                model_input = model_input.to(dtype=weight_dtype)

            # Sample noise that we'll add to the latents
            # If expand when fix sample, the noise shouldn't be fixed.
            if not args.fixed_noise:
                noise = torch.randn_like(model_input)
            else:
                noise_tensor = torch.load("/dcs/pg24/u5649209/data/workspace/diffusers/noise.pt")
                sample_number = args.noise_samples
                # Randomly sample 'sample_number' indices from the noise tensor's first dimension
                total_samples = noise_tensor.shape[0]
                if sample_number > total_samples:
                    raise ValueError(f"Requested {sample_number} samples, but noise tensor only has {total_samples} samples.")
                # indices = torch.randperm(total_samples)[:sample_number]
                # Select the noise samples based on the random indices
                noise_bank = noise_tensor[:sample_number].to(model_input.device, dtype=model_input.dtype)
                noise = noise_bank[torch.randperm(noise_bank.shape[0])[:model_input.shape[0]]]
            bsz = model_input.shape[0]


            u = sample_with_matched_distribution(n=bsz, mean=0, std=1.0)
            # Save tensor u
            # if args.all_fixed:
            #     u_loaded = torch.load("./fixed/u_tensor.pt").to(u.device)
            #     u = u_loaded

            print("u is set to ", u)

            indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
            timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)
            sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
            sigmas = sigmas.detach()
            noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
            def compute_text_embeddings(prompt, text_encoders, tokenizers):
                with torch.no_grad():
                    from train_dreambooth_lora_one_sd3 import encode_prompt
                    prompt_embeds, pooled_prompt_embeds = encode_prompt(
                        text_encoders, tokenizers, prompt, args.max_sequence_length
                    )
                    prompt_embeds = prompt_embeds.to(accelerator.device)
                    pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
                return prompt_embeds, pooled_prompt_embeds
            instance_prompt_hidden_states, instance_pooled_prompt_embeds = compute_text_embeddings(
                args.instance_prompt, text_encoders, tokenizers
            )
            prompt_embeds = instance_prompt_hidden_states
            pooled_prompt_embeds = instance_pooled_prompt_embeds

            # Predict the noise residual
            model_pred = transformer(
                hidden_states=noisy_model_input,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]
            # args.precondition_outputs = 0

            if args.precondition_outputs:
                # model_pred = model_pred * (-sigmas) + noisy_model_input
                # model_pred to be pure model_input
                model_pred = model_pred * (-sigmas.detach()) + noisy_model_input

            weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
            weighting = weighting.detach()
            # print(weighting)
            # flow matching loss
            if args.precondition_outputs:
                target = model_input.detach()  
            else:
                target = noise - model_input.detach()  
            # So target is model_input
            if args.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                target, target_prior = torch.chunk(target, 2, dim=0)
                weighting, weighting_prior = torch.chunk(weighting, 2, dim = 0)
                # Compute prior loss
                prior_loss = torch.mean(
                    (weighting.float() * (model_pred_prior.float() - target_prior.float()) ** 2).reshape(
                        target_prior.shape[0], -1
                    ),
                    1,
                )
                prior_loss = prior_loss.mean()  
            loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
            loss = loss.mean()
            if args.with_prior_preservation:
                # Add the prior loss to the instance loss.
                loss = loss + args.prior_loss_weight * prior_loss
            print(loss.item())
            # loss.backward()
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                params_to_clip = (
                    itertools.chain(
                        transformer_lora_parameters, text_lora_parameters_one, text_lora_parameters_two
                    )
                    if args.train_text_encoder
                    else transformer_lora_parameters
                )
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

            # print_gpu_memory_usage(0)
            get_record_gradient_hook(transformer, named_grads)(None)  # get gradient of last layer
            # make sure the gradient is cleared
            for n, p in transformer.named_parameters():
                if p.grad is not None:
                    p.grad = None
        torch.cuda.empty_cache()

        from tqdm import tqdm
        
        for key in tqdm(named_grads.keys(), desc="Computing gradient averages"):
            try:
                # Stack all tensors in the list along dim=0 and compute mean
                tensors = named_grads[key]
                named_grads[key] = torch.stack(tensors, dim=0).mean(dim=0)
            except Exception as e:
                log.error(f"Error processing key {key}: {e}")
        import os
        torch.save(named_grads, os.path.join(args.output_dir, f"raw_gradient_averages_{epoch}.pt"))
        named_grads = {}

        for hook in hooks:
            hook.remove()
        for name, param in transformer.named_parameters():
            if param.requires_grad == True:
                hook = param.register_hook(get_record_gradient_hook(transformer, named_grads))
                hooks.append(hook)
    torch.cuda.empty_cache()
    return named_grads

@torch.no_grad()
def reinit_lora_module_seg(name, module, init_config, adapter_name, additional_info):
    r"""
    Reinitialize the lora model with the given configuration.
    """

    reinit_start = init_config.get("reinit_pos_start", 0)
    reinit_end = init_config.get("reinit_pos_end", 23)
    reinit_lora_modules = init_config.get("lora_module", "all")
    lora_r = min(module.lora_A[adapter_name].default.weight.shape)
    a_dim = max(module.lora_A[adapter_name].default.weight.shape)
    b_dim = max(module.lora_B[adapter_name].default.weight.shape)

    
    try:
        layer_num_str = name.split(".")[1]
        layer_num = int(layer_num_str)
    except (ValueError, IndexError):
        # Using extra layers for reinit, not only the numbered layers
        layer_num = reinit_start
    except Exception:
        # If not convertible to int, skip assigning layer_num
        layer_num = -1
    inited_signal = False
    
        # print(1)

    if reinit_lora_modules == "crossAtt":
        # only with the write name and write layer can be re-initialzed
        if "add_q_proj" in name or "add_v_proj" in name:
            reinit_start = 0
            reinit_end = 23
            if (layer_num >= reinit_start and layer_num <= reinit_end):
                init_mode = init_config['mode']
                inited_signal = True

    elif reinit_lora_modules == "selfAtt":
        # only with the write name and write layer can be re-initialzed
        if "to_q" in name or "to_v" in name or "to_k" in name:
            init_mode = init_config['mode']
            reinit_start = 2
            reinit_end = 21
            if (layer_num >= reinit_start and layer_num <= reinit_end):
                init_mode = init_config['mode']
    elif reinit_lora_modules == "all":
        init_mode = init_config['mode']
        inited_signal = True
    else:
        init_mode = "simple"
        init_config["lora_A"] = "kaiming"
        init_config["lora_B"] = "zeros"
        inited_signal = False 

    if init_mode == "simple":
        match init_config["lora_A"]:
            case "gaussian":
                torch.nn.init.normal_(
                    module.lora_A.default.weight, mean=0.0, std=init_config["lora_A_std"]
                )
            case "kaiming":
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                torch.nn.init.kaiming_uniform_(module.lora_A.default.weight, a=math.sqrt(5))
            case "fan_out_kaiming":
                torch.nn.init.kaiming_normal_(
                    module.lora_A.default.weight, mode="fan_out"
                )
            case "xavier":
                torch.nn.init.xavier_normal_(module.lora_A.default.weight)
            case "zeros":
                torch.nn.init.zeros_(module.lora_A.default.weight)
            case "unit":
                torch.nn.init.normal_(
                    module.lora_A.default.weight, mean=0.0, std=1.0 / (a_dim**0.5)
                )
            case "orthogonal":
                torch.nn.init.orthogonal_(module.lora_A.default.weight)
            case _:
                raise ValueError(f"Unknown lora_A initialization: {init_config['lora_A']}")
        match init_config['lora_B']:
            case "gaussian":
                torch.nn.init.normal_(
                    module.lora_B.default.weight, mean=0.0, std=init_config['lora_B_std']
                )
            case "kaiming":
                torch.nn.init.kaiming_normal_(module.lora_B.default.weight)
            case "fan_out_kaiming":
                torch.nn.init.kaiming_normal_(
                    module.lora_B.default.weight, mode="fan_out"
                )
            case "xavier":
                torch.nn.init.xavier_normal_(module.lora_B.default.weight)
            case "zeros":
                torch.nn.init.zeros_(module.lora_B.default.weight)
            case "unit":
                torch.nn.init.normal_(
                    module.lora_B.default.weight, mean=0.0, std=1.0 / (b_dim**0.5)
                )
            case "orthogonal":
                torch.nn.init.orthogonal_(module.lora_B.default.weight)
            case _:
                raise ValueError(f"Unknown lora_B initialization: {init_config.lora_B}")
    # if init_config.get("scale", "") == "stable":
    #     # gamma = init_config.stable_gamma
    #     gamma = 1
    #     module.lora_B.default.weight.data *= (m**0.25) / gamma**0.5
    #     module.lora_A.default.weight.data *= (n**0.25) / gamma**0.5
    elif init_mode == "svd":
        U, S, V = torch.svd_lowrank(module.weight.float(), q=4 * lora_r, niter=4)
        V = V.T
        m, n = module.weight.shape
        if init_config.scale == "default":
            S = S / module.scaling["default"]
            module.lora_B.default.weight = torch.nn.Parameter(
                (U[:, :lora_r] * torch.sqrt(S[:lora_r])).contiguous()
            )
            module.lora_A.default.weight = torch.nn.Parameter(
                (V[:lora_r, :].T * torch.sqrt(S[:lora_r])).T.contiguous()
            )
        elif init_config.scale == "stable":
            gamma = init_config.stable_gamma
            module.lora_B.default.weight = torch.nn.Parameter(
                (U[:, :lora_r] * (m**0.25) / gamma**0.5).contiguous()
            )
            module.lora_A.default.weight = torch.nn.Parameter(
                (V[:lora_r, :] * (n**0.25) / gamma**0.5).contiguous()
            )
        elif init_config.scale == "unit":
            module.lora_B.default.weight = torch.nn.Parameter(
                (U[:, :lora_r]).contiguous()
            )
            module.lora_A.default.weight = torch.nn.Parameter(
                (V[:lora_r, :]).contiguous()
            )
        elif init_config.scale == "normalized":
            S_sum = S[:lora_r].sum()
            module.lora_B.default.weight = torch.nn.Parameter(
                (U[:, :lora_r] * torch.sqrt(S[:lora_r])/torch.sqrt(S_sum)*lora_r**0.5).contiguous()
            )
            module.lora_A.default.weight = torch.nn.Parameter(
                (V[:lora_r, :].T * torch.sqrt(S[:lora_r])/torch.sqrt(S_sum)*lora_r**0.5).T.contiguous()
            )
    elif init_mode == "gradient":
        named_grad = additional_info["named_grads"]
        print("*************************")
        grad_name = name + '.weight'
        grads = named_grad[grad_name]

        if init_config['direction'] == 'LoRA-One':
            # V = V.T
            grads = -grads.cuda().float()
            m, n = grads.shape

            # grads = grads * (m**0.5)
            U, S, V = torch.linalg.svd(grads)
            # print(grads.numel()**0.5)
            rank = (S > 1e-5).sum().item()

            # B = U[:, :lora_r] @ torch.diag(torch.sqrt(S[:lora_r])) / torch.sqrt(S[0])
            # A = torch.diag(torch.sqrt(S[:lora_r])) @ V[:lora_r, :] / torch.sqrt(S[0])
            B = U[:, :lora_r] @ torch.diag(torch.sqrt(S[:lora_r])) / torch.sqrt(S[0])
            A = torch.diag(torch.sqrt(S[:lora_r])) @ V[:lora_r, :] / torch.sqrt(S[0])
            # B = U[:, :lora_r] @ torch.diag(torch.sqrt(S[:lora_r]))
            # A = torch.diag(torch.sqrt(S[:lora_r])) @ V[:lora_r, :]
            if torch.isnan(A).any() or torch.isnan(B).any():
                print(f"SVD initialization resulted in NaN for {name}. Skipping initialization.")
                return
        elif init_config['direction'] == "LoRA-GA":
            m, n = grads.shape
            print(m,n)
            U, S, V = torch.linalg.svd(grads.float())
            B = U[:, lora_r : 2 * lora_r]
            A = V[:lora_r, :]
        scaling_factor = module.scaling["default"]
        if init_config["scale"] == "gd":
            A = A / scaling_factor
            B = B / scaling_factor
        elif init_config["scale"] == "unit":
            # Because A,B is orthogonal, do not need to scale
            pass
        elif init_config["scale"] == "stable":
          if init_config["direction"] == "LoRA-One":
            gamma = init_config["stable_gamma"]
            B = B / gamma**0.5
            A = A / gamma**0.5
          else:
            m, n = grads.shape # m: feature_out, n: feature_in
            # the scale of output is only related to the feature_out
            gamma = init_config["stable_gamma"]
            B = B * m**0.25 / gamma**0.5
            A = A * m**0.25 / gamma**0.5
        elif init_config["scale"] == "weightS":
            _, S, _ = torch.svd_lowrank(module.weight.float(), q=4 * lora_r, niter=4)
            S = S / module.scaling["default"]
            avg_s = torch.sqrt(S[:lora_r]).mean().to(A.device)
            B = B * avg_s
            A = A * avg_s

        # construct new magnitude vectors if use DoRA
        # if peft_conf.get("dora", False):
        #    # temp matrix
        #    V = module.weight.float() + (peft_conf.lora_alpha/math.sqrt(lora_r)) * B @ A
        #    mag_vec = torch.norm(V, p=2, dim=1)
        # else:
        #    pass        
        # he_lora_weights = utils._calculate_he(torch.matmul(B, A).float())
        module.lora_B[adapter_name].default.weight = torch.nn.Parameter(B.contiguous().cuda())
        module.lora_A[adapter_name].default.weight = torch.nn.Parameter(A.contiguous().cuda())
        # if peft_conf.get("dora", False):
        #    module.lora_magnitude_vector.default.weight = torch.nn.Parameter(mag_vec.contiguous().cuda())

    with torch.no_grad():
        # if peft_conf.get("dora", False): #DoRA uses fp16
        #         module.lora_A.default.weight.data = module.lora_A.default.weight.data.to(
        #             torch.float16
        #         )
        #         module.lora_B.default.weight.data = module.lora_B.default.weight.data.to(
        #             torch.float16
        #         )
        #         module.lora_magnitude_vector.default.weight.data = module.lora_magnitude_vector.default.weight.data.to(
        #             torch.float16
        #         )
        # else:
        # consider dtype not in init_config
        if "dtype" not in init_config:
            pass
        elif init_config["dtype"] == "bf16":
            module.lora_A.default.weight.data = module.lora_A.default.weight.data.to(
                torch.bfloat16
            )
            module.lora_B.default.weight.data = module.lora_B.default.weight.data.to(
                torch.bfloat16
            )
        elif init_config["dtype"] == "fp32":
            module.lora_A.default.weight.data = module.lora_A.default.weight.data.to(
                torch.float32
            )
            module.lora_B.default.weight.data = module.lora_B.default.weight.data.to(
                torch.float32
            )

        # If lora_A@lora_B is not zero, then we need to subtract lora_A@lora_B from the original weight matrix
        if init_config["direction"] == "LoRA-One":
            pass
        else:
            offset = (module.lora_B.default.weight @ module.lora_A.default.weight).to(
                module.weight.data.device
            )
            scaling_factor = module.scaling["default"]
            offset *= scaling_factor
            if "norm_clip" in init_config and init_config.norm_clip:
                # for numerical stability, offset's largest value must be less then weight's largest value
                ratio = torch.max(torch.abs(module.weight.data)) / torch.max(
                    torch.abs(offset)
                )
                if ratio < 1:
                    offset *= ratio
                    module.lora_A.default.weight.data *= ratio**0.5
                    module.lora_B.default.weight.data *= ratio**0.5
                    log.warning(f"Clipping offset by {ratio}")
            try:
                module.weight.data -= offset
            except:
                breakpoint()
        return inited_signal

@torch.no_grad()
def reinit_lora_module(name, module, init_config, additional_info):
    r"""
    Reinitialize the lora model with the given configuration.
    """

    reinit_start = init_config.get("reinit_pos_start", 0)
    reinit_end = init_config.get("reinit_pos_end", 23)
    reinit_lora_modules = init_config.get("lora_module", "all")
    lora_r = min(module.lora_A.default.weight.shape)
    a_dim = max(module.lora_A.default.weight.shape)
    b_dim = max(module.lora_B.default.weight.shape)

    
    try:
        layer_num_str = name.split(".")[1]
        layer_num = int(layer_num_str)
    except (ValueError, IndexError):
        # Using extra layers for reinit, not only the numbered layers
        layer_num = reinit_start
    except Exception:
        # If not convertible to int, skip assigning layer_num
        layer_num = -1
    inited_signal = False
    
        # print(1)

    if reinit_lora_modules == "crossAtt":
        # only with the write name and write layer can be re-initialzed
        if "add_q_proj" in name or "add_v_proj" in name:
            reinit_start = 0
            reinit_end = 23
            if (layer_num >= reinit_start and layer_num <= reinit_end):
                init_mode = init_config['mode']
                inited_signal = True

    elif reinit_lora_modules == "selfAtt":
        # only with the write name and write layer can be re-initialzed
        if "to_q" in name or "to_v" in name or "to_k" in name:
            init_mode = init_config['mode']
            reinit_start = 2
            reinit_end = 21
            if (layer_num >= reinit_start and layer_num <= reinit_end):
                init_mode = init_config['mode']
    elif reinit_lora_modules == "all":
        init_mode = init_config['mode']
        inited_signal = True
    else:
        init_mode = "simple"
        init_config["lora_A"] = "kaiming"
        init_config["lora_B"] = "zeros"
        inited_signal = False 

    if init_mode == "simple":
        match init_config["lora_A"]:
            case "gaussian":
                torch.nn.init.normal_(
                    module.lora_A.default.weight, mean=0.0, std=init_config["lora_A_std"]
                )
            case "kaiming":
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                torch.nn.init.kaiming_uniform_(module.lora_A.default.weight, a=math.sqrt(5))
            case "fan_out_kaiming":
                torch.nn.init.kaiming_normal_(
                    module.lora_A.default.weight, mode="fan_out"
                )
            case "xavier":
                torch.nn.init.xavier_normal_(module.lora_A.default.weight)
            case "zeros":
                torch.nn.init.zeros_(module.lora_A.default.weight)
            case "unit":
                torch.nn.init.normal_(
                    module.lora_A.default.weight, mean=0.0, std=1.0 / (a_dim**0.5)
                )
            case "orthogonal":
                torch.nn.init.orthogonal_(module.lora_A.default.weight)
            case _:
                raise ValueError(f"Unknown lora_A initialization: {init_config['lora_A']}")
        match init_config['lora_B']:
            case "gaussian":
                torch.nn.init.normal_(
                    module.lora_B.default.weight, mean=0.0, std=init_config['lora_B_std']
                )
            case "kaiming":
                torch.nn.init.kaiming_normal_(module.lora_B.default.weight)
            case "fan_out_kaiming":
                torch.nn.init.kaiming_normal_(
                    module.lora_B.default.weight, mode="fan_out"
                )
            case "xavier":
                torch.nn.init.xavier_normal_(module.lora_B.default.weight)
            case "zeros":
                torch.nn.init.zeros_(module.lora_B.default.weight)
            case "unit":
                torch.nn.init.normal_(
                    module.lora_B.default.weight, mean=0.0, std=1.0 / (b_dim**0.5)
                )
            case "orthogonal":
                torch.nn.init.orthogonal_(module.lora_B.default.weight)
            case _:
                raise ValueError(f"Unknown lora_B initialization: {init_config.lora_B}")
    # if init_config.get("scale", "") == "stable":
    #     # gamma = init_config.stable_gamma
    #     gamma = 1
    #     module.lora_B.default.weight.data *= (m**0.25) / gamma**0.5
    #     module.lora_A.default.weight.data *= (n**0.25) / gamma**0.5
    elif init_mode == "svd":
        U, S, V = torch.svd_lowrank(module.weight.float(), q=4 * lora_r, niter=4)
        V = V.T
        m, n = module.weight.shape
        if init_config["scale"] == "default":
            S = S / module.scaling["default"]
            module.lora_B.default.weight = torch.nn.Parameter(
                (U[:, :lora_r] * torch.sqrt(S[:lora_r])).contiguous()
            )
            module.lora_A.default.weight = torch.nn.Parameter(
                (V[:lora_r, :].T * torch.sqrt(S[:lora_r])).T.contiguous()
            )
        elif init_config["scale"] == "stable":
            gamma = init_config["stable_gamma"]
            module.lora_B.default.weight = torch.nn.Parameter(
                (U[:, :lora_r] * (m**0.25) / gamma**0.5).contiguous()
            )
            module.lora_A.default.weight = torch.nn.Parameter(
                (V[:lora_r, :] * (n**0.25) / gamma**0.5).contiguous()
            )
        elif init_config["scale"] == "unit":
            module.lora_B.default.weight = torch.nn.Parameter(
                (U[:, :lora_r]).contiguous()
            )
            module.lora_A.default.weight = torch.nn.Parameter(
                (V[:lora_r, :]).contiguous()
            )
        elif init_config["scale"] == "normalized":
            S_sum = S[:lora_r].sum()
            module.lora_B.default.weight = torch.nn.Parameter(
                (U[:, :lora_r] * torch.sqrt(S[:lora_r])/torch.sqrt(S_sum)*lora_r**0.5).contiguous()
            )
            module.lora_A.default.weight = torch.nn.Parameter(
                (V[:lora_r, :].T * torch.sqrt(S[:lora_r])/torch.sqrt(S_sum)*lora_r**0.5).T.contiguous()
            )
    elif init_mode == "gradient":
        named_grad = additional_info["named_grads"]
        # print("*************************")
        grad_name = name + '.weight'
        grads = named_grad[grad_name]

        if init_config['direction'] == 'LoRA-One':
            # V = V.T
            grads = -grads.cuda().float()
            m, n = grads.shape

            # grads = grads * (m**0.5)
            U, S, V = torch.linalg.svd(grads)

            rank = (S > 1e-5).sum().item()

            # B = U[:, :lora_r] @ torch.diag(torch.sqrt(S[:lora_r])) / torch.sqrt(S[0])
            # A = torch.diag(torch.sqrt(S[:lora_r])) @ V[:lora_r, :] / torch.sqrt(S[0])
            B = U[:, :lora_r] @ torch.diag(torch.sqrt(S[:lora_r])) / torch.sqrt(S[0])
            A = torch.diag(torch.sqrt(S[:lora_r])) @ V[:lora_r, :] / torch.sqrt(S[0])
            # B = U[:, :lora_r] @ torch.diag(torch.sqrt(S[:lora_r]))
            # A = torch.diag(torch.sqrt(S[:lora_r])) @ V[:lora_r, :]
            if torch.isnan(A).any() or torch.isnan(B).any():
                print(f"SVD initialization resulted in NaN for {name}. Skipping initialization.")
                return
        elif init_config['direction'] == "LoRA-GA":
            m, n = grads.shape
            U, S, V = torch.linalg.svd(grads.float())
            B = U[:, lora_r : 2 * lora_r]
            A = V[:lora_r, :]
            if torch.isnan(A).any() or torch.isnan(B).any():
                print(f"SVD initialization resulted in NaN for {name}. Skipping initialization.")
                return
        scaling_factor = module.scaling["default"]
        if init_config["scale"] == "gd":
            A = A / scaling_factor
            B = B / scaling_factor
        elif init_config["scale"] == "unit":
            # Because A,B is orthogonal, do not need to scale
            pass
        elif init_config["scale"] == "stable":
          if init_config["direction"] == "LoRA-One":
            gamma = init_config["stable_gamma"]
            B = B / gamma**0.5
            A = A / gamma**0.5
          else:
            m, n = grads.shape # m: feature_out, n: feature_in
            # the scale of output is only related to the feature_out
            gamma = init_config["stable_gamma"]
            B = B * m**0.25 / gamma**0.5
            A = A * m**0.25 / gamma**0.5
        elif init_config["scale"] == "weightS":
            _, S, _ = torch.svd_lowrank(module.weight.float(), q=4 * lora_r, niter=4)
            S = S / module.scaling["default"]
            avg_s = torch.sqrt(S[:lora_r]).mean().to(A.device)
            B = B * avg_s
            A = A * avg_s

        # construct new magnitude vectors if use DoRA
        # if peft_conf.get("dora", False):
        #    # temp matrix
        #    V = module.weight.float() + (peft_conf.lora_alpha/math.sqrt(lora_r)) * B @ A
        #    mag_vec = torch.norm(V, p=2, dim=1)
        # else:
        #    pass        
        # he_lora_weights = utils._calculate_he(torch.matmul(B, A).float())
        module.lora_B.default.weight = torch.nn.Parameter(B.contiguous().cuda())
        module.lora_A.default.weight = torch.nn.Parameter(A.contiguous().cuda())
        # if peft_conf.get("dora", False):
        #    module.lora_magnitude_vector.default.weight = torch.nn.Parameter(mag_vec.contiguous().cuda())

    with torch.no_grad():
        # if peft_conf.get("dora", False): #DoRA uses fp16
        #         module.lora_A.default.weight.data = module.lora_A.default.weight.data.to(
        #             torch.float16
        #         )
        #         module.lora_B.default.weight.data = module.lora_B.default.weight.data.to(
        #             torch.float16
        #         )
        #         module.lora_magnitude_vector.default.weight.data = module.lora_magnitude_vector.default.weight.data.to(
        #             torch.float16
        #         )
        # else:
        # consider dtype not in init_config
        if "dtype" not in init_config:
            pass
        elif init_config["dtype"] == "bf16":
            module.lora_A.default.weight.data = module.lora_A.default.weight.data.to(
                torch.bfloat16
            )
            module.lora_B.default.weight.data = module.lora_B.default.weight.data.to(
                torch.bfloat16
            )
        elif init_config["dtype"] == "fp32":
            module.lora_A.default.weight.data = module.lora_A.default.weight.data.to(
                torch.float32
            )
            module.lora_B.default.weight.data = module.lora_B.default.weight.data.to(
                torch.float32
            )

        # If lora_A@lora_B is not zero, then we need to subtract lora_A@lora_B from the original weight matrix
        if init_config["direction"] == "LoRA-One" or init_config["direction"] == "LoRA-GA":
            pass
        else:
            offset = (module.lora_B.default.weight @ module.lora_A.default.weight).to(
                module.weight.data.device
            )
            scaling_factor = module.scaling["default"]
            offset *= scaling_factor
            if "norm_clip" in init_config and init_config.norm_clip:
                # for numerical stability, offset's largest value must be less then weight's largest value
                ratio = torch.max(torch.abs(module.weight.data)) / torch.max(
                    torch.abs(offset)
                )
                if ratio < 1:
                    offset *= ratio
                    module.lora_A.default.weight.data *= ratio**0.5
                    module.lora_B.default.weight.data *= ratio**0.5
                    log.warning(f"Clipping offset by {ratio}")
            try:
                module.weight.data -= offset
            except:
                breakpoint()
        return inited_signal

from peft.tuners.lora import LoraLayer
def reinit_lora(model, init_config, additional_info):
    r"""
    Reinitialize the lora model with the given configuration.
    """
    inited_modules = []
    for name, module in tqdm(
        model.named_modules(),
        desc="Reinitializing Lora",
        total=len(list(model.named_modules())),
    ):
        
        if isinstance(module, LoraLayer):
            if_init = reinit_lora_module(name, module, init_config, additional_info)
            if if_init:
                inited_modules.append(name)
    return model, inited_modules

@torch.no_grad()
def reinit_fft_weights(name, module, init_config, additional_info):
    r"""
    Reinitialize LoRA A/B from full-finetune weight deltas.

    For gradient-based modes (LoRA-One, LoRA-GA): uses (fft_weight - pretrained_weight)
    as a substitute for the estimated gradient, then performs SVD decomposition.
    For SVD mode: uses the fft weight directly instead of module.weight.
    For simple mode: standard random init (unchanged).

    additional_info must contain:
        - "fft_weights": state dict from trainable_weights.pt
        - "pretrained_weights": state dict from the pretrained model
    """
    reinit_start = init_config.get("reinit_pos_start", 0)
    reinit_end = init_config.get("reinit_pos_end", 23)
    reinit_lora_modules = init_config.get("lora_module", "all")
    lora_r = min(module.lora_A.default.weight.shape)
    a_dim = max(module.lora_A.default.weight.shape)
    b_dim = max(module.lora_B.default.weight.shape)

    try:
        layer_num_str = name.split(".")[1]
        layer_num = int(layer_num_str)
    except (ValueError, IndexError):
        layer_num = reinit_start
    except Exception:
        layer_num = -1
    inited_signal = False

    if reinit_lora_modules == "crossAtt":
        if "add_q_proj" in name or "add_v_proj" in name:
            reinit_start = 0
            reinit_end = 23
            if layer_num >= reinit_start and layer_num <= reinit_end:
                init_mode = init_config['mode']
                inited_signal = True
    elif reinit_lora_modules == "selfAtt":
        if "to_q" in name or "to_v" in name or "to_k" in name:
            init_mode = init_config['mode']
            reinit_start = 2
            reinit_end = 21
            if layer_num >= reinit_start and layer_num <= reinit_end:
                init_mode = init_config['mode']
    elif reinit_lora_modules == "all":
        init_mode = init_config['mode']
        inited_signal = True
    else:
        init_mode = "simple"
        init_config["lora_A"] = "kaiming"
        init_config["lora_B"] = "zeros"
        inited_signal = False

    # Resolve fft weight and pretrained weight for this module
    weight_name = name + '.weight'
    fft_weights = additional_info["fft_weights"]
    pretrained_weights = additional_info["pretrained_weights"]

    if init_mode == "simple":
        match init_config["lora_A"]:
            case "gaussian":
                torch.nn.init.normal_(
                    module.lora_A.default.weight, mean=0.0, std=init_config["lora_A_std"]
                )
            case "kaiming":
                torch.nn.init.kaiming_uniform_(module.lora_A.default.weight, a=math.sqrt(5))
            case "fan_out_kaiming":
                torch.nn.init.kaiming_normal_(
                    module.lora_A.default.weight, mode="fan_out"
                )
            case "xavier":
                torch.nn.init.xavier_normal_(module.lora_A.default.weight)
            case "zeros":
                torch.nn.init.zeros_(module.lora_A.default.weight)
            case "unit":
                torch.nn.init.normal_(
                    module.lora_A.default.weight, mean=0.0, std=1.0 / (a_dim**0.5)
                )
            case "orthogonal":
                torch.nn.init.orthogonal_(module.lora_A.default.weight)
            case _:
                raise ValueError(f"Unknown lora_A initialization: {init_config['lora_A']}")
        match init_config['lora_B']:
            case "gaussian":
                torch.nn.init.normal_(
                    module.lora_B.default.weight, mean=0.0, std=init_config['lora_B_std']
                )
            case "kaiming":
                torch.nn.init.kaiming_normal_(module.lora_B.default.weight)
            case "fan_out_kaiming":
                torch.nn.init.kaiming_normal_(
                    module.lora_B.default.weight, mode="fan_out"
                )
            case "xavier":
                torch.nn.init.xavier_normal_(module.lora_B.default.weight)
            case "zeros":
                torch.nn.init.zeros_(module.lora_B.default.weight)
            case "unit":
                torch.nn.init.normal_(
                    module.lora_B.default.weight, mean=0.0, std=1.0 / (b_dim**0.5)
                )
            case "orthogonal":
                torch.nn.init.orthogonal_(module.lora_B.default.weight)
            case _:
                raise ValueError(f"Unknown lora_B initialization: {init_config['lora_B']}")
    elif init_mode == "svd":
        # Use fft weight directly instead of module.weight
        if weight_name not in fft_weights:
            log.warning(f"FFT weight not found for {weight_name}, skipping.")
            return inited_signal
        fft_w = fft_weights[weight_name].float().cuda()
        U, S, V = torch.svd_lowrank(fft_w, q=4 * lora_r, niter=4)
        V = V.T
        m, n = fft_w.shape
        if init_config["scale"] == "default":
            S = S / module.scaling["default"]
            module.lora_B.default.weight = torch.nn.Parameter(
                (U[:, :lora_r] * torch.sqrt(S[:lora_r])).contiguous()
            )
            module.lora_A.default.weight = torch.nn.Parameter(
                (V[:lora_r, :].T * torch.sqrt(S[:lora_r])).T.contiguous()
            )
        elif init_config["scale"] == "stable":
            gamma = init_config["stable_gamma"]
            module.lora_B.default.weight = torch.nn.Parameter(
                (U[:, :lora_r] * (m**0.25) / gamma**0.5).contiguous()
            )
            module.lora_A.default.weight = torch.nn.Parameter(
                (V[:lora_r, :] * (n**0.25) / gamma**0.5).contiguous()
            )
        elif init_config["scale"] == "unit":
            module.lora_B.default.weight = torch.nn.Parameter(
                (U[:, :lora_r]).contiguous()
            )
            module.lora_A.default.weight = torch.nn.Parameter(
                (V[:lora_r, :]).contiguous()
            )
        elif init_config["scale"] == "normalized":
            S_sum = S[:lora_r].sum()
            module.lora_B.default.weight = torch.nn.Parameter(
                (U[:, :lora_r] * torch.sqrt(S[:lora_r]) / torch.sqrt(S_sum) * lora_r**0.5).contiguous()
            )
            module.lora_A.default.weight = torch.nn.Parameter(
                (V[:lora_r, :].T * torch.sqrt(S[:lora_r]) / torch.sqrt(S_sum) * lora_r**0.5).T.contiguous()
            )
    elif init_mode == "gradient":
        # Use delta = fft_weight - pretrained_weight as the gradient
        if weight_name not in fft_weights or weight_name not in pretrained_weights:
            log.warning(f"FFT or pretrained weight not found for {weight_name}, skipping.")
            return inited_signal
        delta = (fft_weights[weight_name] - pretrained_weights[weight_name]).cuda().float()

        if init_config['direction'] == 'LoRA-One':
            grads = -delta
            m, n = grads.shape
            U, S, V = torch.linalg.svd(grads)
            rank = (S > 1e-5).sum().item()
            B = U[:, :lora_r] @ torch.diag(torch.sqrt(S[:lora_r])) / torch.sqrt(S[0])
            A = torch.diag(torch.sqrt(S[:lora_r])) @ V[:lora_r, :] / torch.sqrt(S[0])
            if torch.isnan(A).any() or torch.isnan(B).any():
                print(f"SVD initialization resulted in NaN for {name}. Skipping initialization.")
                return inited_signal
        elif init_config['direction'] == "LoRA-GA":
            m, n = delta.shape
            U, S, V = torch.linalg.svd(delta.float())
            B = U[:, lora_r : 2 * lora_r]
            A = V[:lora_r, :]
            if torch.isnan(A).any() or torch.isnan(B).any():
                print(f"SVD initialization resulted in NaN for {name}. Skipping initialization.")
                return inited_signal

        scaling_factor = module.scaling["default"]
        if init_config["scale"] == "gd":
            A = A / scaling_factor
            B = B / scaling_factor
        elif init_config["scale"] == "unit":
            pass
        elif init_config["scale"] == "stable":
            if init_config["direction"] == "LoRA-One":
                gamma = init_config["stable_gamma"]
                B = B / gamma**0.5
                A = A / gamma**0.5
            else:
                m, n = delta.shape
                gamma = init_config["stable_gamma"]
                B = B * m**0.25 / gamma**0.5
                A = A * m**0.25 / gamma**0.5
        elif init_config["scale"] == "weightS":
            _, S_w, _ = torch.svd_lowrank(module.weight.float(), q=4 * lora_r, niter=4)
            S_w = S_w / module.scaling["default"]
            avg_s = torch.sqrt(S_w[:lora_r]).mean().to(A.device)
            B = B * avg_s
            A = A * avg_s

        module.lora_B.default.weight = torch.nn.Parameter(B.contiguous().cuda())
        module.lora_A.default.weight = torch.nn.Parameter(A.contiguous().cuda())

    with torch.no_grad():
        if "dtype" not in init_config:
            pass
        elif init_config["dtype"] == "bf16":
            module.lora_A.default.weight.data = module.lora_A.default.weight.data.to(torch.bfloat16)
            module.lora_B.default.weight.data = module.lora_B.default.weight.data.to(torch.bfloat16)
        elif init_config["dtype"] == "fp32":
            module.lora_A.default.weight.data = module.lora_A.default.weight.data.to(torch.float32)
            module.lora_B.default.weight.data = module.lora_B.default.weight.data.to(torch.float32)

        if init_mode == "gradient" and init_config["direction"] in ("LoRA-One", "LoRA-GA"):
            pass
        elif init_mode != "gradient":
            offset = (module.lora_B.default.weight @ module.lora_A.default.weight).to(
                module.weight.data.device
            )
            scaling_factor = module.scaling["default"]
            offset *= scaling_factor
            if "norm_clip" in init_config and init_config.get("norm_clip"):
                ratio = torch.max(torch.abs(module.weight.data)) / torch.max(torch.abs(offset))
                if ratio < 1:
                    offset *= ratio
                    module.lora_A.default.weight.data *= ratio**0.5
                    module.lora_B.default.weight.data *= ratio**0.5
                    log.warning(f"Clipping offset by {ratio}")
            try:
                module.weight.data -= offset
            except:
                breakpoint()
        return inited_signal


def reinit_lora_from_fft(model, init_config, fft_state_dict, pretrained_state_dict):
    r"""
    Reinitialize LoRA A/B weights from full-finetune weight deltas.

    Args:
        model: The LoRA model.
        init_config: Init config dict (from yaml).
        fft_state_dict: State dict from trainable_weights.pt (full finetune).
        pretrained_state_dict: State dict from the pretrained model.
    Returns:
        (model, inited_modules)
    """
    additional_info = {
        "fft_weights": fft_state_dict,
        "pretrained_weights": pretrained_state_dict,
    }
    inited_modules = []
    for name, module in tqdm(
        model.named_modules(),
        desc="Reinitializing LoRA from FFT weights",
        total=len(list(model.named_modules())),
    ):
        if isinstance(module, LoraLayer):
            if_init = reinit_fft_weights(name, module, init_config, additional_info)
            if if_init:
                inited_modules.append(name)
    return model, inited_modules


def reinit_lora_seg(model, init_config, additional_info):
    r"""
    Reinitialize the lora model with the given configuration.
    """
    inited_modules = []
    for name, module in tqdm(
        model.named_modules(),
        desc="Reinitializing Lora",
        total=len(list(model.named_modules())),
    ):
        
        if isinstance(module, LoraLayer):
            if_init = reinit_lora_module_seg(name, module, init_config, adapter_name="large", additional_info=additional_info)
            if if_init:
                inited_modules.append(name)
    return model, inited_modules