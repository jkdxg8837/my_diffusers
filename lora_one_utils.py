import sys
import torch
from tqdm import tqdm
import math
import utils
import itertools
from peft.tuners.lora.layer import Linear as LoraLinear
import logging
log = logging.getLogger(__name__)
from typing import Tuple, List, Dict

# Module-level buffer for SVD diagnostics
_svd_diagnostics = []

def reset_svd_diagnostics():
    global _svd_diagnostics
    _svd_diagnostics = []

def print_svd_diagnostics_summary():
    global _svd_diagnostics
    if not _svd_diagnostics:
        return
    print("\n" + "=" * 70, flush=True)
    print("LoRA Init Diagnostics Summary", flush=True)
    print("=" * 70, flush=True)
    print(f"{'Module Name':<55} {'Energy Ratio':>12} {'Allclose':>8}", flush=True)
    print("-" * 70, flush=True)
    for d in _svd_diagnostics:
        print(f"{d['name']:<55} {d['energy_ratio']:>12.6f} {str(d['allclose']):>8}", flush=True)
    energy_ratios = [d['energy_ratio'] for d in _svd_diagnostics]
    allclose_all = all(d['allclose'] for d in _svd_diagnostics)
    print("-" * 70, flush=True)
    print(f"{'Mean energy ratio:':<55} {sum(energy_ratios)/len(energy_ratios):>12.6f}", flush=True)
    print(f"{'Min energy ratio:':<55} {min(energy_ratios):>12.6f}", flush=True)
    print(f"{'Max energy ratio:':<55} {max(energy_ratios):>12.6f}", flush=True)
    print(f"{'All allclose passed:':<55} {str(allclose_all):>8}", flush=True)
    print("=" * 70 + "\n", flush=True)
    _svd_diagnostics = []
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


            # Support custom timestep schedules from init_config
            if hasattr(args, 'init_config_dict') and args.init_config_dict is not None and 'custom_timesteps' in args.init_config_dict:
                custom_timesteps = args.init_config_dict['custom_timesteps']
                u = torch.tensor(custom_timesteps, dtype=torch.float32, device=model_input.device)
                # If custom timesteps don't match batch size, repeat/sample
                if len(u) != bsz:
                    if len(u) < bsz:
                        # Repeat to match batch size
                        u = u.repeat((bsz // len(u)) + 1)[:bsz]
                    else:
                        # Sample random subset
                        indices_sample = torch.randperm(len(u))[:bsz]
                        u = u[indices_sample]
                print(f"Using custom timesteps: {u}")
            else:
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


def collect_gradient_fixed_noise(
    models, dataloader, args, noise_scheduler_copy, accelerator, text_encoders, tokenizers,
    noise_seed=42,
) -> Dict[str, torch.Tensor]:
    r"""
    Collect the one-step full gradient of LoRA-target parameters with fixed (deterministic) noise.

    Similar to estimate_gradient, but:
    - Noise is fixed: generated deterministically from a seeded CPU generator so that
      the same noise is produced across runs for the same seed.
    - Single pass through the entire dataloader (full-batch gradient).
    - Images and text come from the dataloader / args.instance_prompt (not fixed).

    Returns:
        Dict mapping parameter names to their averaged gradient tensors.
    """
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    transformer, vae = models[0], models[1]
    log.info("Collecting gradient with fixed noise (seed=%d)", noise_seed)
    transformer.train()

    for param in transformer.parameters():
        param.requires_grad = True

    grad_params = [p for p in transformer.parameters() if p.requires_grad]
    log.info(f"Number of parameters requiring gradients: {len(grad_params)}")
    if len(grad_params) == 0:
        log.warning("No parameters require gradients!")

    named_grads = {}
    hooks = []
    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    weight_dtype = torch.float16

    for name, param in transformer.named_parameters():
        if param.requires_grad:
            hook = param.register_hook(get_record_gradient_hook(transformer, named_grads))
            hooks.append(hook)

    # Pre-compute text embeddings (fixed prompt, same as training)
    def compute_text_embeddings(prompt, text_encoders, tokenizers):
        with torch.no_grad():
            from train_dreambooth_lora_one_sd3 import encode_prompt
            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                text_encoders, tokenizers, prompt, args.max_sequence_length
            )
            prompt_embeds = prompt_embeds.to(accelerator.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
        return prompt_embeds, pooled_prompt_embeds

    prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
        args.instance_prompt, text_encoders, tokenizers
    )

    # Fixed noise generator on CPU for cross-run reproducibility
    noise_gen = torch.Generator(device='cpu')
    noise_gen.manual_seed(noise_seed)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    num = 0
    for batch in tqdm(dataloader, desc="Collecting gradient (fixed noise)"):
        num += 1

        pixel_values = batch["pixel_values"].to(dtype=vae.dtype, device=vae.device)
        with torch.no_grad():
            model_input = vae.encode(pixel_values).latent_dist.sample()
            model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
            model_input = model_input.to(dtype=weight_dtype)

        bsz = model_input.shape[0]

        # Fixed noise: deterministic from seeded CPU generator
        noise = torch.randn(
            model_input.shape, generator=noise_gen, dtype=model_input.dtype, device='cpu'
        ).to(model_input.device)

        # Sample timesteps (same scheme as estimate_gradient)
        u = sample_with_matched_distribution(n=bsz, mean=0, std=1.0)
        print("u is set to ", u)

        indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
        timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)
        sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
        sigmas = sigmas.detach()
        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

        # Forward pass
        model_pred = transformer(
            hidden_states=noisy_model_input,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]

        if args.precondition_outputs:
            model_pred = model_pred * (-sigmas.detach()) + noisy_model_input

        weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
        weighting = weighting.detach()

        # Flow matching loss
        if args.precondition_outputs:
            target = model_input.detach()
        else:
            target = noise - model_input.detach()

        if args.with_prior_preservation:
            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
            target, target_prior = torch.chunk(target, 2, dim=0)
            weighting, weighting_prior = torch.chunk(weighting, 2, dim=0)
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
            loss = loss + args.prior_loss_weight * prior_loss

        print(f"Batch {num}: loss = {loss.item()}")
        accelerator.backward(loss)

        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(transformer_lora_parameters, args.max_grad_norm)

        # Record gradients (including last layer)
        get_record_gradient_hook(transformer, named_grads)(None)
        for n, p in transformer.named_parameters():
            if p.grad is not None:
                p.grad = None

    torch.cuda.empty_cache()

    # Average gradients across all batches
    for key in tqdm(named_grads.keys(), desc="Computing gradient averages"):
        try:
            tensors = named_grads[key]
            named_grads[key] = torch.stack(tensors, dim=0).mean(dim=0)
        except Exception as e:
            log.error(f"Error processing key {key}: {e}")

    import os
    torch.save(named_grads, os.path.join(args.output_dir, "fixed_noise_gradient.pt"))
    log.info("Saved fixed-noise gradient to %s", os.path.join(args.output_dir, "fixed_noise_gradient.pt"))

    # Cleanup hooks
    for hook in hooks:
        hook.remove()

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

            svd_algo = init_config.get('svd_algo', 'svd')
            if svd_algo == 'pmd':
                U_p, V_p, d = pmd(grads, rank=lora_r,
                                  c_u=init_config.get('pmd_c_u', 1.0),
                                  c_v=init_config.get('pmd_c_v', 1.0),
                                  rerank=init_config.get('rerank', False))
                S_r = torch.abs(d.to(grads.device))
                B = U_p @ torch.diag(torch.sqrt(S_r)) / torch.sqrt(S_r[0])
                A = torch.diag(torch.sqrt(S_r)) @ V_p.T / torch.sqrt(S_r[0])
            else:
                U, S, V = torch.linalg.svd(grads)
                rank = (S > 1e-5).sum().item()
                B = U[:, :lora_r] @ torch.diag(torch.sqrt(S[:lora_r])) / torch.sqrt(S[0])
                A = torch.diag(torch.sqrt(S[:lora_r])) @ V[:lora_r, :] / torch.sqrt(S[0])
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
        grad_name = name + '.weight'
        grads = named_grad[grad_name]

        if init_config['direction'] == 'LoRA-One':
            grads = -grads.cuda().float()
            m, n = grads.shape

            svd_algo = init_config.get('svd_algo', 'svd')
            if svd_algo == 'pmd':
                U_p, V_p, d = pmd(grads, rank=lora_r,
                                  c_u=init_config.get('pmd_c_u', 1.0),
                                  c_v=init_config.get('pmd_c_v', 1.0),
                                  rerank=init_config.get('rerank', False))
                S_r = torch.abs(d.to(grads.device))
                energy_ratio = 0.0  # full singular spectrum unavailable with PMD
                B = U_p @ torch.diag(torch.sqrt(S_r)) / torch.sqrt(S_r[0])
                A = torch.diag(torch.sqrt(S_r)) @ V_p.T / torch.sqrt(S_r[0])
            else:
                U, S, V = torch.linalg.svd(grads)
                energy_ratio = ((S[:lora_r]**2).sum() / (S**2).sum()).item()
                rank = (S > 1e-5).sum().item()
                B = U[:, :lora_r] @ torch.diag(torch.sqrt(S[:lora_r])) / torch.sqrt(S[0])
                A = torch.diag(torch.sqrt(S[:lora_r])) @ V[:lora_r, :] / torch.sqrt(S[0])

            if "diagnostics" in additional_info:
                additional_info["diagnostics"].append({
                    "name": name,
                    "energy_ratio": energy_ratio,
                })
                print(name, energy_ratio)

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
    diagnostics = []
    additional_info["diagnostics"] = diagnostics
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
    lora_r_b = min(module.lora_B.default.weight.shape)
    lora_r = min(lora_r, lora_r_b)
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
        # print(weight_name)
        if weight_name not in fft_weights or weight_name not in pretrained_weights:
            log.warning(f"FFT or pretrained weight not found for {weight_name}, skipping.")
            print(f"FFT or pretrained weight not found for {weight_name}, skipping.")
            return inited_signal
        # For delta W, should use reverse symbol
        delta = -(fft_weights[weight_name] - pretrained_weights[weight_name]).cuda().float()
        
        # Apply scale factor if provided (scales the delta before SVD)
        scale_factor = init_config.get("scale_factor", 1.0)
        if scale_factor != 1.0:
            delta = delta * scale_factor

        if init_config['direction'] == 'LoRA-One':
            grads = -delta
            m, n = grads.shape
            # U, S, V = torch.linalg.svd(grads)
            # Validate if svd de-compose is right

            U, S, Vh = torch.linalg.svd(grads, full_matrices=True)

            # Print S matrix first 30 singular values and mean
            s_top30 = S[:30].cpu().tolist()
            # print(f"[S] {name}: {' '.join(f'{v:.6f}' for v in s_top30)}")
            # print(f"[S mean] {name}: {S.mean().item():.6f}")

            rank = (S > 1e-5).sum().item()
            B = U[:, :lora_r] @ torch.diag(torch.sqrt(S[:lora_r])) / torch.sqrt(S[0])
            A = torch.diag(torch.sqrt(S[:lora_r])) @ Vh[:lora_r, :] / torch.sqrt(S[0])
            if torch.isnan(A).any() or torch.isnan(B).any():
                print(f"SVD initialization resulted in NaN for {name}. Skipping initialization.")
                return inited_signal
            all_close_result_rank = torch.allclose(grads, U[:, :lora_r] @ torch.diag(S[:lora_r]) @ Vh[:lora_r, :])
            energy_ratio = ((S[:lora_r]**2).sum() / (S**2).sum()).item()
            additional_info["diagnostics"].append({
                "name": name,
                "energy_ratio": energy_ratio,
                "allclose_rank": all_close_result_rank,
            })
        elif init_config['direction'] == "LoRA-GA":
            m, n = delta.shape
            U, S, V = torch.linalg.svd(delta.float())

            # Print S matrix first 30 singular values and mean
            s_top30 = S[:30].cpu().tolist()
            print(f"[S] {name}: {' '.join(f'{v:.6f}' for v in s_top30)}")
            print(f"[S mean] {name}: {S.mean().item():.6f}")

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


def reinit_lora_from_fft(model, init_config, fft_state_dict, pretrained_state_dict, output_path, rank=None, rank_threshold=None):
    r"""
    Reinitialize LoRA A/B weights from full-finetune weight deltas.

    Args:
        model: The LoRA model.
        init_config: Init config dict (from yaml).
        fft_state_dict: State dict from trainable_weights.pt (full finetune).
        pretrained_state_dict: State dict from the pretrained model.
        rank: LoRA rank. If provided with rank_threshold, only reinit modules whose rank_t <= rank.
        rank_threshold: Energy threshold (0-1) for computing rank_t via SVD cumulative energy.
    Returns:
        (model, inited_modules)
    """
    import os
    diagnostics = []
    additional_info = {
        "fft_weights": fft_state_dict,
        "pretrained_weights": pretrained_state_dict,
        "diagnostics": diagnostics,
    }
    inited_modules = []
    lora_modules = {name: module for name, module in model.named_modules() if isinstance(module, LoraLayer)}

    # Compute and save all weight deltas (fft - pretrained)
    module_deltas = {
        name: (fft_state_dict[name + '.weight'] - pretrained_state_dict[name + '.weight']).cpu()
        for name in lora_modules
        if name + '.weight' in fft_state_dict and name + '.weight' in pretrained_state_dict
    }
    # torch.save(module_deltas, os.path.join(output_path, "module_deltas.pt"))
    # print(f"Saved {len(module_deltas)} module deltas to {os.path.join(output_path, 'module_deltas.pt')}")

    # Compute rank_t for each module if rank filtering is enabled
    rank_t_info = {}
    if rank is not None and rank_threshold is not None:
        for name, delta in module_deltas.items():
            W = delta.float()
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            energy = torch.cumsum(S ** 2, dim=0) / torch.sum(S ** 2)
            min_dim = min(W.shape)
            t = rank_threshold
            rank_t = int((energy < t).sum().item()) + 1
            rank_t = min(rank_t, min_dim)
            rank_t_info[name] = rank_t
        print(f"\nRank filtering enabled (LoRA rank={rank}, threshold={rank_threshold}):")
        for name, rt in rank_t_info.items():
            status = "REINIT" if rt <= rank else "SKIP"
            print(f"  {name}: rank_t={rt} [{status}]")
        print()

    # Reinitialize LoRA weights
    for name, module in tqdm(lora_modules.items(), desc="Reinitializing LoRA from FFT weights"):
        # Skip modules whose rank_t exceeds the LoRA rank
        if rank is not None and rank_threshold is not None and name in rank_t_info and rank_t_info[name] > rank:
            continue
        if_init = reinit_fft_weights(name, module, init_config, additional_info)
        if if_init:
            inited_modules.append(name)

    # Print diagnostics summary
    if diagnostics:
        print("\n" + "=" * 70)
        print("LoRA Initialization Diagnostics Summary (from FFT weights)")
        print("=" * 70)
        print(f"{'Module Name':<55} {'Energy Ratio':>12} {'Allclose':>8}")
        print("-" * 70)
        for d in diagnostics:
            print(f"{d['name']:<55} {d['energy_ratio']:>12.6f} {str(d['allclose_rank']):>8}")
        energy_ratios = [d['energy_ratio'] for d in diagnostics]
        # allclose_all = all(d['allclose'] for d in diagnostics)
        allclose_rank = all(d['allclose_rank'] for d in diagnostics)
        print("-" * 70)
        print(f"{'Mean energy ratio:':<55} {sum(energy_ratios)/len(energy_ratios):>12.6f}")
        print(f"{'Min energy ratio:':<55} {min(energy_ratios):>12.6f}")
        print(f"{'Max energy ratio:':<55} {max(energy_ratios):>12.6f}")
        print(f"{'All allclose_rank passed:':<55} {str(allclose_rank):>8}")
        print("=" * 70 + "\n")

    return model, inited_modules


def _simple_init_module(module, lora_A_init, lora_B_init, a_dim, b_dim,
                        lora_A_std=0.01, lora_B_std=0.01):
    """Apply simple random initialization to LoRA A and B weight matrices."""
    match lora_A_init:
        case "gaussian":
            torch.nn.init.normal_(module.lora_A.default.weight, mean=0.0, std=lora_A_std)
        case "kaiming":
            torch.nn.init.kaiming_uniform_(module.lora_A.default.weight, a=math.sqrt(5))
        case "fan_out_kaiming":
            torch.nn.init.kaiming_normal_(module.lora_A.default.weight, mode="fan_out")
        case "xavier":
            torch.nn.init.xavier_normal_(module.lora_A.default.weight)
        case "zeros":
            torch.nn.init.zeros_(module.lora_A.default.weight)
        case "unit":
            torch.nn.init.normal_(module.lora_A.default.weight, mean=0.0, std=1.0 / (a_dim ** 0.5))
        case "orthogonal":
            torch.nn.init.orthogonal_(module.lora_A.default.weight)
        case _:
            torch.nn.init.kaiming_uniform_(module.lora_A.default.weight, a=math.sqrt(5))

    match lora_B_init:
        case "gaussian":
            torch.nn.init.normal_(module.lora_B.default.weight, mean=0.0, std=lora_B_std)
        case "kaiming":
            torch.nn.init.kaiming_normal_(module.lora_B.default.weight)
        case "fan_out_kaiming":
            torch.nn.init.kaiming_normal_(module.lora_B.default.weight, mode="fan_out")
        case "xavier":
            torch.nn.init.xavier_normal_(module.lora_B.default.weight)
        case "zeros":
            torch.nn.init.zeros_(module.lora_B.default.weight)
        case "unit":
            torch.nn.init.normal_(module.lora_B.default.weight, mean=0.0, std=1.0 / (b_dim ** 0.5))
        case "orthogonal":
            torch.nn.init.orthogonal_(module.lora_B.default.weight)
        case _:
            torch.nn.init.zeros_(module.lora_B.default.weight)


def reinit_lora_from_fft_p2p(
    model,
    init_config,
    fft_state_dict,
    pretrained_state_dict,
    output_path,
    top_k: int = None,
    rank95_threshold: float = 0.95,
    last_k: int = None,
):
    r"""
    Reinitialize LoRA A/B weights using a priority-to-priority (p2p) strategy based on
    per-module rank95 scores computed from full-finetune weight deltas.

    For each LoRA module the weight delta  Δ = fft_weight − pretrained_weight  is computed.
    Its *rank95* score is the minimum SVD rank needed to capture ``rank95_threshold`` (default
    0.95) of the total singular-value energy of Δ.  Modules are then ranked by this score in
    ascending order (lower rank95 → more intrinsically low-rank → better LoRA candidate).

    Exactly one of ``top_k`` or ``last_k`` must be supplied:

    * **top_k** – selects the K modules with the *lowest* rank95 (most low-rank deltas) for
      LoRA-One reconstruction.
    * **last_k** – selects the K modules with the *highest* rank95 (least low-rank deltas) for
      LoRA-One reconstruction.

    Selected modules are initialized via LoRA-One reconstruction from Δ,
    **without** the ``/ torch.sqrt(S[0])`` normalization used in the original LoRA-One:
        B = U[:, :r] @ diag(sqrt(S[:r]))
        A = diag(sqrt(S[:r])) @ Vh[:r, :]

    Remaining modules receive standard simple init (kaiming_uniform A, zeros B by default,
    or whatever ``lora_A``/``lora_B`` keys specify in ``init_config``).

    Args:
        model: The PEFT LoRA-wrapped model.
        init_config: Init config dict (from yaml).  Recognised keys:
            ``scale``       – post-SVD scaling strategy (``"unit"`` / ``"gd"`` / ``"stable"``).
            ``stable_gamma``– gamma for ``"stable"`` scale.
            ``scale_factor``– scalar multiplier applied to Δ before SVD (default 1.0).
            ``lora_A``      – simple-init method for A (default ``"kaiming"``).
            ``lora_B``      – simple-init method for B (default ``"zeros"``).
            ``lora_A_std``  – std for gaussian A init.
            ``lora_B_std``  – std for gaussian B init.
            ``dtype``       – cast to ``"bf16"`` / ``"fp32"`` after init.
        fft_state_dict: State dict of the full-finetune checkpoint.
        pretrained_state_dict: State dict of the pretrained base model.
        output_path: Directory for optional diagnostics artifacts (currently unused).
        top_k: Number of modules with *lowest* rank95 to receive LoRA-One reconstruction.
        rank95_threshold: Energy fraction for computing rank95 (default 0.95).
        last_k: Number of modules with *highest* rank95 to receive LoRA-One reconstruction.

    Returns:
        (model, inited_modules): Updated model and list of names that received LoRA-One init.
    """
    if top_k is None and last_k is None:
        raise ValueError("Either top_k or last_k must be provided.")
    if top_k is not None and last_k is not None:
        raise ValueError("top_k and last_k are mutually exclusive; provide only one.")
    from peft.tuners.lora import LoraLayer

    lora_modules = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, LoraLayer)
    }

    # ------------------------------------------------------------------
    # Step 1: compute rank95 for every LoRA module
    # ------------------------------------------------------------------
    rank95_scores: Dict[str, int] = {}
    print(f"\nComputing rank95 scores (threshold={rank95_threshold}) for {len(lora_modules)} modules...")
    for name, module in tqdm(lora_modules.items(), desc="Computing rank95"):
        weight_name = name + ".weight"
        if weight_name not in fft_state_dict or weight_name not in pretrained_state_dict:
            log.warning(f"Weight not found for {weight_name}, assigning rank95=inf.")
            rank95_scores[name] = int(1e9)
            continue
        delta = (fft_state_dict[weight_name] - pretrained_state_dict[weight_name]).float()
        # Use SVD values only (no need for U/Vh here)
        S = torch.linalg.svdvals(delta)
        total_energy = (S ** 2).sum()
        if total_energy < 1e-12:
            rank95_scores[name] = 1
            continue
        cumulative = torch.cumsum(S ** 2, dim=0) / total_energy
        # First rank where cumulative energy >= threshold
        rank95 = int((cumulative < rank95_threshold).sum().item()) + 1
        rank95 = min(rank95, min(delta.shape))
        rank95_scores[name] = rank95

    # ------------------------------------------------------------------
    # Step 2: rank modules by rank95 (ascending) and select top-K
    # ------------------------------------------------------------------
    sorted_modules: List[Tuple[str, int]] = sorted(rank95_scores.items(), key=lambda x: x[1])
    n_total = len(sorted_modules)

    if top_k is not None:
        k_effective = min(top_k, n_total)
        topk_set = {name for name, _ in sorted_modules[:k_effective]}
        mode_label = f"top_k={k_effective} (lowest rank95)"
    else:  # last_k
        k_effective = min(last_k, n_total)
        topk_set = {name for name, _ in sorted_modules[-k_effective:]}
        mode_label = f"last_k={k_effective} (highest rank95)"

    print(f"\nRank95 ranking (ascending) — {mode_label} get LoRA-One init:")
    print(f"  {'Rank':>4}  {'Method':<8}  {'rank95':>6}  Module")
    print(f"  {'-'*4}  {'-'*8}  {'-'*6}  {'-'*40}")
    for i, (name, score) in enumerate(sorted_modules):
        tag = "LORA-ONE" if name in topk_set else "SIMPLE"
        score_str = str(score) if score < int(1e9) else "inf"
        print(f"  {i+1:>4}  {tag:<8}  {score_str:>6}  {name}")


    # ------------------------------------------------------------------
    # Step 3 & 4: initialize each module
    # ------------------------------------------------------------------
    lora_A_init = init_config.get("lora_A", "kaiming")
    lora_B_init = init_config.get("lora_B", "zeros")
    lora_A_std  = init_config.get("lora_A_std", 0.01)
    lora_B_std  = init_config.get("lora_B_std", 0.01)
    scale        = init_config.get("scale", "unit")
    scale_factor = init_config.get("scale_factor", 1.0)
    stable_gamma = init_config.get("stable_gamma", 1.0)
    dtype        = init_config.get("dtype", None)

    diagnostics = []
    inited_modules = []

    for name, module in tqdm(lora_modules.items(), desc="Reinitializing LoRA (p2p)"):
        lora_r = min(min(module.lora_A.default.weight.shape),
                     min(module.lora_B.default.weight.shape))
        a_dim  = max(module.lora_A.default.weight.shape)
        b_dim  = max(module.lora_B.default.weight.shape)
        weight_name = name + ".weight"

        if name in topk_set:
            # ---- LoRA-One reconstruction (without /sqrt(S[0])) ----
            if weight_name not in fft_state_dict or weight_name not in pretrained_state_dict:
                log.warning(f"Skipping LoRA-One init for {name}: weight not found. Falling back to simple init.")
                _simple_init_module(module, lora_A_init, lora_B_init, a_dim, b_dim, lora_A_std, lora_B_std)
            else:
                # delta sign convention: negate like existing reinit_fft_weights gradient mode
                delta = (fft_state_dict[weight_name] - pretrained_state_dict[weight_name]).cuda().float()
                if scale_factor != 1.0:
                    delta = delta * scale_factor

                U, S, Vh = torch.linalg.svd(delta, full_matrices=False)

                # Diagnostics
                energy_ratio = ((S[:lora_r] ** 2).sum() / ((S ** 2).sum() + 1e-12)).item()
                allclose_rank = torch.allclose(
                    delta, U[:, :lora_r] @ torch.diag(S[:lora_r]) @ Vh[:lora_r, :]
                )
                diagnostics.append({
                    "name": name,
                    "rank95": rank95_scores[name],
                    "energy_ratio": energy_ratio,
                    "allclose_rank": allclose_rank,
                })

                # LoRA-One init WITHOUT /sqrt(S[0])
                B = U[:, :lora_r] @ torch.diag(torch.sqrt(S[:lora_r]))
                A = torch.diag(torch.sqrt(S[:lora_r])) @ Vh[:lora_r, :]

                if torch.isnan(A).any() or torch.isnan(B).any():
                    log.warning(f"SVD produced NaN for {name}. Falling back to simple init.")
                    _simple_init_module(module, lora_A_init, lora_B_init, a_dim, b_dim, lora_A_std, lora_B_std)
                else:
                    # Post-SVD scaling
                    scaling_factor = module.scaling["default"]
                    if scale == "gd":
                        A = A / scaling_factor
                        B = B / scaling_factor
                    elif scale == "stable":
                        B = B / (stable_gamma ** 0.5)
                        A = A / (stable_gamma ** 0.5)
                    # "unit" or unrecognised: no additional scaling

                    module.lora_B.default.weight = torch.nn.Parameter(B.contiguous())
                    module.lora_A.default.weight = torch.nn.Parameter(A.contiguous())
                    inited_modules.append(name)

                del delta, U, S, Vh
        else:
            # ---- Simple init ----
            _simple_init_module(module, lora_A_init, lora_B_init, a_dim, b_dim, lora_A_std, lora_B_std)

        # dtype cast (applied to all modules)
        with torch.no_grad():
            if dtype == "bf16":
                module.lora_A.default.weight.data = module.lora_A.default.weight.data.to(torch.bfloat16)
                module.lora_B.default.weight.data = module.lora_B.default.weight.data.to(torch.bfloat16)
            elif dtype == "fp32":
                module.lora_A.default.weight.data = module.lora_A.default.weight.data.to(torch.float32)
                module.lora_B.default.weight.data = module.lora_B.default.weight.data.to(torch.float32)

    # ------------------------------------------------------------------
    # Print diagnostics for LoRA-One-initialized modules
    # ------------------------------------------------------------------
    if diagnostics:
        print("\n" + "=" * 76)
        print(f"LoRA P2P Init Diagnostics  (top_k={k_effective}, threshold={rank95_threshold})")
        print("=" * 76)
        print(f"  {'Module Name':<50} {'rank95':>6}  {'Energy':>10}  {'Allclose':>8}")
        print(f"  {'-'*50} {'-'*6}  {'-'*10}  {'-'*8}")
        for d in diagnostics:
            print(
                f"  {d['name']:<50} {d['rank95']:>6d}  {d['energy_ratio']:>10.6f}  {str(d['allclose_rank']):>8}"
            )
        energy_ratios = [d["energy_ratio"] for d in diagnostics]
        allclose_all  = all(d["allclose_rank"] for d in diagnostics)
        print(f"  {'-'*50} {'-'*6}  {'-'*10}  {'-'*8}")
        print(f"  {'Mean energy ratio:':<50} {'':>6}  {sum(energy_ratios)/len(energy_ratios):>10.6f}")
        print(f"  {'Min energy ratio:':<50} {'':>6}  {min(energy_ratios):>10.6f}")
        print(f"  {'Max energy ratio:':<50} {'':>6}  {max(energy_ratios):>10.6f}")
        print(f"  {'All allclose_rank passed:':<50} {'':>6}  {'':>10}  {str(allclose_all):>8}")
        print("=" * 76 + "\n")

    print(f"reinit_lora_from_fft_p2p complete: {len(inited_modules)} LoRA-One modules, "
          f"{len(lora_modules) - len(inited_modules)} simple-init modules.")
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
def soft_threshold(a: torch.Tensor, lam: float) -> torch.Tensor:
    """Elementwise soft-thresholding (shrinkage) operator."""
    return torch.sign(a) * torch.relu(torch.abs(a) - lam)
def _binary_search_lambda(z: torch.Tensor, c: float, tol: float = 1e-8) -> float:
    """
    Find lambda such that ||S(z, lambda)||_1 == c via binary search.
    If ||z||_1 <= c already, lambda = 0 (no shrinkage needed).
    """
    if z.norm(1).item() <= c:
        return 0.0
    lo, hi = 0.0, z.abs().max().item()
    for _ in range(100):
        mid = (lo + hi) / 2.0
        val = soft_threshold(z, mid).norm(1).item()
        if abs(val - c) < tol:
            break
        if val > c:
            lo = mid
        else:
            hi = mid
    return mid
def pmd_rank1(
    X: torch.Tensor,
    c_u: float = 1.0,
    c_v: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Rank-1 PMD: find sparse u, v and scalar d such that X ≈ d * u * v^T.

    Args:
        X:       (m, n) matrix
        c_u:     L1 bound on u  (1 <= c_u <= sqrt(m))
        c_v:     L1 bound on v  (1 <= c_v <= sqrt(n))
        max_iter: max power-iteration steps
        tol:     convergence threshold on ||u_new - u_old||

    Returns:
        u: (m,) unit-norm sparse left vector
        v: (n,) unit-norm sparse right vector
        d: scalar singular value
    """
    m, n = X.shape
    # Clamp c values to valid range
    c_u = max(1.0, min(c_u, m ** 0.5))
    c_v = max(1.0, min(c_v, n ** 0.5))

    # Initialize v from the leading right singular vector for fast convergence
    _, _, Vt = torch.linalg.svd(X, full_matrices=False)
    v = Vt[0]

    u = torch.zeros(m, dtype=X.dtype, device=X.device)
    for _ in range(max_iter):
        # --- update u ---
        z_u = X @ v
        lam_u = _binary_search_lambda(z_u, c_u)
        u_new = soft_threshold(z_u, lam_u)
        norm_u = u_new.norm()
        if norm_u < 1e-12:
            break
        u_new = u_new / norm_u

        # --- update v ---
        z_v = X.T @ u_new
        lam_v = _binary_search_lambda(z_v, c_v)
        v_new = soft_threshold(z_v, lam_v)
        norm_v = v_new.norm()
        if norm_v < 1e-12:
            break
        v_new = v_new / norm_v

        # convergence check
        delta = (u_new - u).norm().item() + (v_new - v).norm().item()
        u, v = u_new, v_new
        if delta < tol:
            break

    d = (u @ X @ v).item()
    return u, v, d

def pmd(
    X: torch.Tensor,
    rank: int = 3,
    c_u: float = 1.0,
    c_v: float = 1.0,
    rerank: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Rank-k PMD via sequential deflation.

    Args:
        rerank: If True, re-order components by descending singular value after
                deflation (sequential deflation does not guarantee this ordering).

    Returns:
        U: (m, rank) — left sparse factors
        V: (n, rank) — right sparse factors
        d: (rank,)   — singular values
    """
    m, n = X.shape
    c_u = max(1.0, min(c_u, m ** 0.5))
    c_v = max(1.0, min(c_v, n ** 0.5))

    R = X.clone()
    Us, Vs, ds = [], [], []

    for _ in range(rank):
        u, v, d = pmd_rank1(R, c_u=c_u, c_v=c_v, **kwargs)
        Us.append(u)
        Vs.append(v)
        ds.append(d)
        R = R - d * torch.outer(u, v)  # deflate

    U = torch.stack(Us, dim=1)
    V = torch.stack(Vs, dim=1)
    d = torch.tensor(ds)

    if rerank:
        order = torch.argsort(d, descending=True)
        U, V, d = U[:, order], V[:, order], d[order]

    return U, V, d