---
base_model: Tongyi-MAI/Z-Image
library_name: diffusers
license: apache-2.0
instance_prompt: a photo of sks dog
widget: []
tags:
- text-to-image
- diffusers-training
- diffusers
- lora
- z-image
- template:sd-lora
---

<!-- This model card has been generated automatically according to the information the training script had access to. You
should probably proofread and complete it, then remove this comment. -->


# Z Image DreamBooth LoRA - trained-z-image-lora-prodigy

<Gallery />

## Model description

These are trained-z-image-lora-prodigy DreamBooth LoRA weights for Tongyi-MAI/Z-Image.

The weights were trained using [DreamBooth](https://dreambooth.github.io/) with the [Z Image diffusers trainer](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_z_image.md).

Quant training? None

## Trigger words

You should use `a photo of sks dog` to trigger the image generation.

## Download model

[Download the *.safetensors LoRA](trained-z-image-lora-prodigy/tree/main) in the Files & versions tab.

## Use it with the [🧨 diffusers library](https://github.com/huggingface/diffusers)

```py
from diffusers import AutoPipelineForText2Image
import torch
pipeline = AutoPipelineForText2Image.from_pretrained("Tongyi-MAI/Z-Image", torch_dtype=torch.bfloat16).to('cuda')
pipeline.load_lora_weights('trained-z-image-lora-prodigy', weight_name='pytorch_lora_weights.safetensors')
image = pipeline('a photo of sks dog').images[0]
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation on loading LoRAs in diffusers](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters)

## License

Apace License 2.0


## Intended uses & limitations

#### How to use

```python
# TODO: add an example code snippet for running this diffusion pipeline
```

#### Limitations and bias

[TODO: provide examples of latent issues and potential remediations]

## Training details

[TODO: describe the data used to train the model]