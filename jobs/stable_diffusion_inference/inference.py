import torch
import wandb
from diffusers import (
    DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    PNDMScheduler,
)

wandb_config = {
    "model_id": "runwayml/stable-diffusion-v1-5",
    "prompts": [
        "A bee with a mischievous cartoon face, waving a magical wandb in front of a psychedelic paradise.",
        "Two giraffes sipping noodles out a hot tub, 4K cinematic.",
        "Street cats playing in a jazz band on a summer night, painted in the style of Van Gogh.",
    ],
    "num_inference_steps": 25,
    "random_seed": 42,
    "height": 512,
    "width": 512,
    "guidance_scale": 7.5,
    "fp_bits": 32,
    "scheduler": "dpms",
}

with wandb.init(config=wandb_config):
    config = wandb.config

    torch_dtype = {
        16: torch.float16,
        32: torch.float32,
    }.get(config.fp_bits)
    if torch_dtype is None:
        raise ValueError(f"Unsupported fp_bits: {config.fp_bits}, must be 16 or 32")

    scheduler = {
        "pndm": PNDMScheduler,
        "dpms": DPMSolverMultistepScheduler,
        "ddim": DDIMScheduler,
    }.get(config.scheduler)
    if scheduler is None:
        raise ValueError(
            f"Unsupported scheduler: {config.scheduler}, must be pndm or dpms"
        )

    pipeline = DiffusionPipeline.from_pretrained(
        config.model_id,
        safetensors=True,
        torch_dtype=torch_dtype,
    )

    pipeline = pipeline.to("cuda")
    pipeline.scheduler = scheduler.from_config(pipeline.scheduler.config)
    generator = torch.Generator("cuda").manual_seed(config.random_seed)

    img_table = wandb.Table(columns=["prompt", "image"])
    images = pipeline(
        config.prompts,
        generator=generator,
        guidance_scale=config.guidance_scale,
        num_inference_steps=config.num_inference_steps,
    ).images
    for image, prompt in zip(images, config.prompts):
        img_table.add_data(prompt, wandb.Image(image))

    wandb.log({"images": img_table})
