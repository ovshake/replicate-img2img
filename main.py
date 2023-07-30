import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize
import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline, LMSDiscreteScheduler
import os
from urllib.parse import urlparse

def initialize_pipeline(pipeline_model="nitrosocke/Ghibli-Diffusion", device="cuda"):
    # Initialize pipeline
    dtype = torch.float32 if device == "cpu" else torch.float16
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(pipeline_model,
                                                          torch_dtype=dtype).to(device)
    # Update scheduler
    lms = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = lms

    return pipe

def generate_image(pipe, device="cuda", url_or_filepath=None, prompt=None,
                   strength=0.75, guidance_scale=7.5, seed=1024):
    if url_or_filepath is None or prompt is None:
        raise ValueError("Both 'url_or_filepath' and 'prompt' must be provided.")

    # Parse image from url or local file
    parsed = urlparse(url_or_filepath)
    if parsed.scheme in ['http', 'https']:
        # Fetch image from url
        response = requests.get(url_or_filepath)
        init_image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        # Open image from local file
        init_image = Image.open(url_or_filepath).convert("RGB")

    init_image.thumbnail((768, 768))

    # Set up random generator
    generator = torch.Generator(device=device).manual_seed(seed)

    # Generate initial image
    image = pipe(prompt=prompt, image=init_image, strength=strength,
                 guidance_scale=guidance_scale, generator=generator, num_inference_steps=5).images[0]

    return image

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = initialize_pipeline(pipeline_model="nitrosocke/Ghibli-Diffusion", device=device)
    image = generate_image(pipe, device=device, url_or_filepath="assets/sketch-mountains-input.jpg", prompt="A van gogh painting of a starry night.")
    image.save("assets/output.png")

