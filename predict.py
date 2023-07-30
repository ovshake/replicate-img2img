from cog import BasePredictor, Input, Path
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, LMSDiscreteScheduler

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline_model = "nitrosocke/Ghibli-Diffusion"
        self.dtype = torch.float32 if self.device == "cpu" else torch.float16
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(self.pipeline_model,
                                                                   torch_dtype=self.dtype).to(self.device)
        lms = LMSDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.scheduler = lms

    def predict(self,
                image: Path = Input(description="Input image"),
                prompt: str = Input(description="Guidance prompt for the image transformation"),
                strength: float = 0.75,
                guidance_scale: float = 7.5,
                seed: int = 1024,
                num_inference_steps: int = 50
               ) -> Path:
        """Run a single prediction on the model"""
        if image is None or prompt is None:
            raise ValueError("Both 'image' and 'prompt' must be provided.")

        # Open image from local file
        init_image = Image.open(image).convert("RGB")
        init_image.thumbnail((768, 768))

        # Set up random generator
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Generate initial image
        output_image = self.pipe(prompt=prompt, image=init_image, strength=strength,
                                 guidance_scale=guidance_scale, generator=generator, num_inference_steps=num_inference_steps).images[0]


        # Save and return output image path
        output_path = Path("/tmp/output_image.jpg")
        output_image.save(output_path)

        return output_path
