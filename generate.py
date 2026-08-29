import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# Step 1: Load the pipeline directly from the local cache folder
# (bypasses Hub repo-ID lookup entirely — no network, no completeness checks)
model_path = os.path.join(
    os.environ["USERPROFILE"],
    ".cache", "huggingface", "hub",
    "models--runwayml--stable-diffusion-v1-5",
    "snapshots",
    "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
)

pipe = StableDiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float32,   # CPU doesn't support float16, so we stay in float32
    use_safetensors=True,
)

# Step 2: Swap the default scheduler for a faster one
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Step 3: Move pipeline to CPU explicitly (this is the default anyway, but explicit is better)
pipe = pipe.to("cpu")

# Step 4: Define our prompt
prompt = "a futuristic cyberpunk city street at night, neon lights, rain, cinematic"

# Step 5: Generate the image
image = pipe(
    prompt,
    num_inference_steps=25,
    height=512,
    width=512,
).images[0]

# Step 6: Save it
image.save("output.png")
print("Image saved as output.png")