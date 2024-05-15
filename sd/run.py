import model_loader
import pipeline
from PIL import Image
from transformers import CLIPTokenizer
import torch
import os
from safetensors.numpy import load_file
import sys
sys.path.append("esrgan/")
from inference import inference_esrgan, save_generated, save_upscaled

DEVICE = "cpu"

WIDTH = 512
HEIGHT = 768

SAVE_PATH = "static/"
DO_UPSCALE = True
UPSCALER_PATH = "esrgan/weights/R-ESRGAN_x4.pth"
ALLOW_CUDA = True

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"


tokenizer = CLIPTokenizer(vocab_file="data/vocab.json" , merges_file="data/merges.txt")
model_file = "data/Anime_Amore.safetensors"


## TEXT TO IMAGE
prompt = "1girl, blonde hair, short hair, bun, blue eyes, red blazer, white shirt, blue skirt, 8k, masterpiece"
uncond_prompt = "bad resolution, worst quality, extra digits, ugly"
do_cfg = True
cfg_scale = 7

## IMAGE TO IMAGE
input_image = None
image_path = ""
# input_image = Image.open(image_path)
strength = 0.8

sampler = "ddpm"
num_inference_steps = 10
seed = None

def runSD(prompt=prompt, uncond_prompt=uncond_prompt, input_image=input_image, strength=strength, do_cfg=do_cfg, 
          cfg_scale=cfg_scale, sampler_name=sampler, n_inference_steps=num_inference_steps, seed=seed, model_file=model_file,
          HEIGHT=HEIGHT, WIDTH=WIDTH, device=DEVICE, idle_device='cpu', tokenizer=tokenizer, SAVE_PATH=SAVE_PATH, 
          DO_UPSCALE=DO_UPSCALE):

    print(f"Loading model weights from folder: {model_file}")
    models = model_loader.preload_models_from_standard_weights(model_file, device)
    print(f"Using device: {device}")

    output_image = pipeline.generate(
        prompt=prompt,
        uncond_prompt=uncond_prompt,
        input_image=input_image,
        strength=strength,
        do_cfg=do_cfg,
        cfg_scale=cfg_scale,
        sampler_name=sampler,
        n_inference_steps=n_inference_steps,
        seed=seed,
        models=models,
        HEIGHT=HEIGHT,
        WIDTH=WIDTH,
        device=DEVICE,
        idle_device="cpu",
        tokenizer=tokenizer
    )

    image_path = save_generated(SAVE_PATH, output_image)
    print(image_path)

    if DO_UPSCALE:
        upscaled_path = inference_esrgan(save_path = SAVE_PATH, 
                         image_path = image_path, 
                         model_path = UPSCALER_PATH, 
                         device=DEVICE)
        os.remove(image_path)
        return upscaled_path

    return image_path