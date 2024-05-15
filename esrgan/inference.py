import torch
from torchvision.utils import save_image
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import Generator, initialize_weights
from dict_converter import convert_model_weights
import os

def count_digits(str: str) -> int:
    """ Count digits in string """
    count = 0
    for digit in str:
        count += 1
    return count

def add_digits(n: int) -> str:
    """ Add digits to a string, up to 4 digits - (0001, 0002, ... ) """
    image_name = str(n)
    while(count_digits(image_name) < 4):
        image_name = "0" + image_name
    return image_name

def save_generated(save_path: str, output_image: torch.Tensor) -> str:
    """ Returns generated image path as constant to delete it later """
    image_path = save_path + "temp.jpg"
    if(torch.is_tensor(output_image)):
        save_image(output_image, image_path)
    else:
        Image.fromarray(output_image).save(image_path)
    print("Generated image saved")

    return image_path

def save_upscaled(save_path: str, output_image: torch.Tensor) -> str:
    # 0001.jpg, 0002.jpg... - format of storing images
    """ Returns upscaled image path """
    n_images = len(os.listdir(save_path))
    for n in range(1, n_images + 1):
        image_name = add_digits(n) + ".jpg"
        if not os.path.isfile(save_path + image_name):
            break

    image_path = save_path + image_name
    print(f"Saving image in directory: {image_path}...")
    # upscaler returns a tensor, sd - numpy arr
    if(torch.is_tensor(output_image)):
        save_image(output_image, image_path)
    else:
        Image.fromarray(output_image).save(image_path)
    print("Upscaled image saved.")
    
    return image_path

def load_model(checkpoint_path: str, model: torch.nn.Module, device = "cpu"):
    if checkpoint_path == "esrgan/weights/R-ESRGAN_x4.pth":
        checkpoint = convert_model_weights(checkpoint_path, device=device)
    elif checkpoint_path == "esrgan/weights/ESRGAN_gen.pth":
        checkpoint = torch.load(checkpoint_path, map_location = device, weights_only = False)
    else:
        raise Exception("Unsupported model dict.. Try downloading one of the models provided on git repository..")
    
    model.load_state_dict(checkpoint["params_ema"], strict = True)

def upscale(save_path: str, gen: torch.nn.Module, image_path: str, device):

    transform = A.Compose([
        A.Normalize(mean=(0, 0, 0), std = (1, 1, 1)),
        ToTensorV2()
    ])

    gen.eval()
    image = Image.open(image_path)

    with torch.no_grad():
        upscaled_img = gen(transform(image=np.asarray(image))["image"].unsqueeze(0).to(device))
    
    image_path = save_upscaled(save_path, upscaled_img)
    return image_path

def inference_esrgan(save_path: str, image_path: str, model_path: str, device: str = "cpu"):
    print(f"Upscaling the image...")
    print(f"Image path: {image_path} | Using upscaler path: {model_path}" )
    

    gen = Generator(in_channels=3).to(device)

    initialize_weights(gen)
    load_model(model_path, gen, device)

    image_path = upscale(save_path, gen, image_path, device)
    return image_path
    
