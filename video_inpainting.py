# https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
import os

# import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
import requests
import io
import base64
from PIL import Image, PngImagePlugin

inpaint = True
imgNumFrom = 1
imgNumTo = 58

prompt = "red toy car, colorful, saturated, (ultra detailed face and eyes:1.1), (album art:1.1), nostalgia, (cinematic:1.5), moody, dramatic lighting, (photo:0.6), majestic, oil painting, high detail, soft focus, (neon:0.7), golden hour, bokeh, (centered:1.5), (rough brushstrokes:1.3), (concept art:1.5), rimlight"
neg_prompt = "cartoon, 3d, zombie, disfigured, deformed, extra limbs, b&w, black and white, duplicate, morbid, mutilated, cropped, out of frame, extra fingers, mutated hands, mutation, extra limbs, clone, out of frame, too many fingers, long neck, tripod, photoshop, video game, tiling, cut off head, patterns, borders, (frame:1.4), symmetry, intricate, signature, text, watermark"

referencePath = "C:/Users/Admin/Documents/Informatik/BA_thesis/sd-dino_videos/original/car-turn_cropped/car-turn/ref_refined.png"
imagePath = "C:/Users/Admin/Documents/Informatik/BA_thesis/sd-dino_videos/original/car-turn_cropped/"
maskPath = "C:/Users/Admin/Documents/Informatik/BA_thesis/sd-dino_videos/original/car-turn_cropped/car-turn/"
outputPath = "C:/Users/Admin/Documents/Informatik/BA_thesis/sd-dino_videos/original/car-turn_cropped/edited/"

img_extension = ".jpg"
mask_extension = ".png"

# Erstelle Ordner für die Abspeicherung
if not os.path.exists(outputPath):
    os.mkdir(outputPath)

# Initzalisiere den predictor mit GPU falls kompatibel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Verwende', device, 'für das Inpainting.')

for i in range(imgNumFrom, imgNumTo+1):
    # Read an image
    s = str(i)
    img_name = s.zfill(5)
    print(img_name)
    img_prev_name = str(i-1).zfill(5)  + ".png"

    mask_filename = img_name + "_mask" + mask_extension
    out_filename = img_name + "" + img_extension
    inpainted_filename = img_name + ".png"

    og_img = cv2.imread(imagePath + img_name + img_extension)
    og_img = cv2.resize(og_img, (512, 512))

    img = cv2.imread(maskPath + img_name + mask_extension)
    img = cv2.resize(img, (512, 512))

    img_prev = cv2.imread(outputPath + img_prev_name)
    img_prev = cv2.resize(img_prev, (512, 512))

    mask = cv2.imread(maskPath + mask_filename)
    mask = cv2.resize(mask, (512, 512))

    reference = cv2.imread(referencePath)
    reference = cv2.resize(reference, (512, 512))

    url = "http://127.0.0.1:7860"

    opt = requests.get(url=f'{url}/sdapi/v1/options')
    response = opt.json()
    response['sd_model_checkpoint'] = 'dreamshaper_631Inpainting.safetensors'
    requests.post(url=f'{url}/sdapi/v1/options', json=response)

    _, buffer = cv2.imencode('.jpg', og_img)
    encoded_og_image = base64.b64encode(buffer).decode('utf-8')
    _, buffer = cv2.imencode('.jpg', img)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    _, buffer = cv2.imencode('.jpg', img_prev)
    encoded_image_prev = base64.b64encode(buffer).decode('utf-8')
    _, buffer = cv2.imencode('.jpg', mask)
    encoded_mask = base64.b64encode(buffer).decode('utf-8')
    _, buffer = cv2.imencode('.jpg', reference)
    encoded_reference = base64.b64encode(buffer).decode('utf-8')

    height, width = img.shape[:2]

    payload = {
      "init_images": [
        encoded_image
      ],
      "resize_mode": 0,
      "denoising_strength": 0.2,
      "image_cfg_scale": 7,
      "mask": encoded_mask,
      "mask_blur": 4,
      "inpainting_fill": 1, #0: fill (start with mean surrounding color), 1: original (start with reference img), 2: latent noise (start with pure noise), 3: latent nothing (start with "black" latent)
      "inpaint_full_res": False,
      "inpaint_full_res_padding": 4,
      "inpainting_mask_invert": 0,
      "initial_noise_multiplier": 0,
      "prompt": prompt,
      "seed": 240,
      "subseed": -1,
      "subseed_strength": 0,
      "seed_resize_from_h": -1,
      "seed_resize_from_w": -1,
      "sampler_name": "DDIM",
      "batch_size": 1,
      "n_iter": 1,
      "steps": 20,
      "cfg_scale": 7,
      "width": width,
      "height": height,
      "restore_faces": True,
      "tiling": False,
      "do_not_save_samples": True,
      "do_not_save_grid": True,
      "negative_prompt": neg_prompt,
      "eta": 0,
      "s_churn": 0,
      "s_tmax": 0,
      "s_tmin": 0,
      "s_noise": 1,
      "override_settings": {},
      "override_settings_restore_afterwards": True,
      "sampler_index": "DDIM",
      "include_init_images": True,
      "send_images": True,
      "save_images": False,
      "alwayson_scripts": {
      "controlnet": {
      #"controlnet_units": [
      "args": [
            {
                "input_image": encoded_reference,
                "module": "reference_only"
            },
            {
                "input_image": encoded_image_prev,
                "module": "reference_only"
            },
            {
                "input_image": encoded_og_image,
                "module": "depth_midas",
                "model": "control_v11f1p_sd15_depth [cfd03158]"
            }
        ]
      }
      },
      
    }

    response = requests.post(url=f'{url}/sdapi/v1/img2img', json=payload)

    r = response.json()

    i = r['images'][0]

    inpainted_image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))

    png_payload = {
        "image": "data:image/png;base64," + i
    }
    response2 = requests.post(url=f'{url}/sdapi/v1/png-info', json=png_payload)

    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("parameters", response2.json().get("info"))
    inpainted_image.save(outputPath + inpainted_filename, pnginfo=pnginfo)