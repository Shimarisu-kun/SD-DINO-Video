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

imgNumFrom = 0
imgNumTo = 200

vid_name = "stroller"

imagePath = "C:/Users/Admin/Documents/Informatik/BA_thesis/sd-dino_videos/original/" + vid_name + "/"
outputPath = "C:/Users/Admin/Documents/Informatik/BA_thesis/sd-dino_videos/original/" + vid_name + "_cropped/"

img_in_extension = ".jpg"
img_out_extension = ".jpg"

width = 480
height = 480

h_min = 0
h_max = h_min + height
w_min = 186
w_max = w_min + width

# Erstelle Ordner für die Abspeicherung
if not os.path.exists(outputPath):
    os.mkdir(outputPath)

for i in range(imgNumFrom, imgNumTo+1):
    # Read an image
    s = str(i)
    img_name = s.zfill(5)
    print(img_name)

    img = cv2.imread(imagePath + img_name + img_in_extension)
    img = img[h_min:h_max, w_min:w_max]


    cv2.imwrite(outputPath + img_name + img_out_extension, img)