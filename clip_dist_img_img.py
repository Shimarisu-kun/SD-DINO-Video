# Adapted from:
# https://medium.com/@jeremy-k/unlocking-openai-clip-part-2-image-similarity-bf0224ab5bb0

import torch
import clip
from PIL import Image
import os
import torch.nn as nn
from statistics import mean
from itertools import pairwise
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

dataset_folder = 'car_edited'

images = []
for root, dirs, files in os.walk(dataset_folder):
	for file in files:
		if file.endswith('jpg') or file.endswith('png'):
			images.append(  root  + '/'+ file)

cos = torch.nn.CosineSimilarity(dim=0)
similarities = []

for image1, image2 in pairwise(images):
	with torch.no_grad():
		image1_preprocess = preprocess(Image.open(image1)).unsqueeze(0).to(device)
		image1_features = model.encode_image( image1_preprocess)

		image2_preprocess = preprocess(Image.open(image2)).unsqueeze(0).to(device)
		image2_features = model.encode_image( image2_preprocess)

		similarity = cos(image1_features[0],image2_features[0]).item()
		similarity = (similarity+1)/2
		similarities.append(similarity)
avrg_sim = mean(similarities)
print("Frame CLIP similarity: ", avrg_sim)