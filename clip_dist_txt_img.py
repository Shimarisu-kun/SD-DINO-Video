# Adapted from:
# https://medium.com/@jeremy-k/unlocking-openai-clip-part-1-intro-to-zero-shot-classification-f81194f4dff7

import torch
import clip
from PIL import Image
import os
import itertools
import torch.nn as nn
from statistics import mean
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
dataset_folder = 'car_edited'

#Load all the images into an array
images = []
for root, dirs, files in os.walk(dataset_folder):
    for file in files:
        if file.endswith('jpg') or file.endswith('png'):
            images.append(  root  + '/'+ file)

text = clip.tokenize(['michael jordan, running']).to(device)
text_features = model.encode_text(text)
result = {}
cos = torch.nn.CosineSimilarity(dim=0)
#For each image, compute its cosine similarity with the prompt and store the result in a dict
for img in images:
    with torch.no_grad():
        image_preprocess = preprocess(Image.open(img)).unsqueeze(0).to(device)
        image_features = model.encode_image( image_preprocess)
        sim = cos(image_features[0],text_features[0]).item()
        sim = (sim+1)/2
        result[img]=sim

avrg_sim = mean(result.values())
print("text-video CLIP similarity: " + str(avrg_sim))

"""        
#Sort the dict and retrieve the first 3 values
sorted_value = sorted(result.items(), key=lambda x:x[1], reverse=True)
sorted_res = dict(sorted_value)
top_3 = dict(itertools.islice(sorted_res.items(), 3))
print(top_3)
"""