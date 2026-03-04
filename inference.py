import os
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import requests


from transformers import AutoProcessor, AutoModel
import torch

# checkpoint = "openai/clip-vit-large-patch14"
# checkpoint = "gzomer/clip-multilingual"
checkpoint = "google/siglip-so400m-patch14-384"


############################################################################
############################# LOAD DATA #################################### 

terms = pd.read_csv("./inference/terms.csv")
tags_EN = list(terms.label_en)
# image_files = pd.Series(sorted(glob("./20260301/data/TTP_images/*.jpg")))

IMAGE_FOLDER = "./S3Bucket/bucket_files"
image_files = glob(os.path.join(IMAGE_FOLDER, "*.jpg"))
image_files = pd.Series(sorted(image_files)[:50], name="filename").apply(lambda f: os.path.split(f)[-1])
# images_files.index = image_files

if not os.path.exists("scores.csv"):
    scores = np.zeros((len(image_files), len(tags_EN)))
    scores = pd.DataFrame(scores, index=image_files, columns=tags_EN)
    scores.to_csv("scores.csv", index=True)
else:
    scores = pd.read_csv("scores.csv").set_index("filename")
    skip = scores[(scores.abs() > 0.).sum(1) == len(scores.columns)].index
    print(f"SKIPPING: {len(skip)} images! ({skip[:7]}, ...)") 
    image_files = image_files[~image_files.isin(skip)]

############################################################################
############################# INIT MODEL ################################### 



model = AutoModel.from_pretrained(checkpoint)
processor = AutoProcessor.from_pretrained(checkpoint)

############################################################################
############################# APPLY ######################################## 

def get_scores(imgs):
    inputs = processor(text=tags_EN, images=imgs, padding="max_length", return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs.logits_per_image.numpy()



BATCH_SIZE = 8
batches = np.array_split(image_files, len(image_files)//BATCH_SIZE)


for i, b in enumerate(tqdm(batches)):
    imgs = [Image.open(os.path.join(IMAGE_FOLDER, f)) for f in b]
    cur_scores = get_scores(imgs)

    scores.loc[b] = cur_scores
    scores.round(3).to_csv("scores.csv")
    # scores.extend(cur_scores)
    # processed.extend(b)
    for i in imgs: i.close()
    
# df = pd.DataFrame(scores, index=processed, columns=tags_EN)
# df = df.round(3)
    