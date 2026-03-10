from glob import glob

import requests
from PIL import Image
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection

device = "cuda:1"



terms = pd.read_csv(args.terms)
term_heading = f"label_{args.language.lower()}"
tags = list(terms[term_heading])


imgs = [Image.open(f) for f in 
        sorted(glob("/mnt/vale_stick/TTP/TTP_images/*.jpg"))[:32]]

inputs = processor(text=tags, images=imgs, return_tensors="pt").to(device)
#                                                                ^^^^^^^^^^^


processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble", device_map=device)
#                                                                                      ^^^^^^^^^^^^^

with torch.no_grad(): # tried with and without this line
    outputs = model(**inputs)

# target_sizes = torch.Tensor([image.size[::-1]])
# target_sizes
# # Convert outputs (bounding boxes and class logits) to Pascal VOC Format (xmin, ymin, xmax, ymax)
# results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
# i = 0  # Retrieve predictions for the first image for the corresponding text queries
# text = texts[i]
# boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]


# batches = np.array_split(image_files, len(image_files) // args.batch_size)
