import os
import argparse
from glob import glob
from time import time
from PIL import Image
from pathlib import Path

import threading
import queue

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", help="Directory of image files")
    parser.add_argument(
        "checkpoint",
        help="Model checkpoint to load; no default.",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="./object_scores.csv",
        help="CSV file to write results to. Default `object_scores.csv`",
    )
    parser.add_argument(
        "-t",
        "--terms",
        default="object_terms.csv",
        help="File of object terms. Default `object_terms.csv`",
    )
    parser.add_argument(
        "-l",
        "--language",
        default="en",
        help="Language of terms to use. Default `en`",
        choices=["en", "nl"],
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=32, help="Batch size. Default 32"
    )
    parser.add_argument(
        "-d",
        "--device",
        help="Device to compute on, e.g. `cuda`, `cpu`. If not specified, attempts default CUDA device, otherwise CPU",
    )

    args = parser.parse_args()

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Computing on: {device}")

    ############################################################################
    ############################# LOAD DATA ####################################

    print(f"Loading terms from {args.terms} and labels in {args.language.upper()}")
    terms = pd.read_csv(args.terms)
    term_heading = f"label_{args.language.lower()}"
    tags = list(terms[term_heading])

    image_dir = Path(args.images)

    image_files = glob(str(image_dir / "*.jpg"))
    image_files = pd.Series(image_files, name="filename").apply(
        lambda f: os.path.basename(f)
    )

    save_file = Path(args.output)

    COLUMN_NAMES = [
        "filename",
        "score",
        "box_x0",
        "box_y0",
        "box_x1",
        "box_y1",
        "tag",
        "tag_index",
    ]

    if os.path.exists(save_file):
        df = pd.read_csv(save_file)
        skip = set(df.filename.unique())
        # skip = scores[(scores.abs() > 0.0).sum(1) == len(scores.columns)].index
        image_files = image_files[~image_files.isin(skip)]
        print(f"SKIPPING {len(skip)} images, as they already have scores!")
    else:
        pd.DataFrame([], columns=COLUMN_NAMES).to_csv(save_file, index=False)



    from transformers import pipeline
    
    # Use any checkpoint from the hf.co/models?pipeline_tag=zero-shot-object-detection

    device = 1 if torch.cuda.is_available() else -1
    checkpoint = "iSEE-Laboratory/llmdet_large"
    checkpoint = "IDEA-Research/grounding-dino-base"
    detector = pipeline(model=checkpoint, 
                        task="zero-shot-object-detection",
                       device=1)

    
    results = []
    for f in tqdm(image_files):
        img = Image.open(image_dir / f)  # .convert("RGB")
        img.load()
        img = img.convert("RGB")

        detections = detector(
            img,
            candidate_labels=tags,
            threshold=0.1,
        )

        recs = []
        for f, cur in zip(files, detections):
            zipped = zip(
                cur["scores"],
                cur["boxes"],
                cur["text_labels"],
                cur["labels"],
                range(20),
            )
            for s, b, l_, l_id, _ in zipped:
                recs.append([f, round(float(s), 3), *map(int, b), l_, int(l_id)])

        results.extend(recs)




        
