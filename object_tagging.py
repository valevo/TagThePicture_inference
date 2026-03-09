import os
import pickle
import argparse
from glob import glob
from PIL import Image
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoProcessor, AutoModel, AutoModelForZeroShotObjectDetection

# checkpoint = "openai/clip-vit-large-patch14"
# checkpoint = "gzomer/clip-multilingual"
# checkpoint = "google/siglip-so400m-patch14-384"


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
        default="./object_scores/",
        help="Folder to write pickled outputs to. Default `./object_scores/`",
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

    save_folder = Path(args.output)

    image_files = glob(str(image_dir / "*.jpg"))
    image_files = pd.Series(sorted(image_files), name="filename").apply(
        lambda f: os.path.basename(f)
    )

    if not os.path.exists(save_folder):
        print(f"Creating {save_folder}...")
        os.makedirs(save_folder)
    else:
        print("Warning! Folder already exists!")

    # if not os.path.exists(save_fn):
    #     print(f"Creating {save_fn}...")
    #     scores = np.zeros((len(image_files), len(tags)))
    #     scores = pd.DataFrame(scores, index=image_files.tolist(), columns=tags)
    #     scores.to_csv(save_fn, index=True)
    # else:
    #     print(f"{save_fn} exists, checking which files to skip...")
    #     scores = pd.read_csv(save_fn).set_index("filename")
    #     skip = scores[(scores.abs() > 0.0).sum(1) == len(scores.columns)].index

    #     print(f"SKIPPING {len(skip)} images, as they already have scores!")

    #     image_files = image_files[~image_files.isin(skip.tolist())]

    ############################################################################
    ############################# INIT MODEL ###################################

    print("Initialising model...")
    print(f"Loading Checkpoint: {args.checkpoint} to {device}")

    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.checkpoint).to(
        device
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.checkpoint)

    ############################################################################
    ############################# APPLY ########################################

    def get_scores(imgs):
        inputs = processor(text=tags, images=imgs, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        return outputs

    print(f"Number of image_files: {len(image_files)}")
    batches = np.array_split(image_files, len(image_files) // args.batch_size)

    def pickle_outputs(iteration_number, filenames, outputs):
        to_pickle = (tuple(filenames), outputs)
        with open(save_folder / f"outputs_{iteration_number:04d}.pkl", "wb") as handle:
            pickle.dump(to_pickle, handle)

    for i, b in enumerate(tqdm(batches)):
        imgs = []
        used_files = []
        for file in b:
            try:
                imgs.append(Image.open(image_dir / file).convert("RGB"))
                used_files.append(file)
            except Exception as e:
                print(f"Exception loading file {image_dir / file}")
                print(e)
                continue

        cur_scores = get_scores(imgs)

        pickle_outputs(i, used_files, cur_scores)

        # scores.loc[b] = cur_scores
        # if i % 10:
        # scores.round(3).to_csv(save_fn)

        for i in imgs:
            i.close()


if __name__ == "__main__":
    main()
