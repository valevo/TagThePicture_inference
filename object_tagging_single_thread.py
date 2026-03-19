import os
import argparse
import time
from glob import glob
from PIL import Image
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


def get_inputs(path, filenames, processor):
    loaded_images = []
    loaded_names = []
    original_sizes = []
    for f in filenames:
        try:
            img = Image.open(path / f)
            img.load()
            # img_size = (img.height, img.width)
            img = img.convert("RGB").resize((960, 960))
            # image_queue.put((path, img, img_size))
            loaded_images.append(img)
            loaded_names.append(f)
            original_sizes.append(img.size)
        except Exception as e:
            print(f"Loader failed on path {path}: {e}")
            continue

    inputs = processor(images=loaded_images, return_tensors="pt", do_resize=True)

    for im in loaded_images:
        im.close()

    return inputs, loaded_names, original_sizes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", help="Directory of image files")
    parser.add_argument(
        "checkpoint",
        help="Model checkpoint to load; no default.",
        default="google/owlv2-base-patch16",
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
    print(f"image_dir: {image_dir}")

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
        "rank",
    ]

    if os.path.exists(save_file):
        df = pd.read_csv(save_file)
        skip = set(df.filename.unique())
        # skip = scores[(scores.abs() > 0.0).sum(1) == len(scores.columns)].index
        image_files = image_files[~image_files.isin(skip)]
        print(f"SKIPPING {len(skip)} images, as they already have scores!")
    else:
        pd.DataFrame([], columns=COLUMN_NAMES).to_csv(save_file, index=False)

    processor = AutoProcessor.from_pretrained(args.checkpoint)
    torch.backends.cudnn.benchmark = True
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.checkpoint).to(
        device
    )
    model.eval()

    total_batches = len(image_files) // args.batch_size
    batches = np.array_split(image_files, total_batches)

    results = []

    t0 = time.time()
    for batch_count, batch_filenames in enumerate(tqdm(batches)):
        inputs, loaded_files, original_sizes = get_inputs(
            image_dir, batch_filenames, processor
        )
        text_inputs = processor(text=[tags] * len(loaded_files), return_tensors="pt")

        inputs = inputs | text_inputs

        with torch.inference_mode():
            outputs = model(**inputs.to(device))

        detections = processor.post_process_grounded_object_detection(
            outputs,
            threshold=0.1,
            # target_sizes=original_sizes,  # [(i.height, i.width) for i in imgs],
            text_labels=[tags] * len(loaded_files),
        )

        for f, cur in zip(loaded_files, detections):
            zipped = zip(
                cur["scores"].cpu().numpy().round(3),
                cur["boxes"].cpu().numpy().round(3),
                cur["text_labels"],
                cur["labels"],
            )
            zipped = zip(sorted(zipped, key=lambda t: t[0], reverse=True), range(20))
            for (s, b, l_, l_id), i in zipped:
                results.append([f, s, *b, l_, i])

        if batch_count % 10 == 0:  # and batch_count > 0:
            t1 = time.time()
            elapsed = t1 - t0
            print()
            print(
                f"{batch_count:04d}/{total_batches:04d} {elapsed / 3600:.2f}hrs [{(batch_count + 1) / elapsed:.2f}it/s]",
                flush=True,
            )
            cur_df = pd.DataFrame(results, columns=COLUMN_NAMES)
            cur_df.to_csv(save_file, index=False, header=False, mode="a")
            results = []

    cur_df = pd.DataFrame(results, columns=COLUMN_NAMES)
    cur_df.to_csv(save_file, index=False, header=False, mode="a")


if __name__ == "__main__":
    main()
