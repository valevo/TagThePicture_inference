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

    num_workers = 4
    image_queue = queue.Queue(num_workers * 16)

    def loader_worker(paths):
        print("Loader worker started")
        for path in paths:
            try:
                img = Image.open(image_dir / path)
                img.load()
                img_size = (img.height, img.width)
                img = img.convert("RGB").resize((960, 960))
                image_queue.put((path, img, img_size))
            except Exception as e:
                print(f"Loader failed on path {path}: {e}")
                continue

    threads = []
    # chunks = [image_files[i::num_workers] for i in range(num_workers)]
    chunks = np.array_split(image_files, num_workers)

    for chunk in chunks:
        t = threading.Thread(target=loader_worker, args=(chunk,), daemon=True)
        t.start()
        threads.append(t)

    ############################################################################
    ############################# INIT MODEL ###################################

    print("Initialising model...")
    print(f"Loading Checkpoint: {args.checkpoint} to {device}")

    processor = AutoProcessor.from_pretrained(args.checkpoint)
    torch.backends.cudnn.benchmark = True
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.checkpoint).to(
        device
    )
    model.eval()

    ############################################################################
    ############################# APPLY ########################################

    batch_queue = queue.Queue(32)

    def assembler():
        print("Batch Assember started")
        files = []
        imgs = []
        img_sizes = []
        finished = False
        processor_asm = AutoProcessor.from_pretrained(args.checkpoint)

        while not finished:
            while len(imgs) < args.batch_size:
                try:
                    file, img, img_size = image_queue.get(timeout=1)
                    imgs.append(img)
                    files.append(file)
                    img_sizes.append(img_size)
                except queue.Empty:
                    # check if threads are finished
                    working = any([t.is_alive() for t in threads])
                    if working:
                        print("[assembler] waiting on workers...")
                        continue

                    # no more images to load
                    finished = True
                    break

            if len(imgs) == 0:
                break

            start = time()
            inputs = processor_asm(
                text=tags, images=imgs, return_tensors="pt", do_resize=True
            )
            elapsed = time() - start
            if elapsed > 10:
                print(f"[assembler] processor took {elapsed:.2f}s")
            batch_queue.put((files, img_sizes, inputs))
            for im in imgs:
                im.close()
            imgs = []
            files = []
            img_sizes = []

    assembler_threads = []

    for _ in range(2):
        assembler_thread = threading.Thread(target=assembler, daemon=True)
        assembler_thread.start()
        assembler_threads.append(assembler_thread)

    results = []
    batch_count = 0

    print(f"Number of image_files: {len(image_files)}")
    total_batches = len(image_files) // args.batch_size
    pbar = tqdm(total=total_batches)

    time_stats = {"batch": [], "model": [], "recs": [], "post": []}
    start = time()
    checkpoint_start = time()
    while True:
        try:
            batch = batch_queue.get(timeout=1)
        except queue.Empty:
            working = any([t.is_alive() for t in assembler_threads])
            if working:
                continue
            print("No more batches and all assembler thread finished, exiting")
            break

        files, img_sizes, inputs = batch

        time_stats["batch"].append(time() - start)
        start = time()

        with torch.no_grad():
            outputs = model(**inputs.to(device))

        inputs = inputs.to("cpu")

        time_stats["model"].append(time() - start)
        start = time()

        detections = processor.post_process_grounded_object_detection(
            outputs,
            threshold=0.1,
            target_sizes=img_sizes,  # [(i.height, i.width) for i in imgs],
            text_labels=[tags] * len(img_sizes),
        )

        time_stats["post"].append(time() - start)
        start = time()

        recs = []
        for f, cur in zip(files, detections):
            zipped = zip(
                cur["scores"],
                cur["boxes"],
                cur["text_labels"],
                cur["labels"],
                range(20),
            )
            for s, b, l_, l_id, _ in sorted(zipped, key=lambda t: t[0], reverse=True):
                recs.append([f, round(float(s), 3), *map(int, b), l_, int(l_id)])

        results.extend(recs)
        time_stats["recs"].append(time() - start)

        if batch_count % 10 == 0 and batch_count > 0:
            start = time()
            cur_df = pd.DataFrame(results, columns=COLUMN_NAMES)
            cur_df.to_csv(save_file, index=False, header=False, mode="a")
            results = []
            save_time = time() - start
            checkpoint_time = time() - checkpoint_start

            print(f"\nTime report for batch {batch_count}/{total_batches} ({100*batch_count/total_batches:.2f}%):")
            print(f"  Checkpoint completed in {checkpoint_time:.2f}s, [{checkpoint_time/10:.2f}s/it]")
            print(
                f"  Mean:  Batch: {(sum(time_stats['batch']) / 10):.2f}s   Model: {(sum(time_stats['model']) / 10):.2f}s   Rec: {(sum(time_stats['recs']) / 10):.2f}s   Post: {(sum(time_stats['post']) / 10):.2f}s"
            )
            print(
                f"   Max:  Batch: {(max(time_stats['batch'])):.2f}s   Model: {(max(time_stats['model'])):.2f}s   Rec: {(max(time_stats['recs'])):.2f}s   Post: {(max(time_stats['post'])):.2f}s"
            )
            print(f"  CSV write time: {save_time:.2f}s")
            time_stats = {"batch": [], "model": [], "recs": [], "post": []}

            checkpoint_start = time()

        batch_count += 1
        pbar.update(1)
        start = time()

    cur_df = pd.DataFrame(results, columns=COLUMN_NAMES)
    cur_df.to_csv(save_file, index=False, header=False, mode="a")

    pbar.close()

    # def get_scores(imgs):
    #     inputs = processor(text=tags, images=imgs, return_tensors="pt").to(device)
    #
    #     with torch.no_grad():
    #         outputs = model(**inputs)
    #
    #     return outputs
    #
    # def get_detections(files, imgs):
    #     outputs = get_scores(imgs)
    #
    #     detections = processor.post_process_grounded_object_detection(
    #         outputs,
    #         threshold=0.1,
    #         target_sizes=[(i.height, i.width) for i in imgs],
    #         text_labels=[tags] * len(imgs),
    #     )
    #
    #     recs = []
    #     for f, cur in zip(files, detections):
    #         zipped = zip(
    #             cur["scores"],
    #             cur["boxes"],
    #             cur["text_labels"],
    #             cur["labels"],
    #             range(20),
    #         )
    #         for s, b, l_, l_id, _ in zipped:
    #             recs.append([f, round(float(s), 3), *map(int, b), l_, int(l_id)])
    #
    #     return recs
    #
    # print(f"Number of image_files: {len(image_files)}")
    # batches = np.array_split(image_files, len(image_files) // args.batch_size)
    #
    # results = []
    # for i, b in enumerate(tqdm(batches)):
    #     imgs = []
    #     used_files = []
    #     for file in b:
    #         try:
    #             imgs.append(Image.open(image_dir / file).convert("RGB"))
    #             used_files.append(file)
    #         except Exception as e:
    #             print(f"Exception loading file {image_dir / file}")
    #             print(e)
    #             continue
    #
    #     detected = get_detections(used_files, imgs)
    #     results.extend(detected)
    #
    #     if i % 10 == 0:
    #         cur_df = pd.DataFrame(results, columns=COLUMN_NAMES)
    #         cur_df.to_csv(save_file, index=False, header=False, mode="a")
    #         results = []
    #
    #     for i in imgs:
    #         i.close()
    #
    # cur_df = pd.DataFrame(results, columns=COLUMN_NAMES)
    # cur_df.to_csv(save_file, index=False, header=False, mode="a")


if __name__ == "__main__":
    main()
