import torch
import torchvision
import cv2
import os
from PIL import Image
import numpy as np
import tqdm


def get_unqs(mask):
    unqs = np.unique(np.array(mask, np.uint8))
    return unqs[1:-1]

pascal_voc = torchvision.datasets.VOCSegmentation(
    root="./pascal_voc",
    year="2012",
    image_set="train",
    download=False
)

class_sets = {}
pbar = tqdm.tqdm(range(len(pascal_voc.masks)))
for i in pbar:
    sample = pascal_voc.__getitem__(i)
    unqs = get_unqs(sample[1])
    for label in unqs:
        if not label in class_sets:
            class_sets[label] = []
        class_sets[label].append([pascal_voc.images[i], pascal_voc.masks[i]])
    pbar.set_description(f"Creating class sets: [{i}/{len(pascal_voc.masks)}]")

print(unqs)