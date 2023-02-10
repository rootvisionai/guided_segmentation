# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 18:50:02 2022

@author: tekin.evrim.ozmermer
"""

from PIL import Image
import os
import numpy as np

raw_dir = "cloth_segmentation_raw"
mask_dir = "masks"
img_dir  = "images"
dataset_dir = raw_dir.replace("_raw","")

try:
    os.mkdir(dataset_dir)
except:
    pass

try:
    os.mkdir(os.path.join(dataset_dir, mask_dir))
except:
    pass

try:
    os.mkdir(os.path.join(dataset_dir, img_dir))
except:
    pass

imgs_dir = os.path.join(raw_dir, img_dir)
masks_dir = os.path.join(raw_dir, mask_dir)

image_paths = [os.path.join(imgs_dir, elm) for elm in os.listdir(imgs_dir)]
mask_paths = [os.path.join(masks_dir, elm) for elm in os.listdir(masks_dir)]

for cnt, paths in enumerate(zip(mask_paths, image_paths)):
    mask = Image.open(paths[0])
    image = Image.open(paths[1])
    
    mask = np.array(mask, dtype=np.uint8)
    image = np.array(image, dtype=np.uint8)
    
    i = 0
    unq = 1
    mask_to_save = np.zeros_like(mask)
    mask_to_save[np.where(mask!=0)] = 255
    new_mask_path = os.path.join(dataset_dir, mask_dir,
                                 paths[0].split("\\")[-1].replace(".png", f"_{i}-{unq}.png"))
    
    Image.fromarray(mask_to_save).save(new_mask_path)
    
    new_img_path = os.path.join(dataset_dir, img_dir,
                                paths[0].split("\\")[-1].replace(".png", f"_{i}-{unq}.png"))
    
    Image.fromarray(image).save(new_img_path)
    
    print(f"{cnt} | {paths[0], paths[1]}")
        
    
    
    
