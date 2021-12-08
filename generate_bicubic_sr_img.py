import os
import numpy as np
from PIL import Image

def main(dir, save_dir, scale):
    img_list = os.listdir(dir)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    for img_name in img_list:
        img_path = os.path.join(dir, img_name)
        img = Image.open(img_path)
        w, h = img.size
        img = img.resize((w*scale, h*scale), Image.BICUBIC)
        save_path = os.path.join(save_dir, img_name)
        img.save(save_path)

for name in ["belly", "lung"]:
    for scale in [2, 4, 8]:
        main(f"/home/zhiyi/data/medical/{name}/x{scale}_valid",
             f"/home/zhiyi/data/medical/{name}/x{scale}_valid_bicubic_sr",
             scale)
