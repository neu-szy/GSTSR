# 生成bicubic下采样的图片
import os
from PIL import Image

def main(data_dir="/home/zhiyi/data/medical/belly", scale=2):
    # Scale factor
    hr_dir = os.path.join(data_dir, "GTmod12")
    assert os.path.isdir(hr_dir), "hr路径不存在"
    lr_dir = os.path.join(data_dir, f"x{scale}")
    if not os.path.isdir(lr_dir):
        os.mkdir(lr_dir)
    hr_list = os.listdir(hr_dir)

    for i in range(len(hr_list)):
        # print(f"第{i}张")
        img_hr = os.path.join(hr_dir, hr_list[i])
        img_hr = Image.open(img_hr)
        dsize = (img_hr.size[0]//scale, img_hr.size[1]//scale)
        img_lr = img_hr.resize(dsize, Image.BICUBIC)
        lr_path = os.path.join(lr_dir, hr_list[i])
        img_lr.save(lr_path, quality=95)


if __name__ == "__main__":
    d = ["B100", "manga109", "urban100"]
    for i in [8]:
        main(f"/home/zhiyi/data/Set14", i)