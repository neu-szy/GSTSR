import argparse
import cv2
import numpy as np
from os import path as osp
import os
from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.utils import scandir
from basicsr.utils.matlab_functions import bgr2ycbcr


def main(args):
    """Calculate PSNR and SSIM for images.
    """
    psnr_all = []
    ssim_all = []
    img_list_gt = sorted(list(scandir(args.gt, recursive=True, full_path=True)))
    img_list_restored = sorted(list(scandir(args.restored, recursive=True, full_path=True)))

    if args.test_y_channel:
        print('Testing Y channel.')
    else:
        print('Testing RGB channels.')

    for i, img_path in enumerate(img_list_gt):
        basename, ext = osp.splitext(osp.basename(img_path))
        img_gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        if args.suffix == '':
            img_path_restored = img_list_restored[i]
        else:
            img_path_restored = osp.join(args.restored, basename + args.suffix + ext)
        img_restored = cv2.imread(img_path_restored, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        if args.correct_mean_var:
            mean_l = []
            std_l = []
            for j in range(3):
                mean_l.append(np.mean(img_gt[:, :, j]))
                std_l.append(np.std(img_gt[:, :, j]))
            for j in range(3):
                # correct twice
                mean = np.mean(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] - mean + mean_l[j]
                std = np.std(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] / std * std_l[j]

                mean = np.mean(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] - mean + mean_l[j]
                std = np.std(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] / std * std_l[j]

        if args.test_y_channel and img_gt.ndim == 3 and img_gt.shape[2] == 3:
            img_gt = bgr2ycbcr(img_gt, y_only=True)
            img_restored = bgr2ycbcr(img_restored, y_only=True)

        # calculate PSNR and SSIM
        psnr = calculate_psnr(img_gt * 255, img_restored * 255, crop_border=args.crop_border, input_order='HWC')
        ssim = calculate_ssim(img_gt * 255, img_restored * 255, crop_border=args.crop_border, input_order='HWC')
        print(f'{i+1:3d}: {basename:25}. \tPSNR: {psnr:.6f} dB, \tSSIM: {ssim:.6f}')
        # 原来是if not psnr
        if psnr:
            psnr_all.append(psnr)
        ssim_all.append(ssim)
    print(args.gt)
    print(args.restored)
    print(f'Average: PSNR: {sum(psnr_all) / len(psnr_all):.6f} dB, SSIM: {sum(ssim_all) / len(ssim_all):.6f}')


if __name__ == '__main__':
    restored_dir = ['/home/zhiyi/data/medical/belly/x{}_valid_bicubic_sr'] + \
             ['/home/zhiyi/data/medical/lung/x{}_valid_bicubic_sr']
    gt_dir = [f'/home/zhiyi/data/medical/{name}/hr_valid' for name in ["belly", "lung"]]
    for i in range(len(gt_dir)):
        for scale in [2, 4, 8]:
            try:
                if not os.listdir(restored_dir[i].format(scale)):
                    continue
            except:
                continue
            parser = argparse.ArgumentParser()
            parser.add_argument('--gt', type=str, default=gt_dir[i], help='Path to gt (Ground-Truth)')
            parser.add_argument('--restored', type=str, default=restored_dir[i].format(scale), help='Path to restored images')
            parser.add_argument('--crop_border', type=int, default=scale, help='Crop border for each side')
            parser.add_argument('--suffix', type=str, default='', help='Suffix for restored images')
            parser.add_argument(
                '--test_y_channel',
                action='store_true',
                help='If True, test Y channel (In MatLab YCbCr format). If False, test RGB channels.')
            parser.add_argument('--correct_mean_var', action='store_true', help='Correct the mean and var of restored images.')
            args = parser.parse_args()
            main(args)
