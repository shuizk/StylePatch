import numpy as np
import os
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import cv2
import pandas as pd
import matplotlib.pyplot as plt

dirs = ["StylePatch/input/000/", 
        "StylePatch/result/AP/vgg19/patch_images/000/",
        "StylePatch/result/LaVAN/vgg19/patch_images/000/",
        "StylePatch/result/PSGAN/vgg19/patch_images/000/",
        "StylePatch/result/Style/vgg19/patch_images/000/"
        ]

origins = os.listdir(dirs[0])
origins.sort(key=lambda x:int(x.split(".")[0]))
APs = os.listdir(dirs[1])
APs.sort(key=lambda x:int(x.split("_")[0]))
LaVANs = os.listdir(dirs[2])
LaVANs.sort(key=lambda x:int(x.split("_")[0]))
PS_GANs = os.listdir(dirs[3])
PS_GANs.sort(key=lambda x:int(x.split("_")[0]))
Styles = os.listdir(dirs[4])
Styles.sort(key=lambda x:int(x.split("_")[0]))
weighted = os.listdir("WAP/cam_npy/")
weighted.sort(key=lambda x:int(x.split("_")[0]))
patch_shape = [50,50]
SSMI_full = []
PSNR_full = []
SSMI_mask = []
PSNR_mask = []

for i in range(len(Styles)):
    index = int(Styles[i].split("_")[0])

    origin = cv2.imread(dirs[0]+origins[index])
    origin = cv2.resize(origin,(224,224))
    ap = cv2.imread(dirs[1]+APs[i])
    lavan = cv2.imread(dirs[2]+LaVANs[i])
    psgan = cv2.imread(dirs[3]+PS_GANs[i])
    psgan = cv2.resize(psgan,(224,224))
    style = cv2.imread(dirs[4]+Styles[i])


    # print("SSIM")
    # print(round(compare_ssim(origin, ap, multichannel=True),4))
    # print(round(compare_ssim(origin, lavan, multichannel=True),4))
    # print(round(compare_ssim(origin, psgan, multichannel=True),4))
    # print(round(compare_ssim(origin, style, multichannel=True),4))
    SSMI_full.append([
        round(compare_ssim(origin, ap, multichannel=True),4),
        round(compare_ssim(origin, lavan, multichannel=True),4),
        round(compare_ssim(origin, psgan, multichannel=True),4),
        round(compare_ssim(origin, style, multichannel=True),4)
    ])
    # print("PSNR")
    # print(round(compare_psnr(origin, ap),4))
    # print(round(compare_psnr(origin, lavan),4))
    # print(round(compare_psnr(origin, psgan),4))
    # print(round(compare_psnr(origin, style),4))
    PSNR_full.append([
        round(compare_psnr(origin, ap),4),
        round(compare_psnr(origin, lavan),4),
        round(compare_psnr(origin, psgan),4),
        round(compare_psnr(origin, style),4)
    ])
    mask = np.load(f"WAP/cam_npy/{weighted[index]}", allow_pickle=True)
    mask_x = np.nonzero(mask[0][0])[0][0]
    mask_y = np.nonzero(mask[0][0])[1][0]

    origin = origin[mask_x: mask_x+patch_shape[0], mask_y: mask_y+patch_shape[1]]
    ap = ap[mask_x: mask_x+patch_shape[0], mask_y: mask_y+patch_shape[1]]
    lavan = lavan[mask_x: mask_x+patch_shape[0], mask_y: mask_y+patch_shape[1]]
    psgan = psgan[mask_x: mask_x+patch_shape[0], mask_y: mask_y+patch_shape[1]]
    style = style[mask_x: mask_x+patch_shape[0], mask_y: mask_y+patch_shape[1]]

    # print("SSIM mask")
    # print(round(compare_ssim(origin, ap, multichannel=True),4))
    # print(round(compare_ssim(origin, lavan, multichannel=True),4))
    # print(round(compare_ssim(origin, psgan, multichannel=True),4))
    # print(round(compare_ssim(origin, style, multichannel=True),4))
    SSMI_mask.append([
        round(compare_ssim(origin, ap, multichannel=True),4),
        round(compare_ssim(origin, lavan, multichannel=True),4),
        round(compare_ssim(origin, psgan, multichannel=True),4),
        round(compare_ssim(origin, style, multichannel=True),4)
    ])
    # print("PSNR mask")
    # print(round(compare_psnr(origin, ap),4))
    # print(round(compare_psnr(origin, lavan),4))
    # print(round(compare_psnr(origin, psgan),4))
    # print(round(compare_psnr(origin, style),4))
    PSNR_mask.append([
        round(compare_psnr(origin, ap),4),
        round(compare_psnr(origin, lavan),4),
        round(compare_psnr(origin, psgan),4),
        round(compare_psnr(origin, style),4)
    ])

SSMI_full = pd.DataFrame(SSMI_full)

temps = [SSMI_full, PSNR_full, SSMI_mask, PSNR_mask]
titles = ['SSMI_FULL','PSNR_FULL','SSMI_MASK','PSNR_MASK']
for temp,title in zip(temps,titles):
    temp_pd = pd.DataFrame(temp)
    name = ['StylePatch','LaVAN','PS-GAN','AP']
    ap = temp_pd[0].mean()
    lavan = temp_pd[1].mean()
    psgan = temp_pd[2].mean()
    style = temp_pd[3].mean()
    value = [style, lavan, psgan, ap]

    fig, ax = plt.subplots()
    if title == 'SSMI_FULL':
        ax.set_xlim((0.9, 1))
    ax.barh(name, value)
    ax.set_title(title)
    ax.set_xlabel('Score')
    plt.savefig(f'./StylePatch/{title}.png')