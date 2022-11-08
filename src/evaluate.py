import os
import sys
import cv2
import yaml
import torch
import pandas as pd
import numpy as np
import albumentations as T
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns
import matplotlib

from matplotlib import cm
from PIL import Image
from utils import get_filenames, create_dir
from patchify import patchify, unpatchify

from training.loss import *
from training.trainer import eval
from training.metrics import MIoU
from training.dataset import ImagesFromFolder
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from scipy import ndimage
from skimage.restoration import denoise_tv_chambolle
from monai.metrics.confusion_matrix import get_confusion_matrix
from monai.metrics.hausdorff_distance import compute_hausdorff_distance
from monai.losses.dice import DiceFocalLoss

def gray_gamma(img, gamma):
    gray = img / 255.
    out = np.array(gray ** gamma)
    out = 255*out
    return out.astype('uint8')

def tv_denoising(img, alpha):
    gray = img / 255.
    out = denoise_tv_chambolle(gray, weight=alpha)
    out = out * 255
    return out.astype('uint8')

def get_segmentation_large(model, image, mask, pred_imgdir, pred_mskdir, pred_predsdir, imgw, imgh, gamma=2, alpha=0.03):
    save_image_filename = pred_imgdir + os.path.splitext(os.path.split(image)[1])[0] + '.png'
    save_mask_filename = pred_mskdir + os.path.splitext(os.path.split(image)[1])[0] + '.png'
    save_pred_filename = pred_predsdir + os.path.splitext(os.path.split(image)[1])[0] + '.png'
    save_ovly_filename = pred_predsdir + 'overlay/'+ os.path.splitext(os.path.split(image)[1])[0] + '.png'

    large_image = np.array(Image.open(image).convert('L'))
    large_mask = np.array(Image.open(mask).convert('L'))

    img = gray_gamma(large_image, gamma=gamma)
    img = tv_denoising(img, alpha=alpha)

    img_h, img_w = img.shape
    
    image_x = F.interpolate(torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float(), (imgh, imgw), mode='bilinear', align_corners=False).squeeze().numpy()
    mask_y = F.interpolate(torch.from_numpy(large_mask).unsqueeze(0).unsqueeze(0).float(), (imgh, imgw), mode='nearest').squeeze().numpy()

    pred, iou = predict(model, image_x, mask_y)

    pred = F.interpolate(torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float(), (img_h, img_w), mode='nearest').squeeze().numpy()

    shape_1 = (pred.shape[0], pred.shape[1], 3)

    rec_pred_rgb = np.zeros(shape=shape_1, dtype='uint8')
    
    norm = matplotlib.colors.Normalize(vmin=0, vmax=pred.max())

    for idx in range(1, int(pred.max())+1):
        rec_pred_rgb[..., 0] = np.where(pred == idx, cm.hsv(norm(idx), bytes=True)[0], rec_pred_rgb[..., 0])
        rec_pred_rgb[..., 1] = np.where(pred == idx, cm.hsv(norm(idx), bytes=True)[1], rec_pred_rgb[..., 1])
        rec_pred_rgb[..., 2] = np.where(pred == idx, cm.hsv(norm(idx), bytes=True)[2], rec_pred_rgb[..., 2])

    rec_pred_rgb = Image.fromarray(rec_pred_rgb)
    rec_img = Image.fromarray((large_image))
    rec_msk = Image.fromarray(((large_mask / large_mask.max()) * 255).astype('uint8'))

    rec_img.save(save_image_filename)
    rec_msk.save(save_mask_filename)
    rec_pred_rgb.save(save_pred_filename)

    rec_pred_rgb = rec_pred_rgb.convert("RGBA")
    rec_img = rec_img.convert("RGBA")
    rec_msk = rec_msk.convert("RGBA")

    overlayed1 = Image.blend(rec_msk, rec_pred_rgb, 0.5)
    overlayed2 = Image.blend(rec_img, overlayed1, 0.5)
    overlayed2.save(save_ovly_filename)

    return iou

def get_segmentation_patches(model, image, mask, pred_imgdir, pred_mskdir, pred_predsdir, imgzs=256):
    save_image_filename = pred_imgdir + os.path.splitext(os.path.split(image)[1])[0] + '.png'
    save_mask_filename = pred_mskdir + os.path.splitext(os.path.split(image)[1])[0] + '.png'
    save_pred_filename = pred_predsdir + os.path.splitext(os.path.split(image)[1])[0] + '.png'
    save_ovly_filename = pred_predsdir + 'overlay/'+ os.path.splitext(os.path.split(image)[1])[0] + '.png'

    large_image = np.array(Image.open(image).convert('L'))
    large_mask = np.array(Image.open(mask).convert('L'))

    large_image = np.pad(large_image, [(8, ), (0, )], 'constant', constant_values=0)
    large_mask = np.pad(large_mask, [(8, ), (0, )], 'constant', constant_values=0)

    patches_images = patchify(large_image, (imgzs, imgzs), step=imgzs)
    patches_masks = patchify(large_mask, (imgzs, imgzs), step=imgzs)

    preds = []
    iou_list = []
    y_trues = []
    y_preds = []

    for i in range(patches_images.shape[0]):
        for j in range(patches_images.shape[1]):
            image_x = patches_images[i, j, :, :]
            mask_y = patches_masks[i, j, :, :]
            pred, iou = predict(model, image_x, mask_y)
            # print(mean_iou)
            y_trues.append(mask_y.reshape(-1))
            y_preds.append(pred.reshape(-1))
            iou_list.append(iou)
            preds.append(pred)

    preds = np.reshape(preds, patches_images.shape)
    preds = np.array(preds)

    
    rec_img = unpatchify(patches=patches_images, imsize=(imgzs * preds.shape[0], imgzs * preds.shape[1]))
    rec_msk = unpatchify(patches=patches_masks, imsize=(imgzs * preds.shape[0], imgzs * preds.shape[1]))
    rec_pred = unpatchify(patches=preds, imsize=(imgzs * preds.shape[0], imgzs * preds.shape[1]))

    shape_1 = (rec_pred.shape[0], rec_pred.shape[1], 3)

    rec_pred_rgb = np.zeros(shape=shape_1, dtype='uint8')

    norm = matplotlib.colors.Normalize(vmin=0, vmax=rec_pred.max())

    for idx in range(1, int(preds.max())+1):
        rec_pred_rgb[..., 0] = np.where(rec_pred == idx, cm.hsv(norm(idx), bytes=True)[0], rec_pred_rgb[..., 0])
        rec_pred_rgb[..., 1] = np.where(rec_pred == idx, cm.hsv(norm(idx), bytes=True)[1], rec_pred_rgb[..., 1])
        rec_pred_rgb[..., 2] = np.where(rec_pred == idx, cm.hsv(norm(idx), bytes=True)[2], rec_pred_rgb[..., 2])

    rec_pred_rgb = Image.fromarray(rec_pred_rgb[8:504, :, :])
    rec_img = Image.fromarray((rec_img[8:504, :]))
    rec_msk = Image.fromarray(((rec_msk[8:504, :] / rec_msk.max()) * 255).astype('uint8'))

    rec_img.save(save_image_filename)
    rec_msk.save(save_mask_filename)
    rec_pred_rgb.save(save_pred_filename)

    rec_pred_rgb = rec_pred_rgb.convert("RGBA")
    rec_img = rec_img.convert("RGBA")
    rec_msk = rec_msk.convert("RGBA")

    overlayed1 = Image.blend(rec_msk, rec_pred_rgb, 0.5)
    overlayed2 = Image.blend(rec_img, overlayed1, 0.5)
    overlayed2.save(save_ovly_filename)
    return np.array(iou_list).mean(axis=0)

def predict(model, x_image, y_mask):
    device = torch.device("cuda")

    iou_fn = MIoU(activation='softmax', ignore_background=True, device=device)

    transforms = T.Compose([
                T.Normalize(mean=0.1338, std=0.1466) # CONTROL: 0.0389,  0.1036,  # FULL: 0.1338, 0.1466
    ])
    image = np.expand_dims(x_image, axis=-1)
    y_mask = torch.tensor(y_mask, device=device, dtype=torch.long).unsqueeze(0).unsqueeze(0)
    # image = np.repeat(image, 3, axis=-1)

    image = transforms(image=image)
    image = image['image'].transpose((2, 0, 1))
    image = torch.tensor(image, dtype=torch.float, device=device)
    image = image.unsqueeze(0)

    y_pred = model(image)
    iou = iou_fn(y_pred, y_mask)
    iou = np.where(iou <= 1e-1, 1, iou)
    y_pred = F.softmax(y_pred, dim=1)
    y_pred = torch.argmax(y_pred, dim=1)
    y_pred = y_pred.squeeze(0).detach().cpu().numpy()
    # print(iou.mean())
    return y_pred, iou

def run_evaluation(model, images, masks, image_sizeh, image_sizew):
    device = torch.device("cuda")
    loss_fn = WeightedCrossEntropyDice(device=device, lambda_=0.6, class_weights=[1 for _ in range(5)])
    # loss_fn = DiceFocalLoss(to_onehot_y=True, softmax=True)
    # loss_fn = CrossEntropyLoss(device=device)

    val_transforms = T.Compose([
        T.Resize(image_sizeh, image_sizew),
        T.Normalize(mean=0.1338, std=0.1466)
        ])

    val_ds = ImagesFromFolder(image_dir=images,
                              mask_dir=masks,
                              transform=val_transforms,
                              preprocess_input=None
                              )

    val_loader = DataLoader(val_ds,
                            batch_size=1,
                            num_workers=12,
                            pin_memory=True,
                            shuffle=False
                            )

    loss_eval, iou_eval, pixel_acc_list, dice_list, precision_list, recall_list = eval(model, val_loader, loss_fn, device)

    div_iou = np.array(iou_eval).mean(axis=0)
    div_pxl = np.array(pixel_acc_list).mean(axis=0)
    div_dice = np.array(dice_list).mean(axis=0)
    div_pre = np.array(precision_list).mean(axis=0)
    div_rec = np.array(recall_list).mean(axis=0)

    print(f'IoU: \n', div_iou, div_iou.mean())
    print(f'P. Acc: \n', div_pxl)
    print(f'Dice: \n', div_dice, div_dice.mean())
    print(f'Prec.: \n', div_pre, div_pre.mean())
    print(f'Recall: \n', div_rec, div_rec.mean())

    print('Loss Eval: ', loss_eval)

def main():
    base_path = 'logs/2022-09-18_22_35_46'
    model_path = os.path.join(base_path, 'checkpoints/model.pth')

    with open(os.path.join(base_path, 'experiment_cfg.yaml'), "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    paths = cfg['paths']
    base = paths['base']
    general = cfg['general']
    image_sizeh = general['img_sizeh']
    image_sizew = general['img_sizew']
    val_imgdir = os.path.join(base, paths['val_imgdir'])
    val_mskdir = os.path.join(base, paths['val_mskdir'])

    test_imgdir = os.path.join(base, paths['test_imgdir'])
    test_mskdir = os.path.join(base, paths['test_mskdir'])

    pred_imgdir = os.path.join(base, paths['save_testimg'])
    pred_mskdir = os.path.join(base, paths['save_testmsk'])
    pred_predsdir = os.path.join(base, paths['save_testpred'])

    create_dir(pred_imgdir)
    create_dir(pred_mskdir)
    create_dir(pred_predsdir)
    create_dir(os.path.join(pred_predsdir, 'overlay'))

    files = get_filenames(test_imgdir, 'png')
    filesM = get_filenames(test_mskdir, 'png')
    
    model = torch.load(model_path)

    # run_evaluation(model, val_imgdir, val_mskdir, image_sizeh, image_sizew)

    iou = []
    fileImage = []

    for im, mk in zip(files, filesM):
        # iou_item = get_segmentation_patches(model, im, mk, pred_imgdir, pred_mskdir, pred_predsdir)
        iou_item = get_segmentation_large(model, im, mk, pred_imgdir, pred_mskdir, pred_predsdir, imgw=496, imgh=496, gamma=1.5, alpha=0.01)
        iou.append(iou_item)
        fileImage.append(os.path.split(im)[1])

    iou = np.array(iou)

    # df2 = pd.DataFrame({'file': fileImage,
    #                     'iou': iou,
    #                     })
    # df2.to_csv('predictions.csv')
    # iou.sort()
    hmean = np.mean(iou, axis=0)
    hstd = np.std(iou, axis=0)
    # _, q1, q3 = np.percentile(iou, 50), np.percentile(iou, 25), np.percentile(iou, 75)
    # sigma = hstd
    # mu = hmean
    # iqr = 1.5 * (q3 - q1)
    # x1 = np.linspace(q1 - iqr, q1)
    # x2 = np.linspace(q3, q3 + iqr)
    # pdf1 = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x1 - mu)**2 / (2 * sigma**2))
    # pdf2 = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x2 - mu)**2 / (2 * sigma**2))

    print(f'Mean per class:{hmean}, Std per class:{hstd}')
    print(f'Mean :{np.mean(iou)}, Std:{np.std(iou)}')
    # pdf = stats.norm.pdf(iou, hmean, hstd)
    # pl.plot(iou, pdf, '-o', label=f'Mean:{hmean:0.3f}, Std:{hstd:0.3f}, Q1:{q1:0.3f}, Q3:{q3:0.3f}')

    arran = np.linspace(0.5, 1, num=(len(iou)//10))
    plt.hist(iou.mean(), bins=arran, edgecolor='black')
    # pl.fill_between(x1, pdf1, 0, alpha=.6, color='green')
    # pl.fill_between(x2, pdf2, 0, alpha=.6, color='green')
    # plt.xlim([0.4, 1.1])
    plt.xlabel('IoU', fontsize=18, fontweight='bold')
    plt.ylabel('No. Images', fontsize=18, fontweight='bold')
    # plt.legend(loc='best')
    plt.savefig('train.png')


if __name__ == '__main__':
    main()