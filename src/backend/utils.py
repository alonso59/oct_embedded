import os
import torch
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def split_data(extention, train_size=0.8, images_path=None, masks_path=None, train_images_dir=None, val_images_dir=None, train_masks_dir=None, val_masks_dir=None):

    create_dir(train_images_dir)
    create_dir(val_images_dir)
    create_dir(train_masks_dir)
    create_dir(val_masks_dir)

    x = get_filenames(images_path, extention)
    y = get_filenames(masks_path, extention)

    X_train, X_val, y_train, y_val = train_test_split(
        x, y, train_size=train_size, shuffle=True
    )

    for i, j in zip(X_train, y_train):
        shutil.copy(os.path.join(i), train_images_dir)
        shutil.copy(os.path.join(j), train_masks_dir)

    for i, j in zip(X_val, y_val):
        shutil.copy(os.path.join(i), val_images_dir)
        shutil.copy(os.path.join(j), val_masks_dir)


def visualize(n, image, mask, pr_mask=None, path_save=None, metric_dict=None):
    """Plot list of images."""
    if image.shape[0] != n:
        image = np.expand_dims(image, axis=0)
    image = image.transpose(0, 2, 3, 1)
    mask = mask.squeeze(1)
    
    if pr_mask != None:
        figure, ax = plt.subplots(nrows=n, ncols=3, figsize=(12,7))
    else:
        figure, ax = plt.subplots(nrows=n, ncols=2, figsize=(12,7))
    
    for i in range(n):
        if image.ndim == 4:
            ax[i, 0].imshow(image[i, :, :, :])
        else:
            ax[i, 0].imshow(image[i, :, :], cmap='gray')
        ax[0, 0].title.set_text('Test image')
        ax[i, 0].axis('off')
        ax[i, 1].imshow(mask[i, :, :], cmap='jet')
        ax[0, 1].title.set_text('Test mask')
        ax[i, 1].axis('off')
        if pr_mask != None:
            ax[i, 2].imshow(pr_mask[i, :, :], cmap='jet')
            ax[0, 2].title.set_text(f'Prediction \n{metric_dict[0]}')
            ax[i, 2].title.set_text(f'{metric_dict[i]}')
            ax[i, 2].axis('off')
    # plt.show()
    plt.savefig(path_save + str(np.random.randint(0, 100)) + ".png")



def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_filenames(path, ext):
    X0 = []
    for i in sorted(os.listdir(path)):
        if i.endswith(ext):
            X0.append(os.path.join(path, i))
    return X0

