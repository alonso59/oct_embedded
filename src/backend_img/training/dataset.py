import os
import sys
import cv2
import yaml
import torch
import numpy as np
import albumentations as T
import elasticdeform as ed

from tqdm import tqdm
from PIL import Image
from utils import visualize, get_filenames
from torch.utils.data import DataLoader, Dataset
from skimage.restoration import denoise_tv_chambolle
from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform

MEAN = 0.1338  # 0.1338
STD = 0.1466  # 0.1466

class ImagesFromFolder(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, preprocess_input=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.preprocess_input = preprocess_input

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])

        image = np.array(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))  # IMREAD_COLOR
        mask = np.array(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE))

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        if np.ndim(image) == 3:
            image = np.transpose(image, (2, 0, 1))
        else:
            image = np.expand_dims(image, axis=0)

        mask = np.expand_dims(mask, axis=0)

        return image, mask


def loaders(train_imgdir,
            train_maskdir,
            val_imgdir,
            val_maskdir,
            batch_size,
            num_workers=12,
            pin_memory=True,
            preprocess_input=None
            ):

    with open('configs/oct.yaml', "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    train_transforms = T.Compose(
        [
            T.Resize(cfg['general']['img_sizeh'], cfg['general']['img_sizew']),
            T.Rotate(limit=(-25, 25), p=1.0, border_mode=cv2.BORDER_CONSTANT),
            T.HorizontalFlip(p=0.5),
            GrayGammaTransform(limit=(0.8, 1.5), p=0.5),
            T.OneOf([
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, p=0.3),
                T.CLAHE(clip_limit=2, tile_grid_size=(4, 4), p=0.3),
            ], p=0.1),
            T.OneOf([
                # T.Affine(scale=(0.9, 1.1), p=0.5),
                TVDenoising(p=0.2),
                T.GaussianBlur(blur_limit=(1, 3), p=0.2),
                # T.GaussNoise(var_limit=(2,4), mean=0, p=0.3),
            ], p=0.3),
            ElasticDeformation(sigma_range=(1,3), points=4, p=0.1),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )

    val_transforms = T.Compose(
        [
            T.Resize(cfg['general']['img_sizeh'], cfg['general']['img_sizew']),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    train_ds = ImagesFromFolder(image_dir=train_imgdir,
                                mask_dir=train_maskdir,
                                transform=train_transforms,
                                preprocess_input=preprocess_input
                                )
    val_ds = ImagesFromFolder(image_dir=val_imgdir,
                              mask_dir=val_maskdir,
                              transform=val_transforms,
                              preprocess_input=preprocess_input
                              )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )
    return train_loader, val_loader, train_transforms


# class GrayLogTransform(ImageOnlyTransform):
#     def __init__(self, always_apply: bool = False, p: float = 0.5):
#         super().__init__(always_apply, p)

#     def apply(self, img, **params):
#         return gray_log(img)

#     def get_transform_init_args_names(self):
#         return ()


class GrayGammaTransform(ImageOnlyTransform):
    def __init__(self, always_apply: bool = False, p: float = 0.5, limit=(0.8, 3)):
        super().__init__(always_apply, p)
        self.limit = limit

    def apply(self, img, **params):
        return self.gray_gamma(img, self.limit)

    def gray_gamma(self, img, limit):
        gamma = np.random.uniform(limit[0], limit[1])
        gray = img / 255.
        out = np.array(gray ** gamma)
        out = 255 * out
        return out.astype('uint8')

    def get_transform_init_args_names(self):
        return ()


class TVDenoising(ImageOnlyTransform):
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        return self.tv_denoising(img)

    def tv_denoising(self, img):
        w = np.random.uniform(0.00000001, 0.1)
        gray = img / 255.
        out = denoise_tv_chambolle(gray, weight=w)
        out = out * 255
        return out.astype('uint8')

    def get_transform_init_args_names(self):
        return ()


class ElasticDeformation(DualTransform):
    def __init__(self, sigma_range=(2, 4), points=3, always_apply: bool = False, p: float = 0.5 ):
        super().__init__(always_apply, p)
        self.p = p
        self.sigma_range = sigma_range
        self.points = points

    def apply(self, img, **params):
        axis, deform_shape = _normalize_axis_list(None, [img])

        if not isinstance(self.points, (list, tuple)):
            self.points = [self.points] * len(deform_shape)

        sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])

        self.displacement = np.random.randn(len(deform_shape), *self.points) * sigma

        X_deformed = ed.deform_grid(img, self.displacement, mode='nearest', axis=axis, order=1)

        return X_deformed
    
    def apply_to_mask(self, img, **params):
        axis, deform_shape = _normalize_axis_list(None, [img])
        Y_deformed = ed.deform_grid(img, self.displacement, mode='nearest', axis=axis, order=0)
        # Y_deformed = np.round(Y_deformed)
        return Y_deformed

    def _repr_(self) -> str:
        return f"{self._class.name_}()"

def _normalize_axis_list(axis, Xs):
    if axis is None:
        axis = [tuple(range(x.ndim)) for x in Xs]
    elif isinstance(axis, int):
        axis = (axis,)
    if isinstance(axis, tuple):
        axis = [axis] * len(Xs)
    assert len(axis) == len(Xs), 'Number of axis tuples should match number of inputs.'
    input_shapes = []
    for x, ax in zip(Xs, axis):
        assert isinstance(ax, tuple), 'axis should be given as a tuple'
        assert all(isinstance(a, int) for a in ax), 'axis must contain ints'
        assert len(ax) == len(axis[0]), 'All axis tuples should have the same length.'
        assert ax == tuple(set(ax)), 'axis must be sorted and unique'
        assert all(0 <= a < x.ndim for a in ax), 'invalid axis for input'
        input_shapes.append(tuple(x.shape[d] for d in ax))
    assert len(set(input_shapes)) == 1, 'All inputs should have the same shape.'
    deform_shape = input_shapes[0]
    return axis, deform_shape

def gray_log(img):
    gray = img / 255.
    c = np.log10(1 + np.max(gray))
    out = c * np.log(1 + gray)
    return out


def test(cfg):
    paths = cfg['paths']
    base = paths['base']
    train_imgdir = os.path.join(base, paths['train_imgdir'])
    train_mskdir = os.path.join(base, paths['train_mskdir'])
    val_imgdir = os.path.join(base, paths['val_imgdir'])
    val_mskdir = os.path.join(base, paths['val_mskdir'])
    train_transforms = T.Compose(
        [
            ElasticDeformation(sigma_range=(2,4), points=10, p=0.5)
            # T.Rotate(limit=(-30, 30), p=1.0, border_mode=cv2.BORDER_REFLECT101),
            # T.HorizontalFlip(p=0.5),
            # T.VerticalFlip(p=0.15),
            # # T.OneOf([
            # #     GrayGammaTransform(p=0.5),
            # #     T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, p=0.5),
            # #     T.CLAHE(clip_limit=3, tile_grid_size=(4, 4), p=0.5),
            # # ], p=0.5),
            # # T.OneOf([
            # #     T.Affine(scale=(0.9, 1.1), p=0.5),
            # #     TVDenoising(p=0.5),
            # #     T.GaussianBlur(blur_limit=(3, 7), p=0.2),
            # # ], p=0.5),
            # T.ElasticTransform(alpha=1, sigma=25, p=1),
            # T.Normalize(mean=MEAN, std=STD),
        ]
    )
    val_transforms = T.Compose(
        [
            T.Normalize(mean=(0.1338, 0.1338, 0.1338), std=(0.1466, 0.1466, 0.1466))
        ]
    )
    train_ds = ImagesFromFolder(image_dir=train_imgdir,
                                mask_dir=train_mskdir,
                                transform=train_transforms,
                                )
    randint = np.random.randint(low=0, high=len(train_ds))
    imgs = []
    msks = []
    for i in range(4):
        image, mask = train_ds[0]
        imgs.append(image)
        msks.append(mask)
    visualize(len(imgs), np.array(imgs), np.array(msks), pr_mask=None, path_save='', metric_dict=None)


if __name__ == "__main__":
    with open('configs/oct.yaml', "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    try:
        test(cfg)  # test
    except KeyError as e:
        print(e)
