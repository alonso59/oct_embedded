from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
from torchvision.utils import draw_segmentation_masks as drawer 
import sys
import numpy as np
from torchvision import transforms
from torch.nn import functional as F
import albumentations as T
from .dataset import MEAN, STD
from matplotlib import cm
import matplotlib


class TensorboardWriter(object):

    def __init__(self, name_dir):

        super(TensorboardWriter).__init__()
        self.writer = SummaryWriter(log_dir=name_dir)

    def loss_epoch(self, train_loss, val_loss, step):
        results_loss = {'Train': train_loss, 'Val': val_loss}
        self.writer.add_scalars("Loss", results_loss, step)
    
    def metrics_epoch(self, train_metric, val_metric, step, metric_name):
        results_metric = {'Train'+'/'+metric_name: train_metric, 'Val'+'/'+metric_name: val_metric}
        self.writer.add_scalars(metric_name, results_metric, step)

    def metric_iter(self, metric, step, stage, metric_name):
        self.writer.add_scalar(stage + '/' + metric_name, metric, step)

    def loss_iter(self, loss, step, stage: str):
        self.writer.add_scalar(stage + '/Loss', loss, step)

    def learning_rate(self, lr_, step):
        self.writer.add_scalar("lr", lr_, step)

    def save_graph(self, model, loader):
        self.writer.add_graph(model, loader)

    def save_text(self, tag, text_string):
        self.writer.add_text(tag=tag, text_string=text_string)

    def save_images(self, x, y, y_pred, step, device, tag):
        gt = image_tensorboard(y[:3, :, :], device)
        if y_pred.shape[1] == 1:
            pred = torch.sigmoid(y_pred[:3, :, :, :])
            pred = torch.round(pred)
        else:
            pred = torch.softmax(y_pred[:3, :, :, :], dim=1)
            pred = torch.argmax(pred, dim=1).unsqueeze(1)
        pred = image_tensorboard(pred, device)
        pred = pred.squeeze(1)
        x1 = denormalize_vis(x[:3, :, :, :])
        self.writer.add_images(f'{tag}/Data', x1[:3, :, :, :], step, dataformats='NCHW')
        self.writer.add_images(f'{tag}/True', gt, step, dataformats='NCHW')
        self.writer.add_images(f'{tag}/Prediction', pred, step, dataformats='NCHW')
        # error = torch.abs(gt - pred )
        # self.writer.add_images(f'{tag}/Error', error, step, dataformats='NCHW')

def image_tensorboard(img, device):
    norm = matplotlib.colors.Normalize(vmin=0, vmax=img.max())

    img_rgb = torch.zeros((img.size(0), 3, img.size(2), img.size(3)), dtype=torch.double, device=device)


    for idx in range(1, int(img.max())+1):
        img_rgb[:, 0, :, :] = torch.where(img.squeeze(1) == idx, cm.hsv(norm(idx))[0], img_rgb[:, 0, :, :])
        img_rgb[:, 1, :, :] = torch.where(img.squeeze(1) == idx, cm.hsv(norm(idx))[1], img_rgb[:, 1, :, :])
        img_rgb[:, 2, :, :] = torch.where(img.squeeze(1) == idx, cm.hsv(norm(idx))[2], img_rgb[:, 2, :, :])
    
    return img_rgb

def denormalize_vis(tensor):
    invTrans = transforms.Normalize(mean=-MEAN, std=1/STD)
    return torch.clamp(invTrans(tensor), 0, 1)