import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class Accuracy(nn.Module):

    def __init__(self, class_index):
        super().__init__()
        self.class_index = class_index
    @property
    def __name__(self):
        return "accuracy"

    def forward(self, y_pr, y_gt):
        num_classes = y_pr.shape[1]
        true_1_hot = torch.eye(num_classes)[y_gt.squeeze(1).long()]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot = true_1_hot.type(y_pr.type())
        dims = (0,) + tuple(range(2, y_gt.ndimension()))
        # Getting probabilities
        y_pr = F.softmax(y_pr, dim=1).unsqueeze(1)
        y_pr = y_pr.reshape(-1)
        true_1_hot = true_1_hot.reshape(-1)
        tp = torch.sum(true_1_hot == y_pr)
        score = tp / true_1_hot.shape[0]
        return score


class Activation(nn.Module):
    def __init__(self, name, **params):

        super().__init__()

        if name is None or name == "identity":
            self.activation = nn.Identity(**params)
        elif name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif name == "softmax":
            self.activation = nn.Softmax(dim=1)
        elif name == "logsoftmax":
            self.activation = nn.LogSoftmax(**params)
        elif name == "tanh":
            self.activation = nn.Tanh()
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError(
                f"Activation should be callable/sigmoid/softmax/logsoftmax/tanh/"
                f"argmax/argmax2d/clamp/None; got {name}"
            )

    def forward(self, x):
        return self.activation(x)

class MIoU(nn.Module):

    def __init__(self, device, ignore_background=False, activation='softmax', average=False):
        super().__init__()
        self.device = device
        self.ignore_background = ignore_background
        self.activation = Activation(activation)
        self.average = average
        
    @property
    def __name__(self):
        return "MIoU"

    def forward(self, logits, true, eps=1e-5):
        num_classes = logits.shape[1]

        true_1_hot = torch.eye(num_classes, device=self.device)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot = true_1_hot.type(true.type())

        probas = self.activation(logits)
        
        dims = (0, 2, 3)
        mult = (probas * true_1_hot)
        intersection = torch.sum(mult, dim=dims) + eps
        union = torch.sum(true_1_hot, dim=dims) + torch.sum(probas, dim=dims) - intersection + eps
        iou = intersection / union

        iou = iou.detach().cpu().numpy()
        # iou = np.where(iou <= 1e-1, 1, iou)

        if self.ignore_background:
            iou = iou[1:]
        if self.average:
            iou = iou.mean()
        return iou