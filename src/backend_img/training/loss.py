import torch
import torch.nn.functional as F
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self, device, weights=None):
        super(CrossEntropyLoss, self).__init__()
        self.device = device
        self.class_weights = torch.tensor(weights, device=device)

        self.CE = nn.CrossEntropyLoss(weight=self.class_weights)

    @property
    def __name__(self):
        return "cross_entropy"

    def forward(self, inputs, targets):
        target1 = targets.squeeze(1).long().to(self.device)
        cross_entropy = self.CE(inputs.to(self.device), target1)
        return cross_entropy

class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self,  device):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.device = device
        self.BCE = nn.BCEWithLogitsLoss()

    @property
    def __name__(self):
        return "binary_cross_entropy"

    def forward(self, inputs, targets):
        # target1 = targets.squeeze(1).long().to(self.device)
        binary = self.BCE(inputs, targets.float())
        return binary


class DiceLoss(nn.Module):
    def __init__(self, device, activation='softmax'):
        super(DiceLoss, self).__init__()
        self.device = device
        self.activation = activation
    @property
    def __name__(self):
        return "dice_loss"

    def forward(self, inputs, targets):
        dice_loss = dice_score(self, inputs=inputs, targets=targets, activation=self.activation)
        return dice_loss

class WeightedCrossEntropyDice(nn.Module):
    def __init__(self, class_weights, device, lambda_, activation='softmax'):
        super(WeightedCrossEntropyDice, self).__init__()
        self.device = device
        self.class_weights = torch.tensor(class_weights).float().to(device)
        self.CE = nn.CrossEntropyLoss(weight=self.class_weights)
        self.activation = activation
        self.lambda_ = lambda_
        
    @property
    def __name__(self):
        return "weigthed_entropy_dice"

    def forward(self, inputs, targets):
        w = torch.ones(inputs.shape).type(inputs.type()).to(self.device)
        for c in range(inputs.shape[1]):
            w[:, c, :, :] = self.class_weights[c]

        dice_loss = dice_score(self, inputs=inputs, targets=targets, activation=self.activation)

        # Compute categorical cross entropy
        target1 = targets.squeeze(1)
        cross = self.CE(inputs, target1)

        return dice_loss * self.lambda_ + cross * (1 - self.lambda_)

class WCEGeneralizedDiceLoss(nn.Module):
    def __init__(self, class_weights, device, activation='softmax'):
        super(WCEGeneralizedDiceLoss, self).__init__()
        self.device = device
        self.class_weights = torch.tensor(class_weights).float().to(device)
        self.CE = nn.CrossEntropyLoss(weight=self.class_weights)
        self.activation = activation
    @property
    def __name__(self):
        return "weighted_entropy_generalized_dice"

    def forward(self, inputs, targets, eps=1e-7):
        num_classes = inputs.shape[1]
        w = torch.ones(inputs.shape).type(inputs.type()).to(self.device)
        for c in range(inputs.shape[1]):
            w[:, c, :, :] = self.class_weights[c]

        # One Hot ground truth
        true_1_hot = torch.eye(num_classes)[targets.squeeze(1).long()]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float().to(self.device)
        true_1_hot = true_1_hot.type(inputs.type())

        # Getting probabilities
        probas = F.softmax(inputs, dim=1)

        # Compute DiceLoss
        mult = (probas * true_1_hot).to(self.device)
        sum_w = torch.sum(w, dim=(0, 2, 3))
        dims = (0, 2, 3)
        intersection = 2 * torch.pow(sum_w, 2) * \
            torch.sum(mult, dim=(0, 2, 3)) + eps
        cardinality = torch.pow(
            sum_w, 2) * (torch.sum(probas, dim=dims) + torch.sum(true_1_hot, dim=dims)) + eps
        dice_loss = 1 - (intersection / cardinality).mean()

        # Compute categorical cross entropy
        target1 = targets.squeeze(1).long().to(self.device)
        cross = self.CE(inputs.to(self.device), target1)

        return dice_loss * 0.6 + cross * 0.4

def dice_score(self, inputs, targets, activation='softmax'):
    num_classes = inputs.shape[1]
    eps = 1e-7
    # One Hot ground truth
    true_1_hot = torch.eye(num_classes)[targets.squeeze(1).long()]
    true_1_hot = true_1_hot.permute(0, 3, 1, 2).float().to(self.device)
    true_1_hot = true_1_hot.type(inputs.type())
    # Getting probabilities
    if activation == 'softmax':
        probas = F.softmax(inputs, dim=1)
    elif activation == 'sigmoid':
        probas = F.sigmoid(inputs)

    # Compute DiceLoss
    mult = (probas * true_1_hot).to(self.device)
    dims = (0, 2, 3)
    intersection = 2 * torch.sum(mult, dim=(0, 2, 3)) + eps
    cardinality = torch.sum(probas, dim=dims) + \
        torch.sum(true_1_hot, dim=dims) + eps
    dice_score = 1 - (intersection / cardinality).mean()
    return dice_score

""" Loss from: https://github.com/azadef/ynet """

class FocalFrequencyLoss(nn.Module):
    """The torch.nn.Module class that implements focal frequency loss - a
    frequency domain loss function for optimizing generative models.
    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>
    Args:
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
        patch_factor (int): the factor to crop image patches for patch-based focal frequency loss. Default: 1
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
    """

    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False, batch_matrix=False):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            'Patch factor should be divisible by image height and width')
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

        # stack to patch tensor
        y = torch.stack(patch_list, 1)

        # perform 2D DFT (real-to-complex, orthonormalization)
        freq = torch.fft.fft2(y, norm='ortho')
        freq = torch.stack([freq.real, freq.imag], -1)
        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    def forward(self, pred, target, matrix=None, **kwargs):
        """Forward function to calculate focal frequency loss.
        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        # calculate focal frequency loss
        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight

class DiceLoss1(nn.Module):
    def forward(self, output, target, weights=None, ignore_index=None):
        """
            output : NxCxHxW Variable
            target :  NxHxW LongTensor
            weights : C FloatTensor
            ignore_index : int index to ignore from loss
            """
        eps = 0.0001

        output = output.exp()
        encoded_target = output.detach() * 0
        if ignore_index is not None:
            mask = target == ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)

        if weights is None:
            weights = 1

        intersection = output * encoded_target
        numerator = 2 * intersection.sum(0).sum(1).sum(1)
        denominator = output + encoded_target

        if ignore_index is not None:
            denominator[mask] = 0
        denominator = denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = weights * (1 - (numerator / denominator))

        return loss_per_channel.sum() / output.size(1)


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.CrossEntropyLoss(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(inputs, targets)


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.cross_entropy_loss = CrossEntropyLoss2d()
        self.dice_loss = DiceLoss1()

    def forward(self, input, target, weight=1, device="cuda"):
        target = target.type(torch.LongTensor).to(device)
        input_soft = F.softmax(input,dim=1)
        y2 = torch.mean(self.dice_loss(input_soft, target))
        y1 = torch.mean(torch.mul(self.cross_entropy_loss.forward(input, target), weight))
        y = y1 + y2
        return y