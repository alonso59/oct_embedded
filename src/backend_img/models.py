import torch
import torch.nn as nn
from summary import summary
from networks.unet import UNet
from networks.swin_unet import SwinUnet

class SegmentationModels(nn.Module):
    def __init__(self, device, in_channels, img_sizeh, img_sizew, config_file, n_classes=1, pretrain=True, pretrained_path=None) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_channels
        self.img_sizeh = img_sizeh
        self.img_sizew = img_sizew
        self.n_classes = n_classes
        self.pretrain = pretrain
        self.config_file = config_file
        self.pretrained_path = pretrained_path

    def model_building(self, name_model='unet'):
        if name_model == 'unet':
            feature_start = self.config_file['unet_architecutre']['feature_start']
            layers = self.config_file['unet_architecutre']['layers']
            bilinear = self.config_file['unet_architecutre']['bilinear']
            dropout = self.config_file['unet_architecutre']['dropout']
            kernel_size = self.config_file['unet_architecutre']['kernel_size']
            stride = self.config_file['unet_architecutre']['stride']
            padding = self.config_file['unet_architecutre']['padding']
            self.model, name = self.UNet(feature_start, layers, bilinear, dropout, kernel_size, stride, padding)

        if name_model == 'swin_unet':
            self.model, name = self.swin_unet()

        if name_model == 'swin_unet_custom':
            embed_dim = self.config_file['swin_unet_custom_architecture']['embed_dim']
            depths = self.config_file['swin_unet_custom_architecture']['depths']
            num_heads = self.config_file['swin_unet_custom_architecture']['num_heads']
            window_size = self.config_file['swin_unet_custom_architecture']['window_size']
            drop_path_rate = self.config_file['swin_unet_custom_architecture']['drop_path_rate']
            self.model, name = self.SwinUnet_Custom(embed_dim, depths, num_heads, window_size, drop_path_rate)

        return self.model, name

    def summary(self, logger=None):
        summary(self.model, input_size=(self.in_channels, self.img_sizeh, self.img_sizew), batch_size=-1, logger=logger)

    def UNet(self, feature_start=16, layers=4, bilinear=False, dropout=0.0, kernel_size=3, stride=1, padding=1):
        model = UNet(
            num_classes=self.n_classes,
            input_channels=self.in_channels,
            num_layers=layers,
            features_start=feature_start,
            bilinear=bilinear,
            dp=dropout,
            kernel_size=(kernel_size, kernel_size),
            padding=padding,
            stride=stride
        ).to(self.device)
        if self.pretrain:
            pass
        return model, model.__name__

    def swin_unet(self,
                  embed_dim=96,
                  depths=[2, 2, 6, 2],
                  num_heads=[3, 6, 12, 24],
                  window_size=8,
                  drop_path_rate=0.1,
                  ):

        model = SwinUnet(
            img_size=self.img_sizeh,
            num_classes=self.n_classes,
            zero_head=False,
            patch_size=4,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            drop_path_rate=drop_path_rate,
        ).to(self.device)

        if self.pretrain:
            model.state_dict()
            model.load_from("pretrained/swin_tiny_patch4_window7_224.pth", self.device)
        return model, model.__name__

    def SwinUnet_Custom(self,
                        embed_dim=24,
                        depths=[2, 2, 2, 2],
                        num_heads=[2, 2, 2, 2],
                        window_size=7,
                        drop_path_rate=0.1,
                        ):

        model = SwinUnet(
            img_size=self.img_sizeh,
            num_classes=self.n_classes,
            zero_head=False,
            patch_size=4,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            drop_path_rate=drop_path_rate,
        ).to(self.device)

        return model, model.__name__

    """
    you can add your own network here
    .
    .
    .
    """
