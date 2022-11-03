import torch.nn as nn
from networks.SUNet_detail import SUNet


class SUNet_model(nn.Module):
    def __init__(
            self, num_classes, img_size, patch_size, embed_dim, depths, num_heads, window_size, mlp_ratio, qkv_bias, qk_scale,
            drop_rate, drop_path_rate, ape, patch_norm, use_checkpoint):
        super(SUNet_model, self).__init__()
        self.swin_unet = SUNet(img_size=img_size,
                               patch_size=patch_size,
                               in_chans=3,
                               out_chans=num_classes,
                               embed_dim=embed_dim,
                               depths=depths,
                               num_heads=num_heads,
                               window_size=window_size,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias,
                               qk_scale=qk_scale,
                               drop_rate=drop_rate,
                               drop_path_rate=drop_path_rate,
                               ape=ape,
                               patch_norm=patch_norm,
                               use_checkpoint=use_checkpoint)
    @property
    def __name__(self):
        return "SUNet"

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.swin_unet(x)
        return logits
