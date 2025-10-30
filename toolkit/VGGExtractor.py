import torch
from torch import nn
from torchvision.models import vgg19, VGG19_Weights

class VGG19FeatureExtractor(nn.Module):
    def __init__(self, layers=(4, 8, 12, 16),  # conv2_2, conv3_4, conv4_4, conv5_4 by conv-count index
                 weights=VGG19_Weights.IMAGENET1K_V1):
        super().__init__()
        vgg = vgg19(weights=weights).features
        # freeze
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg.eval()
        self.capture_layers = set(layers)

    @torch.no_grad()
    def _sanity_check(self, x):
        return x

    def forward(self, x):
        feats = []
        conv_idx = 0
        for m in self.vgg:
            if isinstance(m, nn.Conv2d):
                conv_idx += 1
                x = m(x)
                if conv_idx in self.capture_layers:
                    feats.append(x.clone())
            else:
                x = m(x)
        return feats