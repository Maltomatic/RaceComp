import torch
from torch import nn
from toolkit.VGGExtractor import VGG19FeatureExtractor
import torch.nn.functional as F

class PerceptualLossVGG19(nn.Module):
    def __init__(self, layer_weights=None, layers=(4, 8, 12, 16)):
        super().__init__()
        self.feat_net = VGG19FeatureExtractor(layers=layers)
        if layer_weights is None:
            layer_weights = {l: 1.0 for l in layers}
        self.layer_weights = layer_weights
        self.layers = layers
        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, pred, target):
        with torch.amp.autocast(self.device_type, enabled = False): 
            pred_f   = self.feat_net(pred.float())
            target_f = self.feat_net(target.float())
        loss = 0.0
        for l, Fp, Ft in zip(self.layers, pred_f, target_f):
            w = self.layer_weights.get(l, 1.0)
            loss = loss + w * F.l1_loss(Fp, Ft, reduction='mean')
        return loss