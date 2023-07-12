from torch import nn
from torchvision.models.convnext import convnext_base, ConvNeXt_Base_Weights


class ImgOnly(nn.Module):
    def __init__(self, args):
        super(ImgOnly, self).__init__()
        self.encoder = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.encoder(x)
        return x