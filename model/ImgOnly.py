from torch import nn
from torchvision.models import convnext


class ImgOnly(nn.Module):
    def __init__(self, args):
        super(ImgOnly, self).__init__()
        self.encoder = convnext.convnext_base(weights=convnext.ConvNeXt_Base_Weights.DEFAULT)
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.encoder(x)
        return x
