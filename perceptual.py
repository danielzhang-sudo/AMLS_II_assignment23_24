import torch
import torch.nn as nn
from torchvision.models import vgg19

# conv layer before maxpooling but after activation

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg19(pretrained=True).features[:36].eval().to('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)
        return self.loss(vgg_input_features, vgg_target_features)

class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):

        mse = torch.mean((img1 - img2) ** 2)
        psnr = 20 * torch.log10(255.0 / torch.sqrt(mse))
        return psnr, mse
