import torch
import torch.nn as nn
from model import FirstConv
from torch.autograd import grad

class Critic(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        seq = []
        seq.append(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1))
        seq.append(nn.LeakyReLU(inplace=True))

        seq.append(FirstConv(in_channels=64, out_channels=64, kernel=3, stride=2, padding=1, discriminator=True))
        seq.append(FirstConv(in_channels=64, out_channels=128, kernel=3, stride=1, padding=1, discriminator=True))
        seq.append(FirstConv(in_channels=128, out_channels=128, kernel=3, stride=2, padding=1, discriminator=True))

        #seq.append(FirstConv(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, discriminator=True))
        #seq.append(FirstConv(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, discriminator=True))
        #seq.append(FirstConv(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, discriminator=True))
        #seq.append(FirstConv(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, discriminator=True))

        self.seq = nn.Sequential(*seq)

        self.pool = nn.AdaptiveAvgPool2d((6, 6))
        self.flat = nn.Flatten()
        self.linear1 = nn.Linear(128*6*6, 256)
        self.act = nn.LeakyReLU(inplace=True)
        self.linear2 = nn.Linear(256, 1)
        gan = []
        #gan.append(nn.Linear(512, 1024))
        gan.append(nn.Linear(128, 256))
        gan.append(nn.LeakyReLU(inplace=True))
        #gan.append(nn.Linear(1024, 1))
        gan.append(nn.Linear(256, 1))
        gan.append(nn.Sigmoid()) # TODO delete or not?

        self.gan = nn.Sequential(*gan)

    def forward(self, x):
        i = self.seq(x)
        # i = self.gan(i)
        # print(i.shape)
        i = self.pool(i)
        i = self.flat(i)
        i = self.linear1(i)
        i = self.act(i)
        i = self.linear2(i)
        return i

def gradient_penalty(critic, gen, gt, device):
    batch_size, c, h, w = gt.shape

    # Calculate interpolation
    eps = torch.rand((batch_size, 1, 1, 1)).repeat(1, c, h, w).to(device)

    interpolated = eps * gt + (1 - eps) * gen
    interpolated = interpolated.requires_grad_(True)

    # Calculate probability of interpolated examples
    prob_interpolated = critic(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = grad(inputs = interpolated,
                     outputs = prob_interpolated,
                     grad_outputs = torch.ones_like(interpolated)
                )[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(gradients.shape[0], -1)

    #self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data[0])

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    # gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    gradients_norm = gradients.norm(2, dim=1)
    gradients_penalty = torch.mean((gradients_norm - 1) ** 2)

    # Return gradient penalty
    #return self.gp_weight * ((gradients_norm - 1) ** 2).mean()
    return gradients_penalty
