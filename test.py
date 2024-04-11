
import torch
import  torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import numpy as np
from data import Data
from model import Generator
from wgan import Critic
from perceptual import PSNR
from torchmetrics.functional.image import structural_similarity_index_measure as structural_similarityy


def test(args, gen, test_loader):
    f = open('test_output.txt', 'w')
    test = args.test
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    batch = 11
    if test:
        dataset = Data(lr_path = args.test_lr_path, hr_path = args.test_hr_path)
        test_loader = DataLoader(dataset, batch_size=batch, num_workers=0)
        checkpoint = torch.load(args.path)
        gen = Generator(in_channels=3).to(device)
        gen.load_state_dict(checkpoint['gen_state_dict'])
    else:
        test_loader = test_loader
        gen = gen

    gen.eval()

    mse = nn.MSELoss()
    peak_signal_noise_ratio = PSNR()

    mse_score = []
    psnr_score = []
    ssim_score = []

    mse_sco = 0
    peak_sco = 0
    ssim_sco = 0

    for i, data in enumerate(test_loader):
        lr = data['lr'].to(device)
        hr = data['hr'].to(device)

        fake = gen(lr)
        """
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 6))
        ax1.imshow((np.round(((fake[0].cpu().detach().permute(1, 2, 0).numpy() + 1) * 127.5), 0)) / 255)
        ax1.set_title('SR')
        # plt.imshow((np.round(((fake[0].cpu().detach().permute(1, 2, 0).numpy() + 1) * 127.5), 0)) / 255)
        plt.figure()
        ax2.imshow((np.round(((hr[0].cpu().detach().permute(1, 2, 0).numpy() + 1) * 127.5), 0)) / 255)
        ax2.set_title('GT')
        # plt.imshow((np.round(((hr[0].cpu().detach().permute(1, 2, 0).numpy() + 1) * 127.5), 0)) / 255)
        plt.savefig('figures/in_enumerate.png')
        """
        p, m = peak_signal_noise_ratio((fake*255), ((hr+1)*127.5))

        peak_sco += p * lr.size(0)
        mse_sco += m * lr.size(0)
        ssim_sco += structural_similarityy((fake*255), ((hr+1)*127.5)) * lr.size(0)

    mse_score.append((mse_sco / len(test_loader)).item())
    psnr_score.append((peak_sco / len(test_loader)).item())
    ssim_score.append((ssim_sco / len(test_loader)).item())

    qwerty = range(len(mse_score))
    #print(len(mse_score))
    print(mse_score)
    print(psnr_score)
    print(ssim_score)

    f.write('mse_score: '+str(mse_score)+'\n')
    f.write('psnr_score: '+str(psnr_score)+'\n')
    f.write('ssim_score: '+str(ssim_score)+'\n')
    """
    plt.figure()
    plt.plot(qwerty, mse_score, 'bo', label='mse_score')
    plt.title('test mse scores')
    plt.legend()
    plt.savefig('figures/test_mse_score.png')

    plt.figure()
    plt.plot(qwerty, psnr_score, 'bo', label='psnr_score')
    plt.title('test psnr scores')
    plt.legend()
    plt.savefig('figures/test_psnr_score.png')

    plt.figure()
    plt.plot(qwerty, ssim_score, 'bo', label='ssim_score')
    plt.title('test ssim scores')
    plt.legend()
    plt.savefig('figures/test_ssim_score.png')
    """
    #print(((lr)*255).type(torch.int64))
    #print(((lr)*255).type(torch.int64).shape)

    f.close()

    for i in range(batch):

        fig, (ax1, ax2) = plt.subplots(1,2)
        #ax1.imshow(((hr.cpu()[i].permute(1,2,0))+1)*127.5)
        # ax1.imshow(((hr+1)*127.5)[i].type(torch.int64).cpu().permute(1,2,0))
        ax1.imshow((np.round(((fake[i].cpu().detach().permute(1, 2, 0).numpy() + 1) * 127.5), 0)) / 255)
        ax1.set_title('lr')
        # ax2.imshow(((fake.detach().cpu()[i].permute(1,2,0))*255))
        # ax2.imshow((fake)[i].type(torch.int64).detach().cpu().permute(1,2,0))
        ax2.imshow((np.round(((hr[i].cpu().detach().permute(1, 2, 0).numpy() + 1) * 127.5), 0)) / 255)
        ax2.set_title('hr')

        fig.savefig(f'figures/hr_lr_{i}.png', bbox_inches='tight')
    
if __name__=='__main__':

    parser = parser = argparse.ArgumentParser()

    parser.add_argument('--test', default=True, action='store_true', help='add argument if testing')
    parser.add_argument('--test_lr_path', type=str, default='sr_dataset/lr1', help='path to lr test images')
    parser.add_argument('--test_hr_path', type=str, default='sr_dataset/hr1', help='path to hr test images')
    parser.add_argument('--pre_epochs', type=int, default=5, help='# of pre-training epochs')
    parser.add_argument('--fine_epochs', type=int, default=5, help='# of fine-tuning epochs')
    parser.add_argument('--finetune', default=False, action='store_true', help='add argument if fine-tuning')
    parser.add_argument('--path', type=str, default='', help='path to weights if resuming training')
    parser.add_argument('--augmentations', type=int, default=3, help='how many augmentations to the dataset')

    args= parser.parse_args()

    test(args, None, None)
