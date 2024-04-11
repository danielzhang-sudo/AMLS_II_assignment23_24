# Train code with pretrain and finetuning

import torch
from torch import optim
from matplotlib import pyplot as plt
import numpy as np
from data import Data
from torch.utils.data import DataLoader, ConcatDataset
from model import Generator
from wgan import Critic
from test import test
import torch.nn as nn
from perceptual import VGGLoss, PSNR
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from torchmetrics.functional.image import structural_similarity_index_measure as structural_similarityy # torchmetrics.image.StructuralSimilarityIndexMeasure as structure_similarityy


def train(args):
    f = open('train_output.txt', 'w')

    pre_epochs = args.pre_epochs # 5
    fine_epochs = args.fine_epochs # 5
    curr_epoch = 0
    finetune = args.finetune # False
    path = args.path
    augmentations = args.augmentations

    n_critic = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    aug_dataset = []

    # augment dataset n times
    for i in range(augmentations):
        aug_data = Data(lr_path='Dataset/lr', hr_path='Dataset/hr')
        aug_dataset.append(aug_data)

    training_size = round(900 * augmentations * 0.7)
    validation_size = round(900 * augmentations * 0.15)
    test_size = round(900 * augmentations * 0.15)

    full_dataset = ConcatDataset(aug_dataset)
    print(len(full_dataset))
    train_set, temp_set = torch.utils.data.random_split(full_dataset, [int(training_size), int(validation_size+test_size)]) # [630, 135, 135]
    val_set, test_set = torch.utils.data.random_split(temp_set, [int(validation_size), int(test_size)])
    
    train_loader = DataLoader(train_set, batch_size=30, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=10, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=10, num_workers=0)
    print('data loaded')


    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    vgg_loss = VGGLoss()
    peak_signal_noise_ratio = PSNR()

    gen_losses = []
    disc_losses = []
    gen_val_losses = []
    disc_val_losses = []
    mse_score = []
    psnr_score = []
    ssim_score = []

    mse_train = []
    psnr_train = []
    ssim_train = []

    # Create models
    gen = Generator(in_channels=3).to(device)
    disc = Critic(in_channels=3).to(device)

    # Optimizers
    opt_gen = optim.Adam(gen.parameters(), lr=1e-4, betas=(0.9, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=1e-4, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(opt_gen, step_size = 30, gamma = 0.1)
    
    # Resume training
    if finetune:
        checkpoint = torch.load(path)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        gen.load_state_dict(checkpoint['gen_state_dict'])
        disc.load_state_dict(checkpoint['disc_state_dict'])
        opt_gen.load_state_dict(checkpoint['gen_optim_state_dict'])
        opt_disc.load_state_dict(checkpoint['disc_optim_state_dict'])
        curr_epoch = checkpoint['epoch']
        gen_losses = checkpoint['gen_loss']
        disc_losses = checkpoint['disc_loss']
        gen_val_losses = checkpoint['gen_val_loss']
        disc_val_losses = checkpoint['disc_val_loss']
        mse_train = checkpoint['mse_score']
        psnr_train = checkpoint['psnr_score']
        ssim_train = checkpoint['ssim_score']
        mse_score = checkpoint['mse_score_val']
        psnr_score = checkpoint['psnr_score_val']
        ssim_score = checkpoint['ssim_score_val']

    # Pretraining loop
    if curr_epoch < pre_epochs:
        # Pre-train generator with L2 loss
        for epoch in range(pre_epochs-curr_epoch):
            print(epoch, end=', ')
            gen.train()
            disc.train()

            gen_running_loss = 0
            mse_sco = 0
            peak_sco = 0
            ssim_sco = 0

            mse_sco_train = 0
            peak_sco_train = 0
            ssim_sco_train = 0

            # Pre-train with MSE loss
            for i, data in enumerate(train_loader):
                hr = data['hr'].to(device)
                lr = data['lr'].to(device)

                fake = gen(lr)
                l2_loss = mse(fake, hr)
                opt_gen.zero_grad()
                l2_loss.backward()
                opt_gen.step()

                gen_running_loss += l2_loss.item() * lr.size(0)
                
                p, m = peak_signal_noise_ratio((fake*255), ((hr+1)*127.5))
                peak_sco_train += p * lr.size(0)
                mse_sco_train += m * lr.size(0)
                ssim_sco_train += structural_similarityy(fake*255, ((hr+1)*127.5)) * lr.size(0)

                if i % 5 == 0:
                    print('pretrain _step', end=', ')

            mse_train.append((mse_sco_train / len(train_loader)).item())
            psnr_train.append((peak_sco_train / len(train_loader)).item())
            ssim_train.append((ssim_sco_train / len(train_loader)).item())

            print('')
            gen_losses.append(gen_running_loss / len(train_loader))
            
            
            
            # Eval generator
            gen.eval()
            disc.eval()

            for i, data in enumerate(val_loader):
                gt = data['hr'].to(device)
                lr = data['lr'].to(device)

                fake = gen(lr)
                critic_fake = disc(fake).reshape(-1)
                l2_loss = mse(fake, gt)
                gen_loss = l2_loss

                p, m = peak_signal_noise_ratio((fake*255), ((gt+1)*127.5))

                peak_sco += p * lr.size(0)
                mse_sco += m * lr.size(0)
                ssim_sco += structural_similarityy((fake*255), ((gt+1)*127.5)) * lr.size(0)

                gen_running_loss += gen_loss.item() * lr.size(0)

            gen_val_losses.append(gen_running_loss / len(val_loader))
            mse_score.append((mse_sco / len(val_loader)).item())
            psnr_score.append((peak_sco / len(val_loader)).item())
            ssim_score.append((ssim_sco / len(val_loader)).item())

            # Save weights
            if epoch % 1 == 0:
                torch.save({
                    'epoch':epoch+curr_epoch,
                    'scheduler_state_dict':scheduler.state_dict(),
                    'gen_state_dict':gen.state_dict(),
                    'disc_state_dict':disc.state_dict(),
                    'gen_optim_state_dict':opt_gen.state_dict(),
                    'disc_optim_state_dict':opt_disc.state_dict(),
                    'gen_loss':gen_losses,
                    'disc_loss':disc_losses,
                    'gen_val_loss':gen_val_losses,
                    'disc_val_loss':disc_val_losses,
                    'mse_score': mse_train,
                    'psnr_score': psnr_train,
                    'ssim_score': ssim_train,
                    'mse_score_val': mse_score,
                    'psnr_score_val': psnr_score,
                    'ssim_score_val': ssim_score

                }, f'checkpoints/{epoch+curr_epoch}_model.pt')

            # Write losses to file
            f.write('epoch: '+str(epoch)+'\n')
            f.write('gen_loss: '+str(gen_losses)+'\n')
            f.write('gen_val_loss: '+str(gen_val_losses)+'\n')
            f.write('disc_loss: '+str(disc_losses)+'\n')
            f.write('disc_val_loss: '+str(disc_val_losses)+'\n')
            f.write('mse_score: '+str(mse_score)+'\n')
            f.write('psnr_score: '+str(psnr_score)+'\n')
            f.write('ssim_score: '+str(ssim_score)+'\n')
            f.write('mse_train: '+str(mse_train)+'\n')
            f.write('psnr_train: '+str(psnr_train)+'\n')
            f.write('ssim_train: '+str(ssim_train)+'\n')
            """
            print('epoch: '+str(epoch+curr_epoch))
            print('gen_loss: '+str(gen_losses))
            print('disc_loss: '+str(disc_losses))
            print('gen_val_loss: '+str(gen_val_losses))
            print('disc_val_loss: '+str(disc_val_losses))
            print('mse_score: '+str(mse_score))
            print('psnr_score: '+str(psnr_score))
            print('ssim_score: '+str(ssim_score))
            print('mse_train: '+str(mse_train))
            print('psrn_train: '+str(psnr_train))
            print('ssim_train: '+str(ssim_train))
            print('epoch: '+str(epoch+curr_epoch))
            """
        curr_epoch = pre_epochs
        print('')

    # Finetuning loop
    if curr_epoch >= pre_epochs and curr_epoch < fine_epochs+pre_epochs:

        # Fine-tune with perceptual and adversarial loss
        for epoch in range(fine_epochs+pre_epochs-curr_epoch):
            gen.train()
            disc.train()

            gen_running_loss = 0
            disc_running_loss = 0
            mse_sco = 0
            peak_sco = 0
            ssim_sco = 0

            peak_sco_train = 0
            mse_sco_train = 0
            ssim_sco_train = 0

            for i, data in enumerate(train_loader):
                gt = data['hr'].to(device)
                lr = data['lr'].to(device)

                critic_running_loss = 0
                
                # Train discriminator
                for _ in range(n_critic):
                    fake = gen(lr)
                    critic_real = disc(gt).reshape(-1)
                    critic_fake = disc(fake.detach()).reshape(-1)

                    critic_loss_real = bce(
                        critic_real, torch.ones_like(critic_real) - 0.1*torch.rand_like(critic_real)
                    )
                    critic_loss_fake = bce(critic_fake, torch.zeros_like(critic_fake))

                    # gp = gradient_penalty(disc, fake, gt, device)
                    disc_loss = critic_loss_fake + critic_loss_real # + 10*gp
                    # disc_loss = (-(torch.mean(critic_real) - torch.mean(critic_fake)))

                    opt_gen.zero_grad()
                    opt_disc.zero_grad()
                    disc_loss.backward(retain_graph=True)
                    opt_disc.step()
                    
                    # Weight clipping
                    for p in disc.parameters():
                        p.data.clamp_(-0.01, 0.01)

                    critic_running_loss += disc_loss.item()

                # Train generator
                fake = gen(lr)
                critic_fake = disc(fake).reshape(-1)
                l2_loss = mse(fake, gt)
                gan_loss = 1e-3 * bce(critic_fake, torch.ones_like(critic_fake))
                perceptual_loss = 0.006 * vgg_loss(fake, gt)
                gen_loss = perceptual_loss + gan_loss + l2_loss

                opt_gen.zero_grad()
                opt_disc.zero_grad()
                gen_loss.backward()
                opt_gen.step()

                gen_running_loss += gen_loss.item() * lr.size(0)
                disc_running_loss += (critic_running_loss / n_critic) * lr.size(0)
                
                p, m = peak_signal_noise_ratio((fake*255), ((gt+1)*127.5))
                peak_sco_train += p * lr.size(0)
                mse_sco_train += m * lr.size(0)
                ssim_sco_train += structural_similarityy((fake*255), ((gt+1)*127.5)) * lr.size(0)

                if i % 5 == 0:
                    print('finetune_step', end=', ')

            print('')

            mse_train.append((mse_sco_train / len(train_loader)).item())
            psnr_train.append((peak_sco_train / len(train_loader)).item())
            ssim_train.append((ssim_sco_train / len(train_loader)).item())

            gen_losses.append(gen_running_loss / len(train_loader))
            disc_losses.append(disc_running_loss / len(train_loader))

            
            
            # Evaluate
            gen.eval()
            disc.eval()

            for i, data in enumerate(val_loader):
                gt = data['hr'].to(device)
                lr = data['lr'].to(device)

                # Eval discriminator
                fake = gen(lr)
                critic_running_loss = 0

                critic_real = disc(gt)
                critic_fake = disc(fake.detach())

                critic_loss_real = bce(
                    critic_real, torch.ones_like(critic_real) - 0.1*torch.rand_like(critic_real)
                )
                critic_loss_fake = bce(critic_fake, torch.zeros_like(critic_fake))
                #gp = gradient_penalty(disc, fake, gt, device)
                disc_loss = critic_loss_fake + critic_loss_real #+ 10*gp

                # Eval generator
                critic_fake = disc(fake).reshape(-1)
                l2_loss = mse(fake, gt)
                gan_loss = 1e-3 * bce(critic_fake, torch.ones_like(critic_fake))
                perceptual_loss = 0.006 * vgg_loss(fake, gt)
                gen_loss = perceptual_loss + gan_loss

                gen_running_loss += gen_loss.item() * lr.size(0)
                disc_running_loss += disc_loss.item() * lr.size(0)

                p, m = peak_signal_noise_ratio((fake*255), ((gt+1)*127.5))

                peak_sco += p * lr.size(0)
                mse_sco += m * lr.size(0)
                ssim_sco += structural_similarityy((fake*255), ((gt+1)*127.5)) * lr.size(0)
            
            gen_val_losses.append(gen_running_loss / len(val_loader))
            disc_val_losses.append(disc_running_loss / len(val_loader))
            mse_score.append((mse_sco / len(val_loader)).item())
            psnr_score.append((peak_sco / len(val_loader)).item())
            ssim_score.append((ssim_sco / len(val_loader)).item())

            # Save weights
            if epoch % 1 == 0:
                torch.save({
                    'epoch':epoch+curr_epoch,
                    'scheduler_state_dict':scheduler.state_dict(),
                    'gen_state_dict':gen.state_dict(),
                    'disc_state_dict':disc.state_dict(),
                    'gen_optim_state_dict':opt_gen.state_dict(),
                    'disc_optim_state_dict':opt_disc.state_dict(),
                    'gen_loss':gen_losses,
                    'disc_loss':disc_losses,
                    'gen_val_loss':gen_val_losses,
                    'disc_val_loss':disc_val_losses,
                    'mse_score': mse_train,
                    'psnr_score': psnr_train,
                    'ssim_score': ssim_train,
                    'mse_score_val': mse_score,
                    'psnr_score_val': psnr_score,
                    'ssim_score_val': ssim_score
                }, f'checkpoints/{epoch+curr_epoch}_model.pt')

            # Print losses to file
            f.write('epoch: '+str(epoch+curr_epoch)+'\n')
            f.write('gen_loss: '+str(gen_losses)+'\n')
            f.write('gen_val_loss: '+str(gen_val_losses)+'\n')
            f.write('disc_loss: '+str(disc_losses)+'\n')
            f.write('disc_val_loss: '+str(disc_val_losses)+'\n')
            f.write('mse_score: '+str(mse_score)+'\n')
            f.write('psnr_score: '+str(psnr_score)+'\n')
            f.write('ssim_score: '+str(ssim_score)+'\n')
            f.write('mse_train: '+str(mse_train)+'\n')
            f.write('psnr_train: '+str(psnr_train)+'\n')
            f.write('ssim_train: '+str(ssim_train)+'\n')
            """
            print('epoch: '+str(epoch+curr_epoch))
            print('gen_loss: '+str(gen_losses))
            print('disc_loss: '+str(disc_losses))
            print('gen_val_loss: '+str(gen_val_losses))
            print('disc_val_loss: '+str(disc_val_losses))
            print('mse_score: '+str(mse_score))
            print('psnr_score: '+str(psnr_score))
            print('ssim_score: '+str(ssim_score))
            print('mse_train: '+str(mse_train))
            print('psrn_train: '+str(psnr_train))
            print('ssim_train: '+str(ssim_train))
            print('epoch: '+str(epoch+curr_epoch))
            """
            scheduler.step()
    f.close()
    # Test model
    test(args, gen, test_loader)
