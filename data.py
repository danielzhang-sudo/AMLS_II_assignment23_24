# Working code for augmentation and crop
# Working code for loading data
from torchvision import transforms
import random
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
import matplotlib.pyplot as plt

patch_size = 24
scale = 4

def crop(lr, hr):
    """
    hr_resize = transforms.Resize(96)
    lr_resize = transforms.Resize(24)

    lr_crop = lr_resize(lr)
    hr_crop = hr_resize(hr)

    """
    # print(lr.shape)
    h, w = lr.shape[:2]

    x = random.randrange(0, w - patch_size + 1)
    y = random.randrange(0, h - patch_size + 1)

    #x = 1
    #y = 1

    x2 = x * scale
    y2 = y * scale

    lr_crop = lr[y:y+patch_size, x:x+patch_size]
    hr_crop = hr[y*scale:(y*scale)+(patch_size*scale), x*scale:(x*scale)+(patch_size*scale)]
    # print(lr_crop.shape)

    return lr_crop, hr_crop

def augment(lr, hr):
    h = random.randrange(0,2)
    v = random.randrange(0,2)
    r = random.randrange(0,5)

    if h:
        lr = np.fliplr(lr)
        hr = np.fliplr(hr)

    if v:
        lr = np.flipud(lr)
        hr = np.flipud(hr)

    if r:
        for i in range(r):
            lr = lr.transpose(1,0,2)
            hr = hr.transpose(1,0,2)

    return lr, hr

def transf(lr, hr, augmentations=3):
    lr1, hr1 = crop(lr, hr)
    lr2, hr2 = augment(lr1, hr1)
    return lr2, hr2

class Data(Dataset):
    def __init__(self, lr_path, hr_path):
        super(Data, self).__init__()

        self.lr_path = lr_path
        self.hr_path = hr_path

        self.lr_img = sorted(os.listdir(lr_path))
        self.hr_img = sorted(os.listdir(hr_path))

        #self.lr_img = [np.array(Image.open(os.path.join(self.lr_path, lr)).convert("RGB")).astype(np.uint8) for lr in self.LR_img]
        #self.hr_img = [np.array(Image.open(os.path.join(self.hr_path, gt)).convert("RGB")).astype(np.uint8) for gt in self.GT_img]


    def __len__(self):
        return len(self.lr_img)

    def __getitem__(self, index):
        #lr = self.lr_img[index].astype(np.float32)
        #hr = self.hr_img[index].astype(np.float32)

        lr = np.array(Image.open(os.path.join(self.lr_path, self.lr_img[index])).convert('RGB'))
        hr = np.array(Image.open(os.path.join(self.hr_path, self.hr_img[index])).convert('RGB'))

        #lr_img = lowres_transform(image=lr)['image']
        #hr_img = highres_transform(image=hr)['image']

        img = {}

        lr = (lr / 255) # normalize [0,1]
        hr = (hr / 127.5) - 1 #normalize [-1,1]

        img['lr'], img['hr'] = lr, hr

        # augments data, if in this class, augmentation is done one by one,
        # not merged with original dataset

        img['lr'], img['hr'] = transf(lr, hr)

        img['lr'] = img['lr'].transpose(2, 0, 1).astype(np.float32)
        img['hr'] = img['hr'].transpose(2, 0, 1).astype(np.float32)

        return img


def test():
    augmentations = 1
    aug_dataset = []

    # augment dataset n times
    for i in range(augmentations):
        aug_data = Data(lr_path="sr_dataset/lr1", hr_path='sr_dataset/hr1')
        aug_dataset.append(aug_data)

    full_dataset = ConcatDataset(aug_dataset)

    loader = DataLoader(full_dataset, batch_size=1, num_workers=0)

    print(len(full_dataset))
    for data in loader:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(data['lr'][0].permute(1, 2, 0).numpy())
        ax2.imshow((np.round(((data['hr'][0].permute(1, 2, 0).numpy() + 1) * 127.5), 0)) / 255)
        plt.show()
        break
