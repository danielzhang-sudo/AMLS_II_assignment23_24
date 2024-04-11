from train import train#
from test import test
import argparse

def main(args):
    t = args.test
    if t:
        test(args, None, None)
    else:
        train(args)

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser = parser = argparse.ArgumentParser()

    parser.add_argument('--test', default=False, action='store_true', help='add argument if testing')
    parser.add_argument('--test_lr_path', type=str, default='', help='path to lr test images')
    parser.add_argument('--test_hr_path', type=str, default='', help='path to hr test images')
    parser.add_argument('--pre_epochs', type=int, default=5, help='# of pre-training epochs')
    parser.add_argument('--fine_epochs', type=int, default=5, help='# of fine-tuning epochs')
    parser.add_argument('--finetune', default=False, action='store_true', help='add argument if fine-tuning')
    parser.add_argument('--path', type=str, default='', help='path to weights if resuming training')
    parser.add_argument('--augmentations', type=int, default=3, help='how many augmentations to the dataset')

    args = parser.parse_args()

    main(args=args)
