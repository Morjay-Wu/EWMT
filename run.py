import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

import data_cia
from experiment import Experiment

import os
import faulthandler


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Set the visible CUDA device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
faulthandler.enable()


# --- Command-line arguments for the experiment ---
parser = argparse.ArgumentParser(description='Acquire parameters for the spatiotemporal fusion model.')
parser.add_argument('--lr', type=float, default = 1e-3, help='the initial learning rate')
parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=0, help='number of GPUs to use')
parser.add_argument('--num_workers', type=int, default=12, help='number of threads to load data')
parser.add_argument('--save_dir', type=Path, default=Path('./results/cia'), help='the output directory')

# --- Parameters for data preprocessing ---
parser.add_argument('--train_dir', type=Path, default=('./data/cia/train'), help='the training data directory')
parser.add_argument('--val_dir', type=Path, default=('./data/cia/val'), help='the validation data directory')
parser.add_argument('--test_dir', type=Path, default=('./data/cia/test'), help='the test data directory')
parser.add_argument('--image_size', type=int, nargs='+', default=[1280,1792], help='the size of the coarse image (width, height)')
parser.add_argument('--patch_size', type=int, nargs='+', default=128, help='the coarse image patch size for training')
parser.add_argument('--patch_stride', type=int, nargs='+', default=128, help='the coarse patch stride for image division')
opt = parser.parse_args()

torch.manual_seed(2019)
if not torch.cuda.is_available():
    opt.cuda = False
if opt.cuda:
    torch.cuda.manual_seed_all(2019)
    cudnn.benchmark = True
    cudnn.deterministic = True

torch.autograd.set_detect_anomaly(True)

torch.cuda.empty_cache()

if __name__ == '__main__':
    experiment = Experiment(opt)
    if opt.epochs > 0:
        experiment.train(opt.train_dir, opt.val_dir,
                         opt.patch_size, opt.patch_stride, opt.batch_size,
                         num_workers=opt.num_workers, epochs=opt.epochs)
    experiment.test(opt.test_dir, num_workers=opt.num_workers)

