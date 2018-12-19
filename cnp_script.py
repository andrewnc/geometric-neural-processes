import matplotlib.pyplot as plt
import math

import numpy as np

import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils

import os
import h5py
import pathlib

# code for audio manipulation
import spect

               
import sys
sys.path.append("../fastMRI/")

# facebook provided code to subsample/transform
from common import subsample
from data import transforms
from data.mri_data import SliceData


m,n = 320,320 #28, 28
batch_size = 16

class DataTransform:
    def __init__(self, mask_func, resolution, which_challenge, use_seed=True):
        if which_challenge not in ('singlecoil', 'multicoil'):
            raise ValueError('Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.resolution = resolution
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(self, kspace, target, attrs, fname, slice):

        kspace = np.array(kspace)
        target = np.array(target)
        kspace = transforms.to_tensor(kspace)
        target = transforms.to_tensor(target)
        
        kspace = transforms.complex_center_crop(kspace, (self.resolution, self.resolution))
        target = transforms.center_crop(target, (self.resolution, self.resolution))
        return kspace, target
    

def create_datasets(args):
    train_mask = subsample.MaskFunc(args.center_fractions, args.accelerations)
    dev_mask = subsample.MaskFunc(args.center_fractions, args.accelerations)

    train_data = SliceData(
        root=args.data_path + '{}_train'.format(args.challenge),
        transform=DataTransform(train_mask, args.resolution, args.challenge),
        sample_rate=args.sample_rate,
        challenge=args.challenge
    )
    dev_data = SliceData(
        root=args.data_path + '{}_val'.format(args.challenge),
        transform=DataTransform(dev_mask, args.resolution, args.challenge, use_seed=True),
        sample_rate=args.sample_rate,
        challenge=args.challenge,
    )
    return dev_data, train_data


def create_data_loaders(args):
    dev_data, train_data = create_datasets(args)
    display_data = [dev_data[i] for i in range(0, len(dev_data), len(dev_data) // 16)]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate,
    )
    display_loader = DataLoader(
        dataset=display_data,
        batch_size=16,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate,
    )
    return train_loader, dev_loader, display_loader

def get_log_p(data, mu, sigma):
    return -torch.log(torch.sqrt(2*math.pi*sigma**2)) - (data - mu)**2/(2*sigma**2)

def slice_and_dice(kspace):
    a = np.abs(np.fft.ifft2(kspace))
    b = np.vstack((a[len(a)//2:], a[:len(a)//2]))
    return np.hstack((b[:,b.shape[1]//2:], b[:,:b.shape[1]//2]))



class ARGS():
    def __init__(self, challenge, center_fractions, accelerations, resolution, data_path, sample_rate, batch_size):
        self.challenge = challenge
        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.resolution = resolution
        self.data_path = data_path
        self.sample_rate = sample_rate
        self.batch_size = batch_size


class MRIEncoder(nn.Module):
    def __init__(self):
        super(MRIEncoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(2, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        self.fc = nn.Linear(256, 128)
        
    def forward(self, x):
        x = self.layer3(self.layer2(self.layer1(x)))
        x.transpose_(1, -1)
        out = self.fc(x)
        out = torch.mean(out.view((128,out.shape[2]*38*38)), 1).view(1, 128) # reshape and aggregate (using the mean, which works because it is commutative)
        return out
    
class MRIDecoder(nn.Module):
    def __init__(self, m=320, n=320):
        super(MRIDecoder, self).__init__()
        self.m = m
        self.n = n
        self.fc1 = nn.Linear(130, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)
        
    def forward(self, r):
        """r is the aggregated data used to condition"""
        
        # we only take in r, because in this case x is all points in size of image (n, m)
        x = torch.tensor([[i, j] for i in range(0,self.m) for j in range(0,self.n)]).float().to(device)
        x = torch.cat((x, r.view(1,-1).repeat(1,self.m*self.n).view(self.m*self.n,128)), 1)
        
        h = self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))))
        
        mu_real = h[:,0]
        log_sigma_real = h[:,1]
        
        
        # bound the variance
        sigma_real = 0.1 + 0.9 * F.softplus(log_sigma_real)
        
        return mu_real, sigma_real


def collate(batch):
    batch = sorted(batch, key=lambda x: x[0].shape[0])
    max_size = batch[-1][0].shape[0]
    
    data = torch.zeros((len(batch),max_size, m, n, 2))
    target = torch.zeros((len(batch), m, n))
    
    # don't iterate to the last one because it doesn't need padding
    for i in range(0, len(batch) -1):
        data[i] = torch.cat((batch[i][0], torch.zeros((max_size - batch[i][0].shape[0], m, n, 2))))
        target[i] = batch[i][1]
        
    data[-1] = batch[-1][0]
    target[-1] = batch[-1][1]
    return [data, target]


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if use_cuda else "cpu")

    _dir = "/mnt/pccfs/not_backed_up/andrew_open/mri_data/"
    train_dir = _dir + "singlecoil_train/"
    test_dir = _dir + "singlecoil_test/"

            
    # we will want to vary these and see how the method performs
    args = ARGS("singlecoil",[0.08, 0.04],[4, 8], 320, _dir, 1, 16)

    train_loader, val_loader, display_loader = create_data_loaders(args)

    # most of these are not relevant for the mri experiment
    m,n = 320,320 #28, 28
    num_pixels = m*n

    
    test_batch_size = 1000
    epochs = 10

    log_interval = 500


    min_context_points = num_pixels * 0.05 # always have at least 5% of all pixels
    max_context_points = num_pixels * 0.95 # always have at most 95% of all pixels

    encoder = MRIEncoder()
    decoder = MRIDecoder(m, n)

    encoder = nn.DataParallel(encoder)
    decoder = nn.DataParallel(decoder)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))


    for epoch in range(1, epochs+1):
        encoder.train()
        decoder.train()
        progress = tqdm(enumerate(train_loader))
        for batch_idx, (data, target) in progress:
            data = data.transpose(-1, 1).transpose(-1, -2).transpose(-2, -3)
            
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            
            # run the model to get r
            r = encoder(data)
            mu, sigma = decoder(r)
            
            mu = mu.view(n,m)
            sigma = sigma.view(n,m)
            
            log_p = get_log_p(target, mu, sigma)
            
            loss = -log_p.mean()
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                progress.set_description('Loss: {:.6f} Mean: {}/{} Sig: {}/{}'.format(loss.item(), mu.max(), mu.min(), sigma.max(), sigma.min()))
                with open("encoder_mri.pkl", "wb") as of:
                    pickle.dump(encoder, of)

                with open("decoder_mri.pkl", "wb") as of:
                    pickle.dump(decoder, of)

                with open("optim.pkl", "wb") as of:
                    pickle.dump(optimizer, of)

                
        
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            for i, (data, target) in enumerate(val_loader):
                data = data.transpose(-1, 1).transpose(-1, -2).transpose(-2, -3)

                data = data.to(device)
                target = target.to(device)

                r = encoder(data)
                mu, sigma = decoder(r)

                plt.imsave("{}target{}.png".format(epochs, i), target.detach().view(m,n))
                
                data = data.transpose(1,2).transpose(2, 3).transpose(3,4)
                plt.imsave("{}masked_data{}.png".format(epochs, i), slice_and_dice(data[0][-1][:,:,0]))

                plt.imsave("{}mean{}.png".format(epochs, i), mu.detach().view(m,n))

                plt.imsave("{}var{}.png".format(epochs, i), sigma.detach().view(m,n))

                