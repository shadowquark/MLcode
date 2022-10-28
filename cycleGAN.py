import os
import time
import sys
import math
import functools as ft
import random as rnd
import glob
#import itertools
#from collections.abc import Iterable
from tqdm import tqdm
import h5py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as torF
import torch.optim as optim
import sklearn.metrics as sklF
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
import torchvision.transforms as tvTransforms
from torchvision.utils import make_grid
from PIL import Image
import itertools
from functools import partial as par
from random import shuffle

# g(f(x)) -> F(x, f, g...)
def F(*z):
    z = list(z)
    return [*ft.reduce(lambda x, y: map(y, x), [z[:1]] + z[1:])][0]
# g(f([x1, x2...])) -> FF([x1, x2...], f, g...)
FF = lambda *z: [*ft.reduce(lambda x, y: map(y, x), z)]
# f(x1, x2..., y1, y2...) -> fyx(f, y1, y2...)(x1, x2...)
fyx = lambda f, *x: lambda *y: f(*y, *x)

time_start = time.time()
system = "WSL"
#system = "win"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
#method = "nn"
method = "load_nn"

nn_para = {
    "epochs": 100,
    "train_batch": 3,
    "val_batch": 2,
    "test_batch": 5,
    # learning rate for optimizer
    "lr_D": 0.0002,
    "lr_G": 0.0002,
    # betas = (beta1, beta2) for optimizer
    "betas_D": (0.5, 0.999),
    "betas_G": (0.5, 0.999),
    # lr_lambda for scheduler.
    # lambda_D for Discriminator
    # lambda_G for Generator
    "lambda_D": None,
    "lambda_G": None,
    # totol loss for generator
    # loss_G = loss_GAN
    #   + alpha_identity * loss_identity 
    #   + alpha_cycle * loss_cycle
    "alpha_identity": 5,
    "alpha_cycle": 10
}


def lambda_fn(epoch, decay_epoch):
    return 1 - max(0, epoch - decay_epoch) / (nn_para["epochs"] - decay_epoch)
#nn_para["lambda_D"] = fYx(lambda_fn, decay_epoch = 20)
#nn_para["lambda_G"] = fYx(lambda_fn, decay_epoch = 20)
nn_para["lambda_D"] = par(lambda_fn, decay_epoch = 20)
nn_para["lambda_G"] = par(lambda_fn, decay_epoch = 20)

G_para = dict(
    in_channels = 3,
    latent_channels = 64,
    num_updownSampling = 2,
    num_ResBlocks = 9
#   num_updownSampling = 2,
#   num_ResBlocks = 3
)
D_para = dict(
    in_channels = 3,
    latent_channels = 64,
    num_downSampling = 4,
#   out_channels = 1,
#   num_downSampling = 2,
#   p = 0.2
)

################################################################################
# Data


dataPath = {
    "WSL": ("/mnt/c/Users/maxp4/Documents/"
            "code/github/DLteam/data/cycleGAN1/"),
    "win": ("c:/Users/maxp4/Documents/"
            "code/github/DLteam/data/cycleGAN1/")
}
APath = dataPath[system] + "monet_jpg/"
BPath = dataPath[system] + "photo_jpg/"

transforms = [
    Image.open,
    tvTransforms.Compose([
        tvTransforms.RandomHorizontalFlip(),
        tvTransforms.ToTensor(),
        tvTransforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
]
#print(len(A_images))
#print(len(B_images))
#exit()
#print(A_images[0].shape)
#print(type(A_images[0]))
#print(A_images[0])
if method[:4] != "load":
    A_images = glob.glob(APath + "*.jpg")
    B_images = glob.glob(BPath + "*.jpg")
    shuffle(A_images)
    shuffle(B_images)
    A_images = FF(A_images, *transforms)
    B_images = FF(B_images[:len(A_images)], *transforms)
    imageData = data.TensorDataset(
        torch.stack(A_images),
        torch.stack(B_images)
    )
    trainData, valData = tts(imageData, test_size = 0.1, shuffle = False)
    trainloader = data.DataLoader(
        trainData, batch_size = nn_para["train_batch"]
    )
    valloader = data.DataLoader(valData, batch_size = nn_para["val_batch"])

################################################################################
# Network

class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels)
        )
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, in_channels, latent_channels,
                    num_updownSampling = 2, num_ResBlocks = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(in_channels),
            nn.Conv2d(in_channels, latent_channels, 2 * in_channels + 1),
            nn.InstanceNorm2d(latent_channels),
            nn.ReLU()
        )
        channels = latent_channels
        down = []
        for _ in range(num_updownSampling):
            latent_channels = channels * 2
            down += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(channels, latent_channels, 3, stride = 2),
                nn.InstanceNorm2d(latent_channels),
                nn.ReLU()
            ]
            channels = latent_channels
        self.down = nn.Sequential(*down)
        trans = [ResBlock(channels) for _ in range(num_ResBlocks)]
        self.trans = nn.Sequential(*trans)
        up = []
        for _ in range(num_updownSampling):
            latent_channels = channels // 2
            up += [
                nn.Upsample(scale_factor = 2),
                nn.ReflectionPad2d(1),
                nn.Conv2d(channels, latent_channels, 3),
                nn.InstanceNorm2d(latent_channels),
                nn.ReLU()
            ]
            channels = latent_channels
        self.up = nn.Sequential(*up)
        self.out = nn.Sequential(
            nn.ReflectionPad2d(in_channels),
            nn.Conv2d(channels, in_channels, 2 * in_channels + 1),
            nn.InstanceNorm2d(in_channels),
            nn.Tanh()
        )
    def forward(self, x):
        return F(x, self.conv, self.down, self.trans, self.up, self.out)

class Discriminator(nn.Module):
    def __init__(self, in_channels, latent_channels, out_channels = 1,
                    num_downSampling = 2, p = 0.2):
        super().__init__()
        layers = [nn.ReflectionPad2d(1),
                    nn.Conv2d(in_channels, latent_channels, 3, stride = 2),
                    nn.LeakyReLU(p)]
        channels = latent_channels
        for _ in range(num_downSampling):
            latent_channels = channels * 2
            layers += [nn.ReflectionPad2d(1),
                        nn.Conv2d(channels, latent_channels, 3, stride = 2),
                        nn.InstanceNorm2d(latent_channels), nn.LeakyReLU(p)]
            channels = latent_channels
        layers += [nn.ReflectionPad2d(1), nn.Conv2d(channels, out_channels, 3)]
        self.model = nn.Sequential(*layers)
        self.scale_factor = 2 ** (1 + num_downSampling)
        self.out_channels = out_channels
    def forward(self, x):
        return self.model(x)

# Generator(in_channels, latent_channels, up/down sampling, resBlock)
# Discriminator(in_channels, latent_channels, out_channels, down sampling)
G_AB = Generator(**G_para)
D_B = Discriminator(**D_para)

G_BA = Generator(**G_para)
D_A = Discriminator(**D_para)

print(device)
G_AB = G_AB.to(device)
D_B = D_B.to(device)
G_BA = G_BA.to(device)
D_A = D_A.to(device)

optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()),
    lr = nn_para["lr_G"], betas = nn_para["betas_G"]
)

optimizer_D_A = torch.optim.Adam(
    D_A.parameters(), lr = nn_para["lr_D"], betas = nn_para["betas_D"]
)

optimizer_D_B = torch.optim.Adam(
    D_B.parameters(), lr = nn_para["lr_D"], betas = nn_para["betas_D"]
)

if nn_para["lambda_G"]:
    scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda = nn_para["lambda_G"]
    )
if nn_para["lambda_D"]:
    scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda = nn_para["lambda_D"]
    )
if nn_para["lambda_D"]:
    scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda = nn_para["lambda_D"]
    )

GANLoss = nn.MSELoss()
cycleLoss = nn.L1Loss()
identityLoss = nn.L1Loss()

################################################################################
# Output Functions and Functional

def sample_images(real_A, real_B, name = None, figside = 1.5):
    assert real_A.size() == real_B.size(),\
        "The image size for two domains must be the same"
    G_AB.eval()
    G_BA.eval()
    real_A = real_A.to(device)
    fake_B = G_AB(real_A).detach()
    real_B = real_B.to(device)
    fake_A = G_BA(real_B).detach()
    nrows = real_A.size(0)
    real_A = make_grid(real_A, nrow = nrows, normalize = True)
    fake_B = make_grid(fake_B, nrow = nrows, normalize = True)
    real_B = make_grid(real_B, nrow = nrows, normalize = True)
    fake_A = make_grid(fake_A, nrow = nrows, normalize = True)
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)\
                            .cpu().permute(1, 2, 0)
    plt.figure(figsize = (figside * nrows, figside * 4))
    plt.imshow(image_grid)
    plt.axis("off")
    if name:
        plt.savefig(name + ".pdf")
    else:
        plt.savefig("test.pdf")
if method[:4] != "load":
    real_A, real_B = next(iter(valloader))
    sample_images(real_A, real_B) 

################################################################################
# Train

if method[:4] != "load":
    epochs = nn_para["epochs"]
    for epoch in range(epochs):
        for real_A, real_B in trainloader:
            real_A, real_B = real_A.to(device), real_B.to(device)
            out_shape_A = [
                real_A.size(0), D_A.out_channels,
                real_A.size(2) // D_A.scale_factor,
                real_A.size(3) // D_A.scale_factor
            ]
            out_shape_B = [
                real_B.size(0), D_B.out_channels,
                real_B.size(2) // D_B.scale_factor,
                real_B.size(3) // D_B.scale_factor
            ]
            validA = torch.ones(out_shape_A).to(device)
            validB = torch.ones(out_shape_B).to(device)
            fakeA = torch.zeros(out_shape_A).to(device)
            fakeB = torch.zeros(out_shape_B).to(device)
    
            # Train Generator
            G_AB.train()
            G_BA.train()
            optimizer_G.zero_grad()
            fake_B = G_AB(real_A)
            fake_A = G_BA(real_B)
            loss_id_A = identityLoss(fake_B, real_A)
            loss_id_B = identityLoss(fake_A, real_B)
            loss_identity = (loss_id_A + loss_id_B) / 2
            
            # GAN loss
            loss_GAN_AB = GANLoss(D_B(fake_B), validB) 
            loss_GAN_BA = GANLoss(D_A(fake_A), validA)
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2
            
            # cycle loss
            loss_cycle_A = cycleLoss(G_BA(fake_B), real_A)
            loss_cycle_B = cycleLoss(G_AB(fake_A), real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
            
            # G totol loss
            loss_G = loss_GAN\
                        + nn_para["alpha_identity"] * loss_identity\
                        + nn_para["alpha_cycle"] * loss_cycle
            loss_G.backward()
            optimizer_G.step()
            
            # Train Discriminator A
            optimizer_D_A.zero_grad()
            loss_real = GANLoss(D_A(real_A), validA)
            loss_fake = GANLoss(D_A(fake_A.detach()), fakeA)
            loss_D_A = (loss_real + loss_fake) / 2
            loss_D_A.backward()
            optimizer_D_A.step()
            
            # Train Discriminator B
            optimizer_D_B.zero_grad()
            loss_real = GANLoss(D_B(real_B), validB)
            loss_fake = GANLoss(D_B(fake_B.detach()), fakeB)
            loss_D_B = (loss_real + loss_fake) / 2
            loss_D_B.backward()
            optimizer_D_B.step()
        
        if (nn_para["lambda_G"]):
            scheduler_G.step()
        if (nn_para["lambda_D"]):
            scheduler_D_A.step()
        if (nn_para["lambda_D"]):
            scheduler_D_B.step()
        
        # Validation
        if not (epoch + 1) % 10:
            test_real_A, test_real_B = next(iter(valloader))
            sample_images(test_real_A, test_real_B, str(epoch + 1))
    
            loss_D = (loss_D_A + loss_D_B) / 2
            print(f"Epoch {epoch + 1}/{epochs}")
            print(
                f"G loss: {loss_G.item()} | "
                f"identity: {loss_identity.item()} "
                f"GAN: {loss_GAN.item()} cycle: {loss_cycle.item()}"
            )
            print(
                f"D loss: {loss_D.item()} | "
                f"D_A: {loss_D_A.item()} "
                f"D_B: {loss_D_B.item()}"
            )
    torch.save(G_BA.cpu(), "generator_p2m.model")
else:
    B_images = glob.glob(BPath + "*.jpg")
    step = len(B_images) // 10
    for part, pos in enumerate(range(0, len(B_images), step)):
        part_images = FF(B_images[pos:pos + step], *transforms)
        testloader = data.DataLoader(
            data.TensorDataset(torch.stack(part_images)),
            batch_size = nn_para["test_batch"]
        )
        G_BA = torch.load("generator_p2m.model").to(device).eval()
        save_dir = "output/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for i, photo in enumerate(testloader):
            fake_A = G_BA(photo[0].to(device)).detach().cpu()
            for j in range(fake_A.size(0)):
#               image_grid = make_grid([fake_A[j]], 1, 0, True)
#           #   print(image_grid.shape)
#               plt.figure()
#               plt.imshow(image_grid.permute(1, 2, 0))
#               plt.axis("off")
#               plt.savefig(
#                   save_dir + "part" + str(part + 1) +\
#                   "batch" + str(i + 1) + "Nr" + str(j + 1) + ".pdf"
#               )
#               plt.close()
                img = fake_A[j].permute(1, 2, 0).numpy()
                img = (img - np.min(img)) * 255 / (np.max(img) - np.min(img))
                tvTransforms.ToPILImage()(img.astype(np.uint8)).save(
                    save_dir + "part" + str(part + 1) +\
                    "batch" + str(i + 1) + "Nr" + str(j + 1) + ".pdf"
                )

print(time.time() - time_start)

