import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pyro.contrib.examples.util import MNIST
import torch.nn as nn
import torchvision.transforms as transforms

import kornia as K

import pyro
import pyro.distributions as dist
import pyro.contrib.examples.util  # patches torchvision
from pyro.infer import SVI, Trace_ELBO, TraceGraph_ELBO
from pyro.optim import Adam
from pyro import poutine
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from ae_modules import Encoder,Decoder
from loaders import SimpleDataset,AEDataset
from settings import data_dir, result_dir

USE_CUDA=True

data=pd.read_csv(os.path.join(data_dir,"Train.csv"))
test=pd.read_csv(os.path.join(data_dir,"Test.csv"))
test['index']=range(test.shape[0])

train_ids,val_ids=train_test_split(data['id'].unique(),test_size=0.1,random_state=123)
train,val=data.loc[data['id'].isin(train_ids),:].copy(),data.loc[data['id'].isin(val_ids),:].copy()

train_dataset=AEDataset(data=train,dir_image='path',same_image=True)
train_loader=DataLoader(train_dataset,batch_size=8,shuffle=True,drop_last=True)
test_dataset=AEDataset(data=val,dir_image='path',same_image=True)
test_loader=DataLoader(test_dataset,batch_size=4,shuffle=False)

# image enhancement
#normalization

def equalize(x):
    return K.enhance.equalize_clahe(x,clip_limit=30.,grid_size=(8,8))
    # return x

images=iter(train_loader).__next__()[0]
if USE_CUDA: images=images.cuda()
images2=equalize(images)

n_images=images.shape[0]
fig,axes=plt.subplots( 2, n_images,figsize=(20,4))
for i in range(n_images):
    axes[0,i].imshow(images.cpu().numpy()[i][0:3, :, :].transpose(1, 2, 0), vmax=1, vmin=0)
    axes[1,i].imshow(images2.cpu().numpy()[i][0:3, :, :].transpose(1, 2, 0), vmax=1, vmin=0)
    axes[0, i].set_axis_off()
    axes[1, i].set_axis_off()
plt.savefig("/tmp/vae-reconstruction.png")
plt.show()



class AE(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self, z_dim=32, use_cuda=True):
        super().__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(latent_dim=z_dim)
        self.decoder = Decoder(latent_dim=z_dim)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim

    # define the model p(x|z)p(z)
    def model(self, x,x2):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # decode the latent code z
            loc_img = self.decoder(z)
            # score against actual images
            pyro.sample("obs", dist.ContinuousBernoulli(probs=loc_img).to_event(3), obs=x2)

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x,x2):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)

        with pyro.plate("data", x.shape[0]):
            z=self.encoder(x)
            pyro.sample("latent", dist.Delta(z).to_event(1))

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        with torch.no_grad():
            latent = self.encoder(x)

            z = dist.Delta(latent).sample()
            # decode the image (note we don't sample in image space)
            loc_img = self.decoder(z)
        return loc_img


vae=AE(z_dim=2048,use_cuda=USE_CUDA)

def init_weights(m):
    if isinstance(m,nn.Linear):
        # torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.normal_(m.weight,std=0.001)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Conv2d):
        # nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        torch.nn.init.normal_(m.weight, std=0.001)
    elif isinstance(m, nn.ConvTranspose2d):
        # nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        torch.nn.init.normal_(m.weight, std=0.001)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

vae.apply(init_weights)

# x=torch.randn((5,4,256,256))
# guide_trace = poutine.trace(vae.guide).get_trace(x,x)
# # run the model and replay it against the samples from the guide
# model_trace = poutine.trace(
#         poutine.replay(vae.model, trace=guide_trace)).get_trace(x,x)

aug_list = K.augmentation.AugmentationSequential(
        K.augmentation.RandomResizedCrop(size=(256,256),scale=(0.5,1.0),p=1.0),
        K.augmentation.RandomHorizontalFlip(p=0.5),
        K.augmentation.RandomVerticalFlip(p=0.5),
        # K.RandomAffine(degrees=(0, 90), p=0.25),
        # K.augmentation.RandomGaussianBlur(kernel_size=(3,3), sigma=(0.1, 2.0), p=0.5),
        data_keys=["input",],
        same_on_batch=False,
        random_apply=10,keepdim=True
    )
if USE_CUDA: aug_list=aug_list.cuda()


def train(svi, train_loader, use_cuda=False):
    # initialize loss accumulator
    epoch_loss = 0.
    # do a training epoch over each mini-batch x returned
    # by the data loader
    for x,_ in train_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
            # x2=x2.cuda()
        x = equalize(x)
        x=aug_list(x)
        # x2 = equalize(x2)
        # do ELBO gradient and accumulate loss
        epoch_loss += svi.step(x,x)

    # return epoch loss
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train

def evaluate(svi, test_loader, use_cuda=False):
    # initialize loss accumulator
    test_loss = 0.
    # compute the loss over the entire test set
    for x,x2 in test_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
            x2=x2.cuda()
        x=equalize(x)
        x2=equalize(x2)
        # compute ELBO estimate and accumulate loss
        test_loss += svi.evaluate_loss(x,x2)
    normalizer_test = len(test_loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test

lr=0.001
optimizer = Adam({"lr": lr, "betas": (0.90, 0.999),'weight_decay':0.0001,}, {"clip_norm": 0.1})

# optimizer = pyro.optim.SGD({"eta": lr,})
svi = SVI(vae.model, vae.guide, optimizer, loss=TraceGraph_ELBO(num_particles=5))

pyro.clear_param_store()
train_elbo = []
test_elbo = []
# training loop
for epoch in range(12000):
    total_epoch_loss_train = train(svi, train_loader, use_cuda=USE_CUDA)
    train_elbo.append(-total_epoch_loss_train)
    # print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

    if epoch % 10 == 0:
        # report test diagnostics
        total_epoch_loss_test = evaluate(svi, test_loader, use_cuda=USE_CUDA)
        test_elbo.append(-total_epoch_loss_test)
        print("[epoch %03d] average test loss: %.4f" % (epoch, total_epoch_loss_test))


plt.plot(train_elbo,label="train")
# plt.plot(test_elbo,label="test")
plt.legend()
plt.savefig("/tmp/vae-elbos.png")
plt.show()


images=iter(train_loader).__next__()[0]
if USE_CUDA: images=images.cuda()
images=equalize(images)
recon=vae.reconstruct_img(images)

n_images=images.shape[0]
fig,axes=plt.subplots( 2, n_images,figsize=(20,4))
for i in range(n_images):
    axes[0,i].imshow(images.cpu().numpy()[i][0:3, :, :].transpose(1, 2, 0), vmax=1, vmin=0)
    axes[1,i].imshow(recon.cpu().numpy()[i][0:3, :, :].transpose(1, 2, 0), vmax=1, vmin=0)
    axes[0, i].set_axis_off()
    axes[1, i].set_axis_off()
plt.savefig("/tmp/vae-reconstruction.png")
plt.show()


torch.save((vae.encoder.state_dict(),vae.decoder.state_dict()),os.path.join(result_dir,"vae-weights.pth"))

train_simple_dataset=SimpleDataset(data,dir_image="dir")
train_simple_loader=DataLoader(train_simple_dataset,batch_size=8,shuffle=False)

test_simple_dataset=SimpleDataset(test,dir_image="path")
test_simple_loader=DataLoader(test_simple_dataset,batch_size=8,shuffle=False)

train_reconstructed=[]
for x in train_simple_loader:
    # if on GPU put mini-batch into CUDA memory
    if USE_CUDA:
        x = x.cuda()
    x = equalize(x)
    recon = vae.reconstruct_img(x)
    train_reconstructed.append(recon)
train_reconstructed=torch.cat(train_reconstructed,dim=0).cpu().numpy()
train_ids=data['id'].values

joblib.dump((train_ids,train_reconstructed),filename=os.path.join(result_dir,"AE-train.joblib"))

test_reconstructed=[]
for x in test_simple_loader:
    # if on GPU put mini-batch into CUDA memory
    if USE_CUDA:
        x = x.cuda()
    x = equalize(x)
    recon = vae.reconstruct_img(x)
    test_reconstructed.append(recon)
test_reconstructed=torch.cat(test_reconstructed,dim=0).cpu().numpy()
test_ids=test['id'].values

joblib.dump((test_ids,test_reconstructed),filename=os.path.join(result_dir,"AE-test.joblib"))