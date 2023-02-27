import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from kornia.augmentation.container import AugmentationSequential
import kornia as K

import ray
from torchgeo.transforms import indices
from modules import UNet
# ray.init(address="auto")
ray.init( num_cpus=12,dashboard_host="0.0.0.0")

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from settings import data_dir,result_dir,log_dir
from loaders import SimpleDataset

experiment="Unet-Baseline"
device= 'cuda' if torch.cuda.is_available() else "cpu"
data=pd.read_csv(os.path.join(data_dir,"Train.csv"))
test=pd.read_csv(os.path.join(data_dir,"Test.csv"))
test['index']=range(test.shape[0])

train_ids,val_ids=train_test_split(data['id'].unique(),test_size=0.1,random_state=123)
train,val=data.loc[data['id'].isin(train_ids),:].copy(),data.loc[data['id'].isin(val_ids),:].copy()


train_dataset=SimpleDataset(data=train,dir_image='path',dir_mask='mask_path')
train_loader=DataLoader(train_dataset,batch_size=8,shuffle=True,drop_last=True)

val_dataset=SimpleDataset(data=val,dir_image='path',dir_mask='mask_path')
val_loader=DataLoader(val_dataset,batch_size=4,shuffle=False)

configs = {
    'low_threshold':tune.uniform(0.1,0.5),
    'high_threshold':tune.uniform(0.1,0.49),
    'kernel_size':tune.choice([3,5,7,11]),
    'sigma':tune.loguniform(0.1,10.0),
    'grid_size':tune.choice([5,8,10,16]),
    'clip_limit':tune.choice([20.,30.,50.]),
    'drop_channel':tune.choice([0,1,2,3])

}

config={i:v.sample() for i,v in configs.items()}

def equalize(x,config):
    return K.enhance.equalize_clahe(x,clip_limit=config['clip_limit'],
                                    grid_size=(config['grid_size'],config['grid_size']))


def canny(x,config):
    channels=list(range(4))
    channels.remove(config['drop_channel'])
    magnitude, edges=K.filters.canny(x[:,channels,:,:],low_threshold=config['low_threshold'],
                                     high_threshold=config['low_threshold']+config['high_threshold'],
                    kernel_size=(config['kernel_size'], config['kernel_size']), sigma=(config['sigma'], config['sigma']),
                    hysteresis=True, eps=1e-06)
    return edges

def classify(x,config):
    z=equalize(x,config)
    mask=canny(z,config)
    return mask


images,masks=iter(train_loader).__next__()
pred_masks=classify(images,config)

n_images=images.shape[0]
fig,axes=plt.subplots( 3, n_images,figsize=(8,4))
for i in range(n_images):
    axes[0,i].imshow(images.numpy()[i][0:3, :, :].transpose(1, 2, 0), vmax=1, vmin=0)
    axes[1,i].imshow(masks[i].numpy(), alpha=1.0)
    axes[2, i].imshow(pred_masks[i].numpy().squeeze(0), alpha=1.0)
    axes[1, i].set_axis_off()
    axes[0, i].set_axis_off()
    axes[2, i].set_axis_off()
plt.show()


def augmentation(config):
    aug_list = AugmentationSequential(

        K.augmentation.RandomHorizontalFlip(p=0.5),
        K.augmentation.RandomVerticalFlip(p=0.5),
        # K.RandomAffine(degrees=(0, 90), p=0.25),
        K.augmentation.RandomGaussianBlur(kernel_size=(config['gaussian_kernel_size'], config['gaussian_kernel_size']), sigma=(0.1, 2.0), p=config['prop_gaussian']),
        K.augmentation.RandomResizedCrop(size=(256,256),scale=(0.5,1.0),p=1.0,resample="bicubic",align_corners=True,cropping_mode="resample"),
        data_keys=["input", "mask"],
        same_on_batch=False,
        random_apply=10,keepdim=True
    ).to(device)
    return aug_list