import math
import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


def load_raster(fname):
    with rasterio.open(fname) as src:
        return src.read(1)

def normalize(array):
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)


#@lru_cache(maxsize=32)
def get_image(image_dir,normalize_fun=normalize):
    assets = ('B01.tif', 'B02.tif', 'B03.tif', 'B04.tif')
    source_samples = []
    for asset in assets:
        source= load_raster(os.path.join(image_dir, asset))
        source_samples.append(normalize_fun(source))
    source_samples = np.array(source_samples)

    return source_samples

def get_mask(image_dir):
    mask= load_raster(os.path.join(image_dir, 'raster_labels.tif'))
    return mask

class SimpleDataset(Dataset):
    def __init__(self, data,dir_image,dir_mask=None,transforms=[]):
        self.data=data
        self.dir_image=dir_image
        self.dir_mask=dir_mask
        self.transforms=transforms
        # self.label_dimension = label_dimension
        # self.cached_item = [None, None]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        image=get_image(self.data.iloc[idx][self.dir_image])
        for t in self.transforms:
            image=t(image)
        if self.dir_mask is None:
            return torch.FloatTensor(image)
        else:
            mask=get_mask(self.data.iloc[idx][self.dir_mask])
        image_tensor,mask_tensor=torch.FloatTensor(image), torch.FloatTensor(mask)
        return image_tensor,mask_tensor


class AEDataset(Dataset):
    def __init__(self, data,dir_image,idvar='id',same_image=False):
        self.data=data
        self.dir_image=dir_image
        self.idvar=idvar
        self.ids=data[idvar].unique()
        self.same_image=same_image
        # self.label_dimension = label_dimension
        # self.cached_item = [None, None]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        subset=self.data.loc[self.data[self.idvar]==self.ids[idx],:].sample(2).reset_index()
        image1=get_image(subset.iloc[0][self.dir_image])
        if self.same_image:
            image2=image1
        else:
            image2 = get_image(subset.iloc[1][self.dir_image])
        # for t in self.transforms:
        #     image=t(image)
        # if self.dir_mask is None:
        #     return torch.FloatTensor(image)
        # else:
        #     mask=get_mask(self.data.iloc[idx][self.dir_mask])
        # image_tensor,mask_tensor=torch.FloatTensor(image1), torch.FloatTensor(image2)
        return torch.FloatTensor(image1), torch.FloatTensor(image2)



class GeneratedDataset(Dataset):
    def __init__(self, data,dir_image,dir_generated,dir_mask=None):
        self.data=data
        self.dir_image=dir_image
        self.dir_mask=dir_mask
        self.dir_generated=dir_generated
        # self.label_dimension = label_dimension
        # self.cached_item = [None, None]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        record = self.data.iloc[idx]
        images=joblib.load(os.path.join(self.dir_generated,f'Vae-train_{record["id"]}_{record["period"]}.joblib'))
        selected=torch.randint(images.shape[0],(1,)).item()
        image=images[selected]
        if self.dir_mask is None:
            return torch.FloatTensor(image)
        else:
            mask=get_mask(self.data.iloc[idx][self.dir_mask])
        image_tensor,mask_tensor=torch.FloatTensor(image), torch.FloatTensor(mask)
        return image_tensor,mask_tensor