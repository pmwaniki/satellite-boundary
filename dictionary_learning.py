import rasterio
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from settings import data_dir,result_dir
from loaders import get_image,get_mask



data=pd.read_csv(os.path.join(data_dir,"Train.csv"))
test=pd.read_csv(os.path.join(data_dir,"Test.csv"))
test['index']=range(test.shape[0])

im_file=data.iloc[10]['dir']
mask_file=data.iloc[10]['mask_path']

image=get_image(im_file)
mask=get_mask(mask_file)

fig,axs=plt.subplots(2,1)
axs[0].imshow(image[0:3,:,:].transpose(1,2,0),vmax=1,vmin=0)
axs[1].imshow(mask,alpha=0.15)
axs[0].set_axis_off()
axs[1].set_axis_off()
plt.show()

def plot(images,masks=None):
    n_images=images.shape[0]
    fig,axes=plt.subplots(1 if masks==None else 2, n_images)
    for i in range(n_images):
        axes[0,i].imshow(images[i][0:3, :, :].transpose(1, 2, 0), vmax=1, vmin=0)
        axes[1,i].imshow(masks[i], alpha=0.15)
    plt.show()
    return fig
