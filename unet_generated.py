import os

import joblib
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
from loaders import GeneratedDataset

experiment="Unet-VAE-generated"
device= 'cuda' if torch.cuda.is_available() else "cpu"
data=pd.read_csv(os.path.join(data_dir,"Train.csv"))
test_ids,test_array=joblib.load(os.path.join(result_dir,"VAE-test.joblib"))
# test=pd.read_csv(os.path.join(data_dir,"Test.csv"))
# test['index']=range(test.shape[0])

train_ids,val_ids=train_test_split(data['id'].unique(),test_size=0.1,random_state=123)
train,val=data.loc[data['id'].isin(train_ids),:].copy(),data.loc[data['id'].isin(val_ids),:].copy()


configs = {
    'batch_size':tune.choice([8,]),
    'lr':tune.loguniform(0.0001,0.1),
    'l2':tune.loguniform(0.00001,0.01),
    'gaussian_kernel_size':tune.choice([3,5,7]),
    'prop_gaussian':tune.choice([0.01,0.1,0.25,0.5]),
    'pos_weight': tune.choice([10.0,15.0,20.0, 25.0,35.0,40.0])

}

config={i:v.sample() for i,v in configs.items()}

# def equalize(x):
#     return K.enhance.equalize_clahe(x,clip_limit=30.,grid_size=(8,8))

# add_bands=nn.Sequential(
#     indices.AppendNDVI(index_nir=3, index_red=0),
#     # indices.AppendNDWI(index_green=1, index_nir=3),
#     # indices.AppendGNDVI(index_nir=3,index_green=1),
#     # indices.AppendBNDVI(index_nir=3,index_blue=2),
#     # indices.AppendGRNDVI(index_nir=3,index_green=1,index_red=0),
#     # indices.AppendGBNDVI(index_nir=3,index_green=1,index_blue=2),
# ).to(device)

# broadcast_array_offset=torch.tensor([0] * 4 + [1] * len(add_bands)).unsqueeze(1).unsqueeze(2).to(device)
# broadcast_array_scale=torch.tensor([1] * 4 + [2] * len(add_bands)).unsqueeze(1).unsqueeze(2).to(device)


def augmentation(config):
    aug_list = AugmentationSequential(

        K.augmentation.RandomHorizontalFlip(p=0.5),
        K.augmentation.RandomVerticalFlip(p=0.5),
        # K.RandomAffine(degrees=(0, 90), p=0.25),
        K.augmentation.RandomGaussianBlur(kernel_size=(config['gaussian_kernel_size'], config['gaussian_kernel_size']), sigma=(0.1, 2.0), p=config['prop_gaussian']),
        data_keys=["input", "mask"],
        same_on_batch=False,
        random_apply=10,keepdim=True
    ).to(device)
    return aug_list

def get_train_loader(config):
    train_dataset=GeneratedDataset(data=train,dir_image='path',dir_mask='mask_path',dir_generated=os.path.join(result_dir,"vae-generated"))
    train_loader=DataLoader(train_dataset,batch_size=config['batch_size'],shuffle=True,drop_last=True)
    return train_loader
def get_val_loader():
    val_dataset=GeneratedDataset(data=val,dir_image='path',dir_mask='mask_path',dir_generated=os.path.join(result_dir,"vae-generated"))
    val_loader=DataLoader(val_dataset,batch_size=4,shuffle=False)
    return val_loader

def get_optimizer(config,model):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config['lr'], weight_decay=config['l2'],)
    return optimizer

def get_model(config):
    # model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=4 + len(add_bands), out_channels=1, init_features=32, pretrained=False)
    model=UNet(n_classes=1,n_channels=4)
    return model








class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc

dice_loss=DiceLoss().to(device)

def plot(images,masks=None):
    n_images=images.shape[0]
    fig,axes=plt.subplots(1 if masks is None else 2, n_images,figsize=(16,4))
    for i in range(n_images):
        axes[0,i].imshow(images[i][0:3, :, :].transpose(1, 2, 0), vmax=1, vmin=0)
        axes[1,i].imshow(masks[i].squeeze(0), alpha=1.0)
        axes[1, i].set_axis_off()
        axes[0, i].set_axis_off()
    plt.show()
    return fig


def train_fun(model,optimizer,criterion,train_loader,val_loader,grad_scaler,scheduler=None,iteration=0,aug_list=None):
    model.train()
    train_loss=0
    for batch_x,batch_y in train_loader:
        batch_x,batch_y=batch_x.to(device),batch_y.unsqueeze(1).to(device)
        if aug_list is not None:
            batch_x,batch_y=aug_list(batch_x,batch_y)
        batch_x = batch_x - 0.5 #+ torch.randn(batch_x.size(), device=batch_x.device) * gaus_sd
        logits=model(batch_x)
        # loss=criterion(logits,batch_y)
        # loss = loss + dice_loss(logits.sigmoid(),batch_y)
        loss = dice_loss(logits.sigmoid(), batch_y)
        optimizer.zero_grad()
        # grad_scaler.scale(loss).backward()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        # grad_scaler.step(optimizer)
        # grad_scaler.update()

        train_loss += loss.item() / len(train_loader)

    model.eval()
    val_loss = 0
    pred_val = []
    obs_val = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device, dtype=torch.float), batch_y.unsqueeze(1).to(device)
            batch_x=batch_x - 0.5
            logits = model(batch_x)

            # loss = criterion(logits, batch_y)
            # loss = loss + dice_loss(logits.sigmoid(),batch_y)
            loss = dice_loss(logits.sigmoid(), batch_y)
            val_loss += loss.item() / len(val_loader)
            pred_val.append(logits.sigmoid().squeeze().cpu().numpy().reshape(-1))
            obs_val.append(batch_y.squeeze().cpu().numpy().reshape(-1))
    if scheduler: scheduler.step()
    pred_val = np.concatenate(pred_val)
    obs_val = np.concatenate(obs_val)
    f1 = f1_score(obs_val, (pred_val > 0.5) * 1.0)
    return train_loss,val_loss,f1


class Trainer(tune.Trainable):
    def setup(self, config):
        self.model=get_model(config).to(device)
        self.optimizer=get_optimizer(config,self.model)
        self.criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config['pos_weight'])).to(device)
        # self.criterion=DiceLoss().to(device)
        self.grad_scaler=torch.cuda.amp.GradScaler(enabled=True)
        self.scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=100,gamma=0.5)
        self.train_loader=get_train_loader(config)
        self.val_loader=get_val_loader()
        self.aug_list=augmentation(config)


    def step(self):
        train_loss,loss,f1=train_fun(self.model,self.optimizer,self.criterion,self.train_loader,self.val_loader,
                                     self.grad_scaler,self.scheduler,self.iteration,self.aug_list)
        return {'loss':loss,'f1':f1,'train_loss':train_loss}

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save((self.model.state_dict(),self.optimizer.state_dict()), checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        model_state,optimizer_state=torch.load(checkpoint_path)
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)

epochs=250
scheduler = ASHAScheduler(
        metric="f1",
        mode="max",
        max_t=epochs,
        grace_period=50,
        reduction_factor=4)



reporter = CLIReporter( metric_columns=["loss","train_loss","f1", "training_iteration"])
# early_stopping=tune.stopper.EarlyStopping(metric='auc',top=10,mode='max',patience=10)
result = tune.run(
    Trainer,
    # metric='loss',
    # mode='min',
    checkpoint_at_end=True,
    resources_per_trial={"cpu": 6, "gpu": 0.5},
    config=configs,
    local_dir=log_dir,
    num_samples=100,
    name=experiment,
    # stop=MaxIterStopper(),
    resume=False,
    scheduler=scheduler,
    progress_reporter=reporter,
    reuse_actors=False,
    raise_on_failed_trial=False,
    # max_failures=1
)



df = result.results_df
metric='f1';mode="max"; scope='last'
print(result.get_best_trial(metric,mode,scope=scope).last_result)
# df.to_csv(os.path.join(data_dir, "results/hypersearch.csv"), index=False)
best_trial = result.get_best_trial(metric, mode, scope=scope)
best_config=result.get_best_config(metric,mode,scope=scope)

# test_dataset=SimpleDataset(test,dir_image='path')
# test_loader=DataLoader(test_dataset,shuffle=False,batch_size=4,num_workers=4)

best_checkpoint=result.get_best_checkpoint(best_trial,metric,mode,return_path=True)
model_state,_=torch.load(os.path.join(best_checkpoint,"model.pth"))
torch.save((model_state,best_config),f=os.path.join(result_dir,f"{experiment}-weights.pth"))
# best_trainer=Trainer(best_config)
best_model=get_model(best_config)
best_model.load_state_dict(model_state)
best_model.to(device)
# Test model accuracy

best_model.eval()
with torch.no_grad():
    batch_x=torch.FloatTensor(test_array)
    batch_x = batch_x.to(device, dtype=torch.float)
    batch_x = batch_x - 0.5
    pred_test=best_model(batch_x).sigmoid().squeeze(1).cpu().numpy()





submission=[]
for index in range(13):
    for row in range(256):
        for col in range(256):
            submission.append({"id":index,'row':row,'column':col,'label':pred_test[index][row][col]})

submission=pd.DataFrame(submission)
submission['tile_row_column']=submission.apply(lambda row:f'Tile{row["id"]:02.0f}_{row["row"]:.0f}_{row["column"]:.0f}',axis=1)
submission['label']=submission['label'].map(lambda x:int(x>=0.5))

submission[['tile_row_column','label']].to_csv(os.path.join(result_dir,f"Submission-{experiment}.csv"),index=False)