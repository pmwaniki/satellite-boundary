import os
from pathlib import Path
import re

import pandas as pd
from radiant_mlhub import Dataset, DownloadIfExistsOpts

from settings import data_dir

dataset_id = 'nasa_rwanda_field_boundary_competition'
assets = ['labels']

dataset = Dataset.fetch(dataset_id)

dataset.download(data_dir,if_exists=DownloadIfExistsOpts.overwrite)



test_pattern="nasa_rwanda_field_boundary_competition_source_test"
train_pattern="nasa_rwanda_field_boundary_competition_source_train"
mask_pattern="nasa_rwanda_field_boundary_competition_labels_train"

mask_dirs=[i for i in Path(os.path.join(data_dir,"nasa_rwanda_field_boundary_competition/nasa_rwanda_field_boundary_competition_labels_train")).glob(mask_pattern+"*")]
train_dirs=[i for i in Path(os.path.join(data_dir,"nasa_rwanda_field_boundary_competition",train_pattern)).glob(train_pattern+"*")]
test_dirs=[i for i in Path(os.path.join(data_dir,"nasa_rwanda_field_boundary_competition",test_pattern)).glob(test_pattern+"*")]

train_data=pd.DataFrame({'path':[p.as_posix() for p in train_dirs],
                         'folder':[p.parts[-1] for p in train_dirs]})
train_data['id']=train_data['folder'].map(lambda x: re.search("nasa_rwanda_field_boundary_competition_source_train_(\d+)_",x).groups()[0])
train_data['period']=train_data['folder'].map(lambda x: re.search("nasa_rwanda_field_boundary_competition_source_train_\d+_(\d+_\d+)",x).groups()[0])

test_data=pd.DataFrame({'path':[p.as_posix() for p in test_dirs],
                         'folder':[p.parts[-1] for p in test_dirs]})
test_data['id']=test_data['folder'].map(lambda x: re.search("nasa_rwanda_field_boundary_competition_source_test_(\d+)_",x).groups()[0])
test_data['period']=test_data['folder'].map(lambda x: re.search("nasa_rwanda_field_boundary_competition_source_test_\d+_(\d+_\d+)",x).groups()[0])

mask_data=pd.DataFrame({'mask_path':[p.as_posix() for p in mask_dirs],
                         'folder':[p.parts[-1] for p in mask_dirs]})
mask_data['id']=mask_data['folder'].map(lambda x: re.search(".+_(\d+)",x).groups()[0])

train_data=pd.merge(train_data,mask_data[['mask_path','id']],how='left',on=['id'])

train_data.to_csv(os.path.join(data_dir,'Train.csv'),index=False)
test_data.to_csv(os.path.join(data_dir,'Test.csv'),index=False)
