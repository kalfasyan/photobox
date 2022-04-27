import logging
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from natsort import index_natsorted
from PIL import Image
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
#from tensorflow.keras.applications import EfficientNetB0
#from tensorflow.keras.layers import (Activation, Dense, Dropout,
#                                     GlobalAveragePooling2D)
#from tensorflow.keras.models import load_model, Model
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from category_encoders import OrdinalEncoder
from tqdm import tqdm
from common import confidence_threshold

from utils import (copy_files, non_max_suppression_fast, overlay_yolo,
                   save_insect_crops)
seed = 42

logger = logging.getLogger(__name__)

# Creating a mapping of insects based on the trained model
oe = OrdinalEncoder(cols=['label'], 
                mapping=[{'col': 'label', 
                        'mapping': {'bl':0, 
                                    'wswl':1,
                                    'sp': 2,
                                    't':3,
                                    'sw':4,
                                    'k':5,
                                    'm':6,
                                    'c':7,
                                    'v':8,
                                    'wmv':9,
                                    'wrl':10,
                                    'other':11}}])

oe_mapping_dict = oe.mapping[0]['mapping']
inv_oe_mapping_dict = {v:k for k,v in oe_mapping_dict.items()}
oe_insect_names = list(oe_mapping_dict.keys())
# ['bl', 'wswl', 'sp', 't', 'sw', 'k', 'm', 'c', 'v', 'wmv', 'wrl', 'other']
oe_insect_codes = list(oe_mapping_dict.values())
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

class ModelDetections(object):

    def __init__(self, path, img_dim=150, target_classes=oe_insect_names):
        self.path = str(path)
        self.df = pd.DataFrame({"filename": list(Path(self.path).rglob('*.png'))}).astype(str)
        self.df['insect_idx'] = self.df.filename.apply(lambda x: x.split('/')[-1].split('_')[-1][:-4])
        self.target_classes = target_classes
        self.le = LabelEncoder()
        self.le.fit(self.target_classes)
        self.nb_classes = len(self.target_classes)
        self.img_dim = img_dim

    def create_data_generator(self):        
        self.dataset = InsectImgDataset(df=self.df, transform=T.Compose(transforms_list_test))
        self.dataset.extract_df_info()
        logger.info(f"Found {len(self.dataset)} insect detections")
        
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)


    def get_predictions(self, model, confidence_threshold=confidence_threshold):
        
        self.softmax_out = [0]*len(self.dataset)
        self.insect_inds = []

        torch.backends.cudnn.benchmark = True
        model.eval()
        with torch.no_grad():
            for i, (x, path,insect_idx,fpath,date,xtra,height,width) in tqdm(enumerate(self.dataloader), total=len(self.dataset), desc='Predicting insect classes..'):
                out = model(x.float()).detach().cpu().float()
                pred = fn.softmax(out, dim=1).detach().cpu().numpy()
            
                self.softmax_out[i] = pred
                self.insect_inds.append(insect_idx[0])
            
        self.softmax_out = np.vstack(self.softmax_out)
        
        df = pd.DataFrame(self.softmax_out)*100.
        df.columns = df.columns.map(inv_oe_mapping_dict)
        df['uncertain'] = df[oe_insect_names].max(axis=1)<confidence_threshold
        df['prediction'] = df[oe_insect_names].idxmax(axis=1)#.map(inv_oe_mapping_dict)
        df['insect_idx'] = pd.Series(self.insect_inds)#.astype(int)
        df['top_prob'] = df[oe_insect_names].max(axis=1)
        df['top_class'] = df[oe_insect_names].idxmax(axis=1)
        
        df = pd.merge(self.df, df, on='insect_idx')
        self.df = df.sort_values(by="insect_idx", 
                                key=lambda x: np.argsort(index_natsorted(x))).reset_index(drop=True)


class InsectModel(object):

    def __init__(self, modelname='', img_dim=150, nb_classes=len(oe_insect_names)):
        self.img_dim = img_dim
        self.nb_classes = nb_classes
        self.modelname = modelname
        self.path = f'{modelname}_photobox_best.pth.tar'
        
    def load(self):
        model = model_selector(self.modelname, pretrained=True)
        
        if self.modelname.startswith("dense"):
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Sequential(
                                        nn.Linear(num_ftrs,512),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(512, self.nb_classes))
        if self.modelname.startswith("resn"):
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs,len(oe_insect_names))
        if self.modelname.startswith("efficient"):
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs,len(oe_insect_names))
        if self.modelname.startswith("vgg"):
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_ftrs, len(oe_insect_names))
        if self.modelname.startswith("mobile"):
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_ftrs, len(oe_insect_names))

        optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
        model, optimizer = load_checkpoint(self.path, model, optimizer)
        model = model.to('cpu', dtype=torch.float)
        model.eval()

        return model


    def predict(self, generator, batch_size=1, verbose=1):
        return self.model.predict(generator, batch_size=batch_size, verbose=verbose)


# Transforms to be applied to detected insects
transforms_list_test = [
    T.ToPILImage(),
    T.Resize(size=(150,150)),
    T.ToTensor(),
]

class InsectImgDataset(Dataset):
    """
    Dataset class that can take a dataset directory as input.
    It creates a dataframe with all relevant insect info such as: sticky plate name, year, date etc.
    """

    def __init__(self, df=pd.DataFrame(), directory='', ext='.png', setting="photobox", img_dim=150, transform=None):
        self.setting = setting
        self.directory = str(directory)
        self.ext = ext
        self.df = df

        if not len(self.directory):
            assert len(self.df), "You chose to use a pre-made dataframe."
        else:
            assert len(self.directory)
            assert not len(self.df), "You chose to use a directory and load its filenames into a dataframe."

            self.files = get_files(directory, ext=self.ext)
            
            self.df = pd.DataFrame(self.files, columns=['filename'])
            self.df = self.df.astype(str).reset_index(drop=True)
        
        self.img_dim = img_dim
        self.transform = transform

    def extract_df_info(self, fix_cols=False):
        self.df['location'] = self.df.filename.apply(lambda x: x.split('_')[0])
        self.df['date'] = self.df.filename.apply(lambda x: x.split('_')[1])
        self.df['xtra'] = self.df.filename.apply(lambda x: x.split('_')[2])
        self.df['resolution'] = self.df.filename.apply(lambda x: x.split('_')[3])
        self.df['insect_idx'] = self.df.filename.apply(lambda x: x.split('_')[4][:-4])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if len(self.df.columns) > 1:
            sample = self.df.loc[idx]
        else:
            print("Run extract_df_info to read image data.")
            sample = extract_filename_info(str(self.df.loc[idx].filename), setting=self.setting)
            return sample

        fname = sample["full_filename"]
        location = sample["location"]
        resolution = sample["resolution"]
        date = sample["date"]
        xtra = sample["xtra"]
        insect_idx = sample["insect_idx"]

        img = torchvision.io.read_image(fname)
        
        _, width, height = img.shape#tensor_img.size()

        sample = {"img": img,
                "filename": str(fname), 
                "insect_idx": insect_idx, 
                "location": location, 
                "date": date, 
                "xtra": xtra,
                "width": width,
                "height": height}

        if self.transform:
            sample["img"] = self.transform(sample["img"])

        return tuple(sample.values())


    def plot_samples(self, df=pd.DataFrame(), noaxis=True, title='label'):
        import matplotlib.pyplot as plt
        if not len(df):
            df = self.df.sample(5, replace=False, random_state=seed).reset_index(drop=True)
        else:
            df = df.sample(5, replace=False, random_state=seed).reset_index(drop=True)

        plt.figure(figsize=(20,12))

        for i in tqdm(range(5)):
            plt.subplot(2,3,i+1)
            img = read_image(df.loc[i].full_filename)
            plt.imshow(img);
            if title == 'label':
                plt.title(df.loc[i].insect_idx)
            if noaxis:
                plt.axis('off')


def model_selector(modelname, pretrained=False):
    if modelname == 'densenet121':
        from torchvision.models import densenet121
        return densenet121(pretrained=pretrained)
    elif modelname == 'densenet169':
        from torchvision.models import densenet169
        return densenet169(pretrained=pretrained)
    elif modelname == 'mobilenetv2':
        from torchvision.models import mobilenet_v2
        return mobilenet_v2(pretrained=pretrained)
    elif modelname == 'vgg16':
        from torchvision.models import vgg16
        return vgg16(pretrained=pretrained)
    elif modelname == 'vgg19':
        from torchvision.models import vgg19
        return vgg19(pretrained=pretrained)
    elif modelname == 'efficientnetb0':
        from torchvision.models import efficientnet_b0
        return efficientnet_b0(pretrained=pretrained)
    elif modelname == 'efficientnetb1':
        from torchvision.models import efficientnet_b1
        return efficientnet_b1(pretrained=pretrained)
    elif modelname == 'resnet101':
        from torchvision.models import resnet101
        return resnet101(pretrained=pretrained)
    elif modelname == 'resnet50':
        from torchvision.models import resnet50
        return resnet50(pretrained=pretrained)
    else: 
        raise ValueError("No model returned")


def save_checkpoint(state, is_best, filename=''):
    import torch
    from shutil import copyfile
    filename = f'{SAVE_DIR}/{filename}.pth.tar'
    torch.save(state, filename)
    if is_best:
        copyfile(filename, f"{filename.split('.')[0]}_best.pth.tar")

def load_checkpoint(filename, model, optimizer):
    import torch
    assert isinstance(filename, str) and filename.endswith('pth.tar'), "Only works with a pth.tar file."
    checkpoint = torch.load(filename, map_location=torch.device('cpu'))
    
    model.load_state_dict(checkpoint['state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer

def get_files(directory, ext='.jpg'):
    return pd.Series([i for i in os.listdir(directory) if i.endswith('png')])

def read_image(filename, plot=False):
    img = Image.open(filename)
    return img    
