import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
from PIL import Image
from scipy.ndimage import zoom
import regex as re
import os
import sys
from sklearn.preprocessing import OneHotEncoder
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from imblearn.metrics import sensitivity_specificity_support
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torchvision.models as models
import scipy.io as sio
from chexpert_utils import loadCSV

def showImage(img):
    plt.imshow(transforms.functional.to_pil_image(img))

class Chexpert(Dataset): 
    def __init__(self, csv_file, train_csv, file_paths,transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomAffine(degrees = 30,translate = (0.2,0.2)),
    transforms.ToTensor(),
])):
        """
        Args:
            csv_file (pd.DataFrame):csv file with annotations.
            train_csv(pd.DataFrame): train csv file with annotations
            file_paths (string): File paths of all the images with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_file = csv_file
        self.train_csv = train_csv
        self.file_paths = file_paths
        self.transform = transform
        

    def __len__(self):
        return(len(self.csv_file))

    def __getitem__(self, idx):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img = Image.open(self.file_paths[idx])
        img = np.array(img)
        img = transforms.functional.to_pil_image(img)
        if self.transform:
            img = self.transform(img)
        
        unique_classes = [i+1 for i in range(len(self.train_csv['class'].unique()))]
        class_dict = {}
        for i in range(len(self.train_csv['class'].unique())):
            z = torch.zeros(1,max(unique_classes))
            z[:,i] = 1
            class_dict[self.train_csv['class'].unique()[i]] = z
        label = torch.FloatTensor(class_dict[self.csv_file['class'][idx]])
        img.to(device)
        label.to(device)
        sample = [torch.FloatTensor(img).to(device),label.to(device)]

        return sample

def loadChexpert(train_csv_file_path= "./cleaned_reduced_chexpert_labels_train", test_csv_file_path ="./cleaned_reduced_chexpert_labels_test",option='',minVal=100,maxVal=500,num_classes=40):
    loadCSV(option,minVal,maxVal,num_classes)
    train_csv = pd.read_csv(train_csv_file_path+option+".csv")
    test_csv = pd.read_csv(test_csv_file_path+option+".csv")
    
    test_paths = []
    train_paths=[]
    i = 0
    for patient in glob.glob('./CheXpert-v1.0-small/CheXpert-v1.0-small/train/*'):
        for study in glob.glob(patient+'/*'):
            for image_name in glob.glob(study+'/*'):
                if(i in np.array(train_csv['id'])):
                    train_paths.append(image_name)
                i+=1

            
    i = 0
    for patient in glob.glob('./CheXpert-v1.0-small/CheXpert-v1.0-small/train/*'):
        for study in glob.glob(patient+'/*'):
            for image_name in glob.glob(study+'/*'):
                if(i in np.array(test_csv['id'])):
                    test_paths.append(image_name)
                i+=1
    print(len(train_paths))
    train = Chexpert(csv_file = train_csv,train_csv = train_csv, file_paths = train_paths)  
    test = Chexpert(csv_file = test_csv,train_csv = train_csv, file_paths = test_paths)
    return train,test,train_csv, test_csv

