

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
from models import resnet18
from flowers import loadFlowerDataset
from chexpert import loadChexpert


checkpoint = "./cosine_ohem_0.9ratio_affine_-0.2_40.pt"
checkpoint = torch.load(checkpoint)

dataset = "chexpert"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train = []
test = []
train_csv = []
test_csv = []
if(dataset == "chexpert"):
  train, test,train_csv, test_csv = loadChexpert()
elif(dataset=="flowers"):
  train,test, train_csv, test_csv = loadFlowerDataset()



class_length, model = resnet18(dataset,train_csv)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

test_load_all = DataLoader(test, batch_size = 100, shuffle=True)

with torch.no_grad():
    correct = 0
    print(1)
    for b,(X_test, y_test) in enumerate(test_load_all):
        
        y_val = model(X_test)
        y_val = F.softmax(y_val,dim=1)
        predicted = torch.max(y_val.data,1)[1]
        correct += (predicted == torch.argmax(torch.reshape(y_test.long(),(-1,class_length)),dim=1)).sum()
        print(correct)
    print(f'percent correct:{(correct/len(test))*100}')
    
train_losses = checkpoint['train_losses']
test_losses = checkpoint['test_losses']

def plotCorrectandLoss(epochs,title_correct, title_losses,train_losses,test_losses):
  """
      epochs(int) : # of epochs trained
      title_correct (string) : title o
      title_losses (string) : title of second plot against training and testing losses """
  
  plt.plot(np.array(train_losses)[1]/len(train), label='training correct')
  plt.plot(range(0,epochs,3),np.array(test_losses)[1]/len(test),label = 'testing correct')
  plt.legend()
  plt.title(title_correct)
  plt.ylabel('percent samples correct')
  plt.xlabel('epochs')
  
  plt.plot(np.array(train_losses)[0], label='training loss')
  plt.plot(range(0,epochs,3),np.array(test_losses)[0],label = 'testing loss')
  plt.legend()
  plt.title(title_losses)
  plt.ylabel('loss')
  plt.xlabel('epochs')
  
"""testing softmax vals for first 1000 batches and those for last 1000"""
"""can take out if you feel it is not necessary"""
hist = checkpoint['hist']

predicted = [torch.max(np.array(hist)[:,0][i].data,1)[1] for i in range(100)]
third = [np.array(predicted)[i][3] for i in range(len(predicted))]
plt.plot(third)
plt.title('cosine ohem on affine transformed chexpert with ratio 0.9 and lambda 0 confidence (first 100 batches)')
plt.xlabel('batch (each epoch is approximately 70 batches)')
plt.ylabel('softmax value')

predicted = [torch.max(np.array(hist)[:,0][i].data,1)[1] for i in range(len(hist)-100,len(hist))]
third = [np.array(predicted)[i][3] for i in range(len(predicted))]
plt.plot(third)
plt.title('cosine ohem on affine transformed chexpert with ratio 0.9 and lambda 0 confidence (last 100 batches)')
plt.xlabel('batch (each epoch is approximately 70 batches)')
plt.ylabel('softmax value')

