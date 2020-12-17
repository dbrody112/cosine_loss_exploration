

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

class CategoricalCrossEntropyLoss(nn.Module):
    def __init__(self,loss_logger = None):
        super(CategoricalCrossEntropyLoss, self).__init__()
        self.loss_logger = loss_logger

    def forward(self, y_hat, y):
        if(self.loss_logger==None):
            return nn.NLLLoss(reduction = "none")(y_hat, torch.argmax(y,dim=1))
        else:
            return nn.NLLLoss(reduction="none")(y_hat, torch.argmax(y,dim=1)), self.loss_logger
    
    
class cosine_OHEM(nn.Module):
    def __init__(self,ratio,lmbda,loss_logger,weights=None):
        super(cosine_OHEM, self).__init__()
        self.ratio = ratio
        self.lmbda = lmbda
        self.loss_logger = loss_logger
        self.weights = weights
        
        
    def forward(self,y_hat,y):
        y_shape = y.shape[0]
        indexes = 0
        #print(torch.log(y_hat))
        
        topk_loss = nn.NLLLoss(reduction ="none")(y_hat, torch.argmax(y,dim=1))+self.lmbda*(1-torch.sum(y_hat*y,1))
        _,indexes = torch.topk(topk_loss,int(y_shape*self.ratio))
        #print(f'indexes:{indexes}')
        #print(f'y:{y}')
        new_pred = torch.index_select(y_hat,0,indexes)
        self.loss_logger.append([topk_loss,1-torch.sum(y_hat*y,axis=-1)])
        new_train = torch.index_select(y,0,indexes)
        #print(f'new_pred_log: {new_pred}')
        #print(f'new_train_argmax: {torch.argmax(new_train,dim=1)}')
        if(self.weights==None):
            return nn.NLLLoss()(new_pred, torch.argmax(new_train,dim=1)), self.loss_logger
        else:
            return nn.NLLLoss(weight = self.weights)(new_pred, torch.argmax(new_train,dim=1)), self.loss_logger
        


        
class cosine_specificity_OHEM(nn.Module):
    def __init__(self,ratio,lmbda,loss_logger,weights):
        super(cosine_specificity_OHEM, self).__init__()
        self.ratio = ratio
        self.lmbda = lmbda
        self.loss_logger = loss_logger
        self.weights = weights
        
    def forward(self,y_hat,y):
        y_shape = y.shape[0]
        predicted = torch.max(y_hat.data, 1)[1]
        y_train_changed = torch.argmax(y,dim=1)
        sensitivity,_,_ = sensitivity_specificity_support(np.array(predicted.cpu().detach().numpy()),np.array(y_train_changed.cpu().detach().numpy()))
        topk_loss = CategoricalCrossEntropyLoss()(y_hat,y)+self.lmbda*(1-torch.sum(y_hat*y,axis=-1))-sensitivity
        _,indexes = torch.topk(topk_loss,int(y_shape*self.ratio))
        new_pred = torch.index_select(y_hat,0,indexes)
        self.loss_logger.append(topk_loss)
        new_train = torch.index_select(y,0,indexes)
        return nn.NLLLoss(weight = self.weights)(torch.log(new_pred), torch.argmax(new_train,dim=1)), self.loss_logger
    
class specificity_OHEM(nn.Module):
    def __init__(self,ratio,lmbda,loss_logger, weights):
        super(specificity_OHEM, self).__init__()
        self.ratio = ratio
        self.lmbda = lmbda
        self.loss_logger = loss_logger
        self.weights = weights
    def forward(self,y_hat,y):
        y_shape = y.shape[0]
        predicted = torch.max(y_hat.data, 1)[1]
        y_train_changed = torch.argmax(y,dim=1)
        sensitivity,_,_ = sensitivity_specificity_support(np.array(predicted.cpu().detach().numpy()),np.array(y_train_changed.cpu().detach().numpy()))
        topk_loss = CategoricalCrossEntropyLoss()(y_hat,y)-sensitivity
        _,indexes = torch.topk(topk_loss,y_shape-self.topk)
        new_pred = torch.index_select(y_hat,0,indexes)
        self.loss_logger.append(new_pred)
        new_train = torch.index_select(y,0,indexes)
        print(new_train.shape)
        return nn.NLLLoss()(new_pred, torch.argmax(new_train,dim=1)), self.loss_logger
