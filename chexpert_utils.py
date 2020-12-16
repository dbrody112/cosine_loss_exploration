
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

def divide_by_view(df,view):
    indexes = []
    for i in range(len(df)):
        if(re.search(view,np.array(df['Path'])[i])==None):
            print(i)
            indexes.append(i)
    return indexes

def saveCleanedLabels(minVal=100, maxVal=500, view ="",num_classes = 40):
    df = 0
    try:
        df = pd.read_csv("./CheXpert-v1.0-small/CheXpert-v1.0-small/train.csv") 
    except FileNotFoundError as e:
        print("enter the correct params")
        return
    df['Support Devices'] = 0
    df['No Finding'] = 0
    #view is either frontal or lateral
    
    if(view!=""):
        indexes = divide_by_view(df=df,view=view)
        print(indexes)
        df.drop(indexes,inplace=True)
    df["id"] = range(len(df))
    df = df.reset_index()
    dictionary = {}
    counts = {}
    ids = {}
    j=0
    for k in range(len(df)):
        new_dict = {}
        for i in df.columns:
            if(df[i][k]==1):
                new_dict[i] = 1
        if(len(new_dict)!=0):
            if new_dict not in dictionary.values():
                dictionary[j] = new_dict
                counts[j] = 1
                ids[k] = [df["id"][k],j]
                j+=1
            else:
                for i in range(len(dictionary)):
                    if(dictionary[i]==new_dict):
                        counts[i]+=1
                        ids[k] = [df["id"][k],i]
    sorted_dict = {k: v for k, v in sorted(counts.items(), key=lambda item: item[1],reverse = True)}
    keys = []
    for i in sorted_dict.keys():
        keys.append(i)
    vals=[]
    for i in sorted_dict.values():
        vals.append(i)
    count_df= pd.DataFrame({'keys':keys, 'counts':vals})
    keys = []
    for i in dictionary.keys():
        keys.append(i)
    vals=[]
    for i in dictionary.values():
        vals.append(i)
    class_df= pd.DataFrame({'classes':keys, 'vals':vals})
    merge_df=class_df.merge(count_df,left_on = "classes",right_on = "keys")
    merge_df = merge_df.drop('keys',axis=1)
    idss = []
    for i in ids.keys():
        idss.append(ids[i][0])
    
    classes=[]
    for i in ids.keys():
        classes.append(ids[i][1])
    id_df = pd.DataFrame({'id':idss, 'classes':classes})
    merged_df = merge_df.merge(id_df, left_on = "classes", right_on = "classes")
    
    
    
    def return_classes(merged_df,minVal,maxVal,num_classes):
        x =[]
        y =[]
        classes_train = []
        arr = np.array(merged_df)
        classes_list = {}
        for i in range(len(merged_df['counts'])):
            class_dist = np.random.choice([0.1,0.9],p=[0.5,0.5])
            if(merged_df['counts'][i] < maxVal and merged_df['counts'][i] > minVal):
                if(merged_df['classes'][i] not in classes_train):
                    classes_train.append(merged_df['classes'][i])
                    classes_list[merged_df['classes'][i]] = np.array([0,merged_df['counts'][i]])
                if(len(classes_train)>num_classes): 
                    break
                if((classes_list[merged_df['classes'][i]][0] >= classes_list[merged_df['classes'][i]][1]//3)):
                    x.append(arr[i])
                    classes_list[merged_df['classes'][i]][0]+=1
                else:
                    y.append(arr[i])
                    classes_list[merged_df['classes'][i]][0]+=1
                print(len(classes_train))
                print(merged_df['classes'][i])
                
        return x,y
    print(merged_df['counts'])
    print(return_classes(merged_df,minVal,maxVal,num_classes))
    x,y = return_classes(merged_df,minVal,maxVal,num_classes)
    arr_x = np.array(x)
    arr_y = np.array(y)
    
    relation_df_x = pd.DataFrame({'class':arr_x[:,0],'categorical_class':arr_x[:,1], 'id':arr_x[:,3], "counts":arr_x[:,2]})
    relation_df_y = pd.DataFrame({'class':arr_y[:,0],'categorical_class':arr_y[:,1], 'id':arr_y[:,3], "counts":arr_y[:,2]})
    final_df_x = df.merge(relation_df_x, left_on = 'id',right_on = 'id')
    final_df_x.to_csv("cleaned_reduced_chexpert_labels_train"+view+".csv")
    final_df_y = df.merge(relation_df_y, left_on = 'id',right_on = 'id')
    final_df_y.to_csv("cleaned_reduced_chexpert_labels_test"+view+".csv")
    print(f'The length of the output is {len(arr_y)}.')
    return

def loadCSV(option = '',minVal = 100, maxVal = 500, num_classes = 40):
    if(option == ''):
        try:
            pd.read_csv("./cleaned_reduced_chexpert_labels_test.csv")
            pd.read_csv("./cleaned_reduced_chexpert_labels_train.csv")
        except FileNotFoundError as e:
            saveCleanedLabels(minVal = minVal, maxVal = maxVal, num_classes = num_classes)
        
    elif(option== '_frontal'):
        try:
            pd.read_csv("./cleaned_reduced_chexpert_labels_test_frontal.csv")
            pd.read_csv("./cleaned_reduced_chexpert_labels_train_frontal.csv")
        except FileNotFoundError:
            saveCleanedLabels(view='_frontal',minVal = minVal, maxVal = maxVal, num_classes = num_classes)
        
    elif(option=='_lateral'):
        try:
            pd.read_csv("./cleaned_reduced_chexpert_labels_test_lateral.csv")
            pd.read_csv("./cleaned_reduced_chexpert_labels_train_lateral.csv")
        except FileNotFoundError as e:
            saveCleanedLabels(view='_lateral',minVal = minVal, maxVal = maxVal, num_classes = num_classes)
    return
