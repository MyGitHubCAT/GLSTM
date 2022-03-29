from scipy.signal import savgol_filter
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import spektral
import math

import scipy.sparse as sp
class WindowData():
    def __init__(self,ipwidth,labelwidth,offset,train_df,val_df):
        self.train_df=train_df
        self.val_df = val_df
        
        
        self.input_width=ipwidth
        self.label_width = labelwidth
        self.offset = offset
        self.total_width =  self.input_width + self.offset
        
        self.input_slice = slice(0,ipwidth)
        self.label_slice = slice(self.total_width-labelwidth,self.total_width)
        
        allindex = np.arange(self.total_width)
        self.input_index = allindex[self.input_slice]
        self.label_index=allindex[self.label_slice]
        
    def ShowMessage(self):
        print(self.input_index,self.label_index)
        exa_inp,exa_lab=next(iter(t1.train))
        print('input data: \n{}\nlabel data: \n{}'.format(exa_inp[0],exa_lab[0]))
    def split_window(self,seq_data):
        input_data = seq_data[:,self.input_slice,:]
        label_data=seq_data[:,self.label_slice,:]
        
        input_data.set_shape([None, self.input_width, None])
        label_data.set_shape([None, self.label_width, None])
        return input_data,label_data
    def make_dataset(self,data):
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(data=data,
            targets=None,sequence_length=self.total_width,sequence_stride=1,shuffle=True,batch_size=3)
        ds=ds.map(self.split_window)
        return ds
    @property
    def train(self):
        return self.make_dataset(self.train_df)
    @property
    def val(self):
        return self.make_dataset(self.val_df)


def predatabystep(inp,premodel,inpwidth,prewidth):
    datalen=len(inp)
    step = int((datalen-inpwidth)/prewidth)
    inpdata =[]
    for i in range(step):
        inpdata.append(inp[i*prewidth:i*prewidth+inpwidth])
    inpdata = np.array(inpdata)    
    predata = premodel(inpdata).numpy()
    prearr = predata[0]
    for i in range(step-1):
        prearr = np.concatenate((prearr,predata[i+1]),axis=0)
    return prearr


def CalMAE(arr1,comparr2,dim):#2维列表间求空间方向R2
    nums = len(arr1)
    if(len(comparr2)!=nums):
        return 0  
    dis = 0
    for i in range(nums):
        for j in range(dim):
            dis =dis+ abs(arr1[i,j]-comparr2[i,j])
    mae = dis/(nums*dim)
    return mae

def CalMSE(arr1,comparr2,dim):#2维列表间求空间方向R2
    nums = len(arr1)
    if(len(comparr2)!=nums):
        return 0  
    dis = 0
    for i in range(nums):
        for j in range(dim):
            dis =dis+ math.pow(arr1[i,j]-comparr2[i,j],2)
    mse = dis/(nums*dim)
    return mse

def CaldimMAE(realarr,prearr,num):
    nums = num
    loss_MAE = tf.keras.losses.MeanAbsoluteError()
    value_mae_arr =np.array([loss_MAE(realarr[i],prearr[i]).numpy() for i in range(nums)])
    return value_mae_arr,value_mae_arr.mean()


def CaldimMSE(realarr,prearr,num):
    nums = num
    loss_MSE = tf.keras.losses.MeanSquaredError()
    value_mse_arr =np.array([loss_MSE(realarr[i],prearr[i]).numpy() for i in range(nums)])
    return value_mse_arr,value_mse_arr.mean()



def CalR2(arr1,comparr2,dim):#2维列表间求R2
    nums = len(arr1)
    if(len(comparr2)!=nums):
        return 1
    arr1mean = arr1.mean()
    dis1=0
    dis2=0
    for i in range(nums):
        for j in range(dim):
            dis1 = dis1+math.pow(arr1[i,j]-comparr2[i,j],2)
            dis2 = dis2+math.pow(arr1[i,j]-arr1mean,2)
    r2 = 1-(dis1/dis2)
    return r2