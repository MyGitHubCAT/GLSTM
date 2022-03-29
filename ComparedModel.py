#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.signal import savgol_filter
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import spektral
import math
import scipy.sparse as sp


# In[2]:


import sys
sys.path.append(r'XXXXX')
import funcclass
import modelclass


# In[3]:


files = np.arange(201,220)
filepath =['D://sjtu//博1//课程//人工智能//' + str(i) + '.csv' for i in files]
data = [np.array(pd.read_csv(i),dtype='float32') for i in filepath]


# In[4]:


flowdata_ = np.array([i[:,0] for i in data ])
flowdata_ = flowdata_.transpose(1,0)

speeddata_ = np.array([i[:,1] for i in data ])
speeddata_ =speeddata_.transpose(1,0)

flowdata_savgol = np.array([savgol_filter(i[:,0],5,3) for i in data])
flowdata_savgol = flowdata_savgol.transpose(1,0)


# In[5]:


#缩放归一化
flowdata = flowdata_savgol
flowdata_max = flowdata.max(axis=0)
flowdata_min = flowdata.min(axis=0)
flowdata_scalenormal = (flowdata-flowdata_min)/(flowdata_max-flowdata_min)


# In[6]:


alldata = flowdata_scalenormal
trainsplit = int(len(flowdata_scalenormal)*6/7)
train_data = alldata[0:trainsplit,:]
val_data = alldata[trainsplit:-1,:]


# In[7]:


windata = funcclass.WindowData(12,12,12,train_data,val_data)


# In[8]:


winx,_ = next(iter(windata.train))
winx


# In[42]:


cnn_Input=tf.keras.Input(shape=(12,19))
cnn_InputT = tf.transpose(cnn_Input,[0,2,1])
cnn_c1=tf.keras.layers.Conv1D(12,3,activation='relu',strides=3)(cnn_InputT)
cnn_c1T = tf.transpose(cnn_c1,[0,2,1])
cnn_flat=tf.keras.layers.Flatten()(cnn_c1T)
cnn_d2=tf.keras.layers.Dense(32)(cnn_flat)
cnn_d3 =tf.keras.layers.Dense(12*19)(cnn_d2)
cnn_res = tf.keras.layers.Reshape((12,19))(cnn_d3)
CNNmodel= tf.keras.Model(inputs=cnn_Input,outputs=cnn_res)
CNNmodel.summary()


# In[43]:


CNNmodel.compile(optimizer='adam',loss='mse',metrics=tf.keras.metrics.MeanSquaredError())
CNNmodel.fit(windata.train,epochs=50,validation_data=windata.val)


# In[11]:


lstm_Input=tf.keras.Input(shape=(12,19))
lstm_l1 = tf.keras.layers.LSTM(64)(lstm_Input)
lstm_d2= tf.keras.layers.Dense(64,activation='relu')(lstm_l1)
lstm_d3 = tf.keras.layers.Dense(12*19)(lstm_d2)
lstm_res=tf.keras.layers.Reshape((12,19))(lstm_d3)
LSTMmodel= tf.keras.Model(inputs=lstm_Input,outputs=lstm_res)
LSTMmodel.summary()


# In[12]:


LSTMmodel.compile(optimizer='adam',loss='mse',metrics=tf.keras.metrics.MeanSquaredError())
LSTMmodel.fit(windata.train,epochs=50,validation_data=windata.val)


# In[13]:


cnnlstm_Input=tf.keras.Input(shape=(12,19))
cnnlstm_InputT = tf.transpose(cnnlstm_Input,[0,2,1])
cnnlstm_c1=tf.keras.layers.Conv1D(12,3,strides=3)(cnnlstm_InputT)
cnnlstm_c1T = tf.transpose(cnnlstm_c1,[0,2,1])
cnnlstm_l2 = tf.keras.layers.LSTM(64)(cnnlstm_c1T)
cnnlstm_d3= tf.keras.layers.Dense(64,activation='relu')(cnnlstm_l2)
cnnlstm_d4 = tf.keras.layers.Dense(12*19)(cnnlstm_d3)
cnnlstm_res= tf.keras.layers.Reshape((12,19))(cnnlstm_d4)
CNNLSTMmodel= tf.keras.Model(inputs=cnnlstm_Input,outputs=cnnlstm_res)
CNNLSTMmodel.summary()


# In[14]:


CNNLSTMmodel.compile(optimizer='adam',loss='mse',metrics=tf.keras.metrics.MeanSquaredError())
CNNLSTMmodel.fit(windata.train,epochs=50,validation_data=windata.val)


# In[15]:


Baseline_Input=tf.keras.Input(shape=(12,19))
Baselinemodel= tf.keras.Model(inputs=Baseline_Input,outputs=Baseline_Input)


# In[17]:


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


# In[20]:


def plotpredata(inp,lab,pre,issave=False):
    inp_length = len(inp)
    lab_length = len(lab)
    maxindex = np.arange(inp_length+lab_length)
    inpindex = maxindex[slice(0,inp_length)]
    labindex = maxindex[slice(inp_length,None)]
    
    pltdim = inp.shape[1]
    
    ax = plt.figure(figsize=(24,8*pltdim))
    for i in range(pltdim):
        plt.subplot(pltdim,1,i+1)        
        plt.xlabel("Time Index",fontsize=24)
        plt.ylabel("Flow",fontsize=24)
        plt.xticks(size=24)
        plt.yticks(size=24)
        plt.plot(inpindex,inp[:,i],color='r')
        plt.plot(labindex,lab[:,i],color='b')   
        plt.plot(labindex,pre[:,i],color='g')    
    plt.show()
    if issave:
        ax.savefig(r'D:\sjtu\博1\课程\人工智能\GLSTMpredata.jpg')


# In[49]:


premodel = LSTMmodel
prearr = predatabystep(val_data,premodel ,12,12)
prelen = len(prearr)
labarr = val_data[12:12+prelen]
inparr = val_data[0:12]

prearr = prearr*(flowdata_max-flowdata_min) + flowdata_min
labarr=labarr*(flowdata_max-flowdata_min) + flowdata_min
inparr=inparr*(flowdata_max-flowdata_min) + flowdata_min


# In[50]:


print("MAE: {0}".format(funcclass.CalMAE(labarr,prearr,19)))
print("RMSE: {0}".format(math.sqrt(funcclass.CalMSE(labarr,prearr,19))))
print("R2: {0}".format(funcclass.CalR2(labarr,prearr,19)))


# In[48]:


plotpredata(inparr,labarr,prearr,True)


# In[ ]:




