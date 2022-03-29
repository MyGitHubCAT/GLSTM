#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.append(r'XXX')#引用自定义库所在的地址
import funcclass
import modelclass


# In[ ]:


from scipy.signal import savgol_filter
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import spektral
import math
import scipy.sparse as sp


# In[3]:


files = np.arange(201,220)
filepath =['XXXX' + str(i) + '.csv' for i in files]#文件地址
data = [np.array(pd.read_csv(i),dtype='float32') for i in filepath]
#dataarr=np.array(data,dtype='float32')


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


plt.figure(figsize=(24,8))
plt.plot(np.arange(len(flowdata)),flowdata_scalenormal [:,0])
plt.show()


# In[7]:


alldata = flowdata_scalenormal
trainsplit = int(len(flowdata_scalenormal)*6/7)
train_data = alldata[0:trainsplit,:]
val_data = alldata[trainsplit:-1,:]


# In[8]:


import networkx as nx
G = nx.Graph(name='G')
for i in files:
    G.add_node(i, name=i)
edges=[(201,202),(202,203),(203,204),(204,205),(205,206),(206,207),(207,208)
      ,(208,209),(209,210),(209,211),(211,212),(212,213),(213,214),(214,215)
      ,(215,216),(216,217),(217,218),(218,219)]
G.add_edges_from(edges)
plt.figure(figsize=(24,10))
nx.draw(G, with_labels=True,node_size=5000, font_weight='normal',font_size=30,node_color='yellow')
plt.show()



# In[9]:



row = np.array([0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,10,9,10,11,11,12,12,13,13,14,14,15,
               15,16,16,17,17,18,18],dtype='float32')
col = np.array([0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,8, 9,10,10,11,11,12,12,13,13,14,14,
               15,15,16,16,17,17,18],dtype='float32')
adata = np.ones(len(row))
a = sp.csr_matrix((adata,(row,col)),shape=(19,19),dtype='float32').toarray()
a = np.maximum(a,a.T).astype('float32')
a=sp.csr_matrix(a).toarray()


# In[71]:


a.astype(int)


# In[58]:


windata = funcclass.WindowData(12,12,12,train_data,val_data)


# In[59]:


train_model=modelclass.Train_PredictMachine(speeddim=19,ratio=0,a_j=a)


# In[ ]:


#modelclass.train_GLSTM(windata.train,40,train_model)
#train_model.save_weights(r'XXXX\model_weight\trainmodel_12_08\train_model')#保存的模型地址
#train_model.load_weights(r'XXXX\model_weight\trainmodel_12_08\train_model')


# In[60]:


prediction_model=modelclass.PredictionModel(19,12,train_model.encoder,train_model.decoder)


# In[61]:


#modelclass.train_model(windata.train,40,prediction_model)
#prediction_model.save_weights(r'XXXX\premodel_12_08\pre_model')#保存的模型地址
#prediction_model.load_weights(r'XXXX\premodel_12_00\pre_model')


# In[62]:


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


# In[63]:


prearr = predatabystep(val_data,prediction_model,12,12)
prelen = len(prearr)
labarr = val_data[12:12+prelen]
inparr = val_data[0:12]


# In[64]:


prearr = prearr*(flowdata_max-flowdata_min) + flowdata_min
labarr=labarr*(flowdata_max-flowdata_min) + flowdata_min
inparr=inparr*(flowdata_max-flowdata_min) + flowdata_min


# In[65]:


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
        ax.savefig(r'XXX\GLSTMpredata.jpg')


# In[66]:


plotpredata(inparr,labarr,prearr,True)


# In[67]:


print("MAE: {0}".format(funcclass.CalMAE(labarr,prearr,19)))
print("RMSE: {0}".format(math.sqrt(funcclass.CalMSE(labarr,prearr,19))))
print("R2: {0}".format(funcclass.CalR2(labarr,prearr,19))) 


# In[ ]:




