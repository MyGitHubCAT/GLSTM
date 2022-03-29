from scipy.signal import savgol_filter
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import spektral
import scipy.sparse as sp
import time

class GGCN(tf.keras.layers.Layer):
    def __init__(self):
        super(GGCN,self).__init__()
        self.gc1 = spektral.layers.GCNConv(64,activation='relu',name='mygcn1')
        self.gc2 = spektral.layers.GCNConv(64,activation='relu',name='mygcn2')
        self.d1 = tf.keras.layers.Dense(32,activation='relu',name='gcndense')
        self.d2 = tf.keras.layers.Dense(1,name='gcndense')
    @tf.function
    def call(self,inputs):
        x,a = inputs        
        g1 = self.gc1([x,a])
        g2 = self.gc2([g1,a])
        d3 = self.d1(g2)
        res = self.d2(d3)
        return res

#GCN和数据分开用LSTM分析
class GLSTMEncoder(tf.keras.layers.Layer):
    def __init__(self,aj,ratio=0.8):
        super(GLSTMEncoder, self).__init__()
        self.ratio=ratio
        
        self.GCN= GGCN()    
        self.ajmatrix = aj
        self.LSTM1=tf.keras.layers.LSTM(64,recurrent_activation = 'sigmoid',
                                        return_sequences=True,return_state = True,name='mylstm1')
        self.LSTM2=tf.keras.layers.LSTM(64,recurrent_activation = 'sigmoid',
                                        return_state = True,name='mylstm1')
        
    #由于需要计算梯度，因此应该避免使用复杂结构的代码，尽量在call内完成计算，可以考虑将数据处理工作放置在训练数据格式上。
    
    def call(self,inps):#实际输入维度包含了打包量batch需要将打包量拆分,设计输入为[batchdata,aj_matrix]
        inputs =inps 
        gcn_output =[]
        for onebatch in inputs:
            oneGCN = []
            for x in onebatch:                
                x_adddim = tf.expand_dims(x,axis=-1)
                #print(x_adddim)
                gcn_res = self.GCN([x_adddim,self.ajmatrix])               
                oneGCN.append(tf.reduce_sum(gcn_res,axis=-1))
            oneGCNINP = tf.stack(oneGCN,axis=0)
            #oneGCNRES =self.GRU_for_GCN(oneGCNINP)
            gcn_output.append(oneGCNINP)
        GCNGRUINP = tf.stack(gcn_output,axis=0)
        #print(GCNGRUINP.shape)
       # print(inputs.shape)
        com_inp = (1-self.ratio)*GCNGRUINP+self.ratio*inputs
        
        seq1,mstate1,cstate1 = self.LSTM1(com_inp)
        seq2,mstate2,cstate2 = self.LSTM2(seq1)
        #gruconcat = tf.concat([gru_res,gcngru_res],axis=-1)
        #d1 = self.dense1(gruconcat)
        #res = self.dense2(d1)
        #res = self.reshape(res)
        return seq2,mstate1,cstate1,mstate2,cstate2#res 

class GLSTMDecoder(tf.keras.layers.Layer):    
    def __init__(self,speeddim):
        super(GLSTMDecoder,self).__init__()       
        self.speeddim=speeddim
        self.LSTM1=tf.keras.layers.LSTM(64,return_sequences=True,return_state=True,
                                      name='DecoderLSTM',recurrent_activation='sigmoid')
        self.LSTM2=tf.keras.layers.LSTM(64,return_sequences=True,return_state=True,
                                      name='DecoderLSTM',recurrent_activation='sigmoid')
        self.dense1 = tf.keras.layers.Dense(self.speeddim)
    @tf.function
    def call(self,inputs,mstate1,cstate1,mstate2,cstate2):
        
        l1,dmstate1,dcstate1 = self.LSTM1(inputs,initial_state=[mstate1,cstate1])
        l2,dmstate2,dcstate2 = self.LSTM2(l1,initial_state=[mstate2,cstate2])
        d2=self.dense1(l2)
        return d2,dmstate1,dcstate1,dmstate2,dcstate2 

class Train_PredictMachine(tf.keras.Model):
    def __init__(self,speeddim,a_j,ratio=0.8):
        super(Train_PredictMachine,self).__init__()
        self.speeddim = speeddim
        self.encoder = GLSTMEncoder(ratio=0.8,aj=a_j)
        self.decoder = GLSTMDecoder(speeddim)
    
    def call(self,inputs):
        inpx,laby=inputs
        batchdim = len(inpx)
        starttag = tf.ones([batchdim,1,self.speeddim])
        inp_with_starttag=tf.concat([starttag,laby],axis=1)
        seq,m1,c1,m2,c2=self.encoder(inpx)
        res,_,_,_,_=self.decoder(inp_with_starttag,m1,c1,m2,c2)
        res=res[:,:-1,:]
        return res

class PredictionModel(tf.keras.Model):
    def __init__(self,speeddim,prestep,encoder,decoder):
        super(PredictionModel,self).__init__()
        self.speeddim = speeddim
        self.prestep = prestep
        self.encoder = encoder
        self.encoder.trainable = False#解码器不再参与训练
        self.decoder =decoder
        self.decoder.trainable = False#编码器不再参与训练
        self.flatlayer = tf.keras.layers.Flatten()#最好平铺，否则只针对每个点的数据单独进行处理，无法实现误差均分的效果
        self.aveDense= tf.keras.layers.Dense(64,activation='relu')#数据预测中，存在明显的误差累积情况，需要进行误差均分
        self.preDense= tf.keras.layers.Dense(speeddim*prestep)
        self.reshape= tf.keras.layers.Reshape((prestep,speeddim))	
    
    def call(self,inputs):
        batchdim = len(inputs)
        starttag=np.ones((batchdim,1,self.speeddim))
        decinp= starttag
        pre_res=[]
        seq,m1,c1,m2,c2=self.encoder(inputs,training=False)
        for i in range(self.prestep):                   
            decinp,m1,c1,m2,c2=self.decoder(decinp,m1,c1,m2,c2,training=False)
            pre_res.append(decinp)
        pre_res = tf.concat(pre_res,axis=1)
        flatten_pre_res = self.flatlayer(pre_res)
        ave_error_res=self.aveDense(flatten_pre_res)
        pre_ave_error_res=self.preDense(ave_error_res)
        reshape_res = self.reshape(pre_ave_error_res)
        return reshape_res 

#训练模型训练
def train_GLSTM_step(inp,model,step,steptime,lossfunc,opt):
    _,y_label=inp
    with tf.GradientTape() as tape:
        logit = model(inp)
        loss = lossfunc(y_label,logit)
    grads= tape.gradient(loss,model.trainable_weights)
    opt.apply_gradients(zip(grads,model.trainable_weights))
    if step % 50 == 0:
        print("Time {0} . Training loss (for one batch) at step {1}: {2}".format(steptime,step, float(loss)))
        
def train_GLSTM(dataset,epochs,model):
    loss = tf.keras.losses.MeanSquaredError()
    optimazer = tf.keras.optimizers.Adam()
    for epoch in range(epochs+1):
        start=time.time()
        print("\n开始第%d论训练"%(epoch))
        for step,(x_batch,y_batch) in enumerate(dataset):
            #print(x_batch.shape)
            train_GLSTM_step((x_batch,y_batch),model,step,time.time()-start,loss,optimazer)           

#普通模型训练
def train_step(inp,y_label,model,step,steptime,lossfunc,opt):   
    with tf.GradientTape() as tape:
        logit = model(inp)
        loss = lossfunc(y_label,logit)
    grads= tape.gradient(loss,model.trainable_weights)
    opt.apply_gradients(zip(grads,model.trainable_weights))   
    if step % 50 == 0:
        print("Time {0} . Training loss (for one batch) at step {1}: {2}".format(steptime,step, float(loss)))  
      
def train_model(dataset,epochs,model):
    loss = loss = tf.keras.losses.MeanSquaredError()
    optimazer = tf.keras.optimizers.Adam()
    for epoch in range(epochs+1):
        start=time.time()
        print("\n开始第%d论训练"%(epoch))
        for step,(x_batch,y_batch) in enumerate(dataset):
            #print(x_batch.shape)
            train_step(x_batch,y_batch,model,step,time.time()-start,loss,optimazer)        