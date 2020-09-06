from multiprocessing import Process, cpu_count, Manager, Queue
import numpy as np
from time import sleep
import psutil
import tensorflow as tf
import tensorflow_probability as tfp
from keras.backend import clear_session
import os
import seaborn as sns
import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.pyplot import figure
tfd = tfp.distributions



def sleepIfRAMFULL(threshold=75,time_sleep=5):
    ram_percent = psutil.virtual_memory().percent
    if (ram_percent > threshold):
        print("MainThread RAM-usage: {:.2f} .. going to sleep for {}s".format(ram_percent,time_sleep),end="\r")
        sleep(time_sleep)
        return True

    return False
    

def dist2Classes(p,threshold = 0.5):
    # Wahrscheinlichkeit für kein Regen
    
    rain = 1 - np.array(p.prob(0))
    if np.isnan(rain).any():
        print(np.isnan(rain).sum())
    
    
    # Wahrscheinlichkeit Regen größer threshold, 
    mask = (rain > threshold)
    

    return mask

def labelToMask(label):
    # Gibt Maske mit Regen zurück
    return (label > 0)


def get_TP(simple,y,pred,threshold=0.5):

    
    true_pos = np.zeros(y.shape)
    true_neg = true_pos.copy()
    false_pos = true_pos.copy()
    false_neg = true_pos.copy()
    rain_total = 0
    total = 0

    simple_true_pos =  true_pos.copy()
    simple_true_neg =  true_pos.copy()
    simple_false_pos = true_pos.copy()
    simple_false_neg = true_pos.copy()

            
    pred_mask = dist2Classes(pred,threshold=threshold)

    labl_mask = labelToMask(y)

    #              Kein Regen Predicted & Label = Kein Regen
    true_pos   += (pred_mask == True)    & (labl_mask == True)
    #              Regen Predicted       & Label = Kein Regen
    false_neg  += (pred_mask == False)   & (labl_mask == True)
    #              Regen Predicted       & Label = Regen
    true_neg   += (pred_mask == False)   & (labl_mask == False)
    #              Kein Regen Predicted  & Label = Regen
    false_pos  += (pred_mask == True)    & (labl_mask == False)
    rain_total += (~labl_mask).sum()
    total += labl_mask.sum() + (~labl_mask).sum()
    
    

    # Simple Baseline
    # aus irgendeinem Grund können wir hier labelToMask nicht nutzen
    pred_simple = simple > 0

    #              Kein Regen Predicted & Label = Kein Regen
    simple_true_pos  += (pred_simple == True)    & (labl_mask == True)
    #              Regen Predicted       & Label = Kein Regen
    simple_false_neg += (pred_simple == False)   & (labl_mask == True)
    #              Regen Predicted       & Label = Regen
    simple_true_neg  += (pred_simple == False)   & (labl_mask == False)
    #              Kein Regen Predicted  & Label = Regen
    simple_false_pos += (pred_simple == True)    & (labl_mask == False)

    return_dict = {
            "TP"        : true_pos.sum(),
            "TN"        : true_neg.sum(),
            "FP"        : false_pos.sum(),
            "FN"        : false_neg.sum(),
            "TP_simple" : simple_true_pos.sum(),
            "TN_simple" : simple_true_neg.sum(),
            "FP_simple" : simple_false_pos.sum(),
            "FN_simple" : simple_false_neg.sum(),
            "total"     : total,
            "rain total":rain_total}


    return return_dict


def sum_up_keys(d):
    for key in d:
        d[key] = np.sum(d[key])
    return d

def add_FPTP_dicts(d1,d2):
    
    for key in d1:
        d1[key] = d1[key] + d2[key]

    return d1

def ZeroInflated_Binomial(t):

    return  tfp.distributions.Independent(
        tfd.Mixture(
            cat=tfd.Categorical(tf.stack([1-tf.math.sigmoid(t[...,:1]), tf.math.sigmoid(t[...,:1])],axis=-1)),
            components=[tfd.Deterministic(loc=tf.zeros_like(t[...,:1])),
            tfp.distributions.NegativeBinomial(
            total_count=tf.math.softplus(t[..., 1:2]), 
            logits=tf.math.sigmoid(t[..., 2:]) ),])
        ,name="ZeroInflated_Binomial",reinterpreted_batch_ndims=0 )

    
class Categorical(object):
    """docstring for ClassName"""
    def __init__(self,p):
        super(Categorical, self).__init__()
        self.p = p
        
    def prob(self,value):
        shape = self.p.shape
        if value < 0 or value >= shape[-1]:
            return np.zeros_like(self.p)
        a = self.p[:,:,:,value] 
        return a


def worker(procNbr, data_path,return_val,dist):
    

    
    processing = 0
    return_dict = {}

    data_x = data_path[0]
    data_y = data_path[1]
    data_p = data_path[2]
    
    threshold_list = np.arange(40+1)

    for i in threshold_list:
        return_dict[i] = None

    while (len(data_x) > 0) :

        x = np.array([data_x.pop()])
        y = np.array([data_y.pop()])
        p = np.array([data_p.pop()])
        
        p = dist(p)

        for key in threshold_list:
            fptp_dict = get_TP(x[0:,:,:,-1:],y,p,threshold = key/20)
            

            if return_dict[key] is None:
                return_dict[key] = fptp_dict
                continue
            else:
                return_dict[key] = add_FPTP_dicts(return_dict[key],fptp_dict)
        

        processing += 1
        del x
        del y
        del p

    
    del data_x 
    del data_p 
    del data_y
    
    print("Worker {:2d} processed {:6d} images".format(procNbr,processing),end="\r")
    return_val.put(return_dict)
    print("Worker {:2d} finished".format(procNbr))


def save(data,path):
    np.save(path,data,allow_pickle=True)

def load(path):
    return np.load(path,allow_pickle=True)

def multiProc_eval(model,test,getFreshSet,dist=ZeroInflated_Binomial,x_transform=[],y_transform=[]):

    nbrProcesses = cpu_count() * 2
    procCtr = 0
    return_dict = {}
    jobs = []
    finished = False
    shared_dict = {}
    
    train,test = getFreshSet(1)
    d_size = len(test) // nbrProcesses
    batch_size = 200
    data_x = []
    data_y = []
    data_p = []
    train,test = getFreshSet(batch_size)
    j=0
    l = len(test)


    confusionMat = None
    returnQueue = {}
    
    for i,(x,y) in enumerate(test):
        clear_session()
        
        with tf.device("/gpu:0"):
            p = model(x[:,:,:,:],training=False)
        
        
        with tf.device("/cpu:0"):           
            batch = x.shape[0]
            for i in range(batch_size):
                if x_transform:
                    new = x[i,:,:,:]
                    for t in x_transform:
                        new_x = t(x[i,:,:,:])
                    data_x.append(new_x)
                else:
                    data_x.append(x[i,:,:,:])
                data_y.append(y[i,:,:,:])
                data_p.append(p[i,:,:,:])
                
            if len(data_x) < d_size:
                continue
        

            
            data = [data_x,data_y,data_p]
            returnQueue[procCtr] = Queue()
        
            job = Process(name = "DeepRain Eval Process "+str(procCtr),
                          target = worker, 
                          args = (procCtr,data,returnQueue[procCtr],dist ))
            procCtr +=1
            job.start()
            jobs.append(job)
            data_x = []
            data_y = []
            data_p = []


        while (sleepIfRAMFULL()):
            continue


    def emptyQueue(returnQueue,confusionMat):
        for key in returnQueue:
            threshold_list = returnQueue[key].get()
            
            if confusionMat is None:
                confusionMat = threshold_list
                continue

            for k in threshold_list:
                confusionMat[k] = add_FPTP_dicts(confusionMat[k],threshold_list[k])
        return confusionMat


    confusionMat = emptyQueue(returnQueue,confusionMat)
    for job in jobs[j:]: 
        job.join()
    return confusionMat


def plotAUC(baseline_dict,lw=2):
    figure(num=None, figsize=(20, 30), dpi=100, facecolor='w', edgecolor='k')
    tp = []
    fp = []
    
    for key in baseline_dict:
        base = baseline_dict[key]
        tp.append(base["TP"]/(base["TP"]+base["FN"]) )
        fp.append(base["FP"]/(base["FP"]+base["FN"]))
    sns.set(style="ticks", context="talk")
    sns.set_style("darkgrid")
    #plt.style.use("dark_background")
    plt.figure(figsize=(20, 10),dpi=100)
    plt.plot(fp, tp, color='darkorange',
         lw=lw, label='ROC curve ')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()

def quantile(prediction,iv = (0,128),cfdi = 0.90):
    shape = prediction.sample().shape
    q = np.zeros((*shape[:-1],3),dtype=np.float32)
    a         = np.arange(iv[0],iv[1])
        
    prob = prediction.prob(a)
    cdf  = np.cumsum(prob,axis=-1)
    mean = prediction.mean()
 
    intervall = (cdf >= 1 - cfdi) & (cdf <=  cfdi)
    
    _,x,y,_ = shape
    
    x, y = np.meshgrid(np.arange(0,x),np.arange(0,y), sparse=False, indexing='ij')
    

    for i in range(shape[1]):
        for j in range(shape[2]):
            idcs   = np.where(intervall[0,i,j]==True)[0] 
            lower  = idcs[0]
            upper  = idcs[-1]
            q[0,i,j] = [lower,mean[0,i,j],upper]
            
            
    return q


def nBinom(t):
    
    return tfp.distributions.Independent(
                    tfp.distributions.NegativeBinomial(
                    total_count=tf.math.softplus(t[...,1:2]), \
                    logits=tf.math.sigmoid(t[...,2:]))
        ,name="ZeroInflated_Binomial",reinterpreted_batch_ndims=0)




def calculate_quantiles(model,test,max_j=30,dist=nBinom,cfdi=0.90):
    
    quantiles = []
    label     = []
    #test.on_epoch_end()
    batch_size = test[0][0].shape[0]
    length = len(test)
    for j,(x,y) in enumerate(test):
        
        if j >= max_j:
            return (quantiles,label)
        
        print("{}/{}".format(j,length),end="\r")
        for i in range(batch_size):
            pred = model(np.array([x[i,:,:,:]]))
            quantiles.append(quantile(dist(pred),cfdi=cfdi))
            label.append(y[i,:,:,:])
            
        
        
            
    return (quantiles,label)




def getDist(param,dist,iv=(0,255)):
    prediction = dist(param)
    prob = np.array(prediction.sample(255))
    prob = prob.transpose(1,2,3,4,0)
    
    return prob[:,:,:,0,:]
    

def joyplot(model,test,max_j = 2):
    y = test[0][1]
    x_,y_ = np.random.randint(low=0,high=y.shape[1],size=2)

    dists = []
    time_d  = []
    x_series = []

    for j,(x,y) in enumerate(test):
        
        if j >= max_j:
            break
        batch_size = x.shape[0]
        for i in range(batch_size):
            pred = model(np.array([x[i,:,:,:]]))
            d = getDist(pred[:,:,:,1:],nBinom)
            dists.append(np.array(d[0,x_,y_,:]))
            x_series.append(np.arange(0,255))
            
    import joypy
    dframe = pd.DataFrame(np.array(dists[:50]).T)
    fig, axes = joypy.joyplot(dframe,
                              figsize=(5,10),
                              linewidth=0.5,
                              grid='y',
                              background='k',
                              x_range=[1,50],
                              colormap=cm.rainbow)

"""
def plotCFDI(q,l):
    x,y = np.random.randint(low=0,high=q[0].shape[1],size=2)
    
    lower = []
    upper = []
    mean  = []
    label = []
    
    for i in range(len(q)):
        lower.append(q[i][0,x,y,0])
        mean.append(q[i][0,x,y,1])
        upper.append(q[i][0,x,y,2])
        label.append(l[i][x,y,0])

    
    sns.set(style="darkgrid")
    plt.figure(figsize=(20, 10),dpi=100)
    fig, ax = plt.subplots(1, 1,figsize=(20, 10),dpi=100, sharex=True)
    x = np.arange(len(q))
    ax.plot(x, lower,color='black',label="x >= 0.05",alpha = 0.5)
    ax.plot(x, upper, color='black',label="0.95 >=x",alpha = 0.5)
    ax.plot(x, label, color='red',label="True",alpha = 0.5)
    ax.plot(x, mean, color='blue',label="mean",alpha = 0.5)
    plt.legend(loc="upper right")
    ax.fill_between(x, lower, upper, where=upper >= lower, facecolor='green', alpha=0.1,interpolate=True,)
    ax.set_title('Konfidenzintervall')
    plt.ylim([0.0, np.max(label)])
    plt.xlim([0, len(q)])
"""
def plotCFDI(q,l):
    x,y = np.random.randint(low=0,high=q[0].shape[1],size=2)
    
    lower = []
    upper = []
    mean  = []
    label = []
    
    for i in range(len(q)):
        lower.append(q[i][0,x,y,0])
        mean.append(q[i][0,x,y,1])
        upper.append(q[i][0,x,y,2])
        label.append(l[i][x,y,0])
        
    x = np.arange(len(lower))
    
    sns.set(style="darkgrid")
    plt.figure(figsize=(20, 10),dpi=100)
    fig, ax = plt.subplots(1, 1,figsize=(20, 10),dpi=100, sharex=True)
    
    ax.plot(x, lower,color='black',label="x >= 0.05",alpha = 0.5)
    ax.plot(x, upper, color='black',label="0.95 >=x",alpha = 0.5)
    ax.plot(x, label, color='red',label="True",alpha = 0.5)
    ax.plot(x, mean, color='blue',label="mean",alpha = 0.5)
    plt.legend(loc="upper right")
    ax.fill_between(x, lower, upper, where=upper >= lower, facecolor='green', alpha=0.1,interpolate=True)
    ax.set_title('Confidenzintervall')
    plt.ylim([0.0, np.max(label)])
    plt.xlim([0, len(q)])


def histogramm(q,l):
    _,x,y,_ = q[0].shape
    
    hist = np.zeros(256)
    hist_q = hist.copy()
        
    for i,d in enumerate(q):
        label = l[i]
        label = label.ravel()
        d = d.reshape(x*y,3)
        for j in range(x*y):
            hist[int(label[j])] += 1
            v = label[j]
            if d[j][0] <= v and d[j][-1] >= v:
                hist_q[int(label[j])] += 1
    return hist,hist_q


def plotHist(hist,hist_q):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import colors
    from matplotlib.ticker import PercentFormatter
    sns.set(style="darkgrid")
    plt.figure(figsize=(20, 10),dpi=100)
    

    hist_n = hist / hist[1:50].sum()
    plt.title('Liegt im Quantil')
    plt.bar(np.arange(1,51),hist_n[1:51],color="red",alpha=0.5,lw=.1,label="True")
    plt.ylim([0.0, 0.2])
    plt.xlim([0, 50])
    plt.xlabel('Label')
    plt.ylabel('relative Häufigkeit')
    #plt.show()

    hist_q_n = hist_q / hist[1:50].sum()
    plt.bar(np.arange(1,51),hist_q_n[1:51],color="blue",alpha=0.5,lw=.1,label="Bereich")
    plt.ylim([0.0, 0.2])
    plt.xlim([0, 50])
    plt.legend(loc="upper right")
    plt.show()


def someStats(hist,hist_q):
    print("Anzahl an labels           :",hist.sum())
    print("Anzahl liegt im C-Intervall {} | {:.2f}:".format(hist_q.sum(),hist_q.sum()/hist.sum()))
    print("\nOhne Regen")
    print("Anzahl an labels           :",hist[1:].sum())
    print("Anzahl liegt im C-Intervall: {} | {:.2f}%".format(hist_q[1:].sum(),hist_q[1:].sum()/hist[1:].sum()))