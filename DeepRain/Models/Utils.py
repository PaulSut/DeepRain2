import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.pyplot import figure
import numpy as np
import json
import re
import os


def plotHistory(history,filename,title="Model"):

    # Plot training & validation loss values
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(filename+'.png')


def saveHistory(path,history):
    json.dump(history,open(path+".json",'w'))

def loadHistory(path):   
    return json.load(open(path+".json"))


def getBestWeights(path):
    weights = []
    for file in os.listdir(path):
        if file.endswith(".h5"):
            weights.append(os.path.join(path, file))

    if len(weights) == 0:
        return None
    weights.sort()
    
    return weights[-1]


def getBestState(path,history_path):

    regex = r"-([0-9]{3,4})-"
    try:
        modelpath = getBestWeights(path)
        history   =  loadHistory(history_path)
        
    except :
        return None

    matches = re.search(regex, modelpath)


    if matches:
        clampedhist = {}
        retdict = {}
        epoch = int(matches[1])
        for key in history:
            clampedhist[key] = history[key][:epoch]

        retdict["history"]   = clampedhist
        retdict["epoch"]     = epoch
        retdict["modelpath"] = modelpath

        return retdict

    
    print("Couldn't find Epoch in {}".format(modelpath))
    return None

def mergeHist(oldhist,newhist):
    for key in oldhist:
        oldhist[key] += newhist[key]
    return oldhist