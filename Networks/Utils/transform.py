import cv2 
from multiprocessing import Process, cpu_count
import os
from time import sleep
from PIL import Image
import numpy as np
import pandas as pd

def fProcess(listOfFiles,savedir,dim):

    while listOfFiles:
        file = listOfFiles.pop()

        filename = file.split('/')[-1]
        pathToWrite = os.path.join(savedir,filename)

        if os.path.exists(pathToWrite):
            print("File ",pathToWrite," exists",len(listOfFiles))
            continue

        
        img = np.array(Image.open(file))
        x,y = dim
        img = cv2.resize(img,(y,x))
        img = Image.fromarray(img)
        img.save(pathToWrite)


        
def resizeImages(listOfFiles,dim,savedir,saveListOfFiles):
    
    
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    else: 
        return list(pd.read_csv(os.path.join(savedir,saveListOfFiles))["colummn"])

    nbrProcesses = cpu_count() * 2   
    splittedlist = []
    stepsize = len(listOfFiles) // nbrProcesses
    splittedlist = [listOfFiles[i:i + stepsize] for i in range(0, len(listOfFiles), stepsize)]

    jobs = []
    for i in range(nbrProcesses+1):
        p = Process(target=fProcess, args=(splittedlist[i],savedir,dim))
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()

    newListOfFiles = []
    for file in listOfFiles:
        newListOfFiles.append(os.path.join(savedir,file.split("/")[-1]))
    dataframe = pd.DataFrame(newListOfFiles,columns=["colummn"])
    dataframe.to_csv(os.path.join(savedir,saveListOfFiles),index=False)
    return list(pd.read_csv(os.path.join(savedir,saveListOfFiles))["colummn"])





