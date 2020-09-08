import cv2 
from multiprocessing import Process, cpu_count
import os
from time import sleep
import numpy as np
import pandas as pd

class LinBin(object):
    
    """docstring for LinBin"""
    def __init__(self, divisor=12):
        super(LinBin, self).__init__()
        self.div = divisor

    def __call__(self,img):
        return np.ceil(img / self.div)

class LogBin(object):
    """docstring for LogBin"""
    def __init__(self):
        super(LogBin, self).__init__()
        
    def __call__(self,img):
        return np.ceil(np.log(img +1.0))
    
        

class NormalizePerImage(object):
    """docstring for NormalizePerImage"""
    def __init__(self):
        super(NormalizePerImage, self).__init__()

    def __call__(self,img):
        std = img.std()
        if std == 0:
            std = 1

        return (img - img.mean()) / std
        

class Normalize(object):
    """docstring for Normalize"""
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.std  = std
        self.mean = mean
    
    def __call__(self,img):
        #return img / self.std
        return (img - self.mean) / self.std
        #return img - self.mean

class Wiggle(object):
    """docstring for Wiggle"""
    def __init__(self):
        super(Wiggle, self).__init__()
        self.draw()
        
    def draw(self):
        self.wiggle_idx = np.random.randint(low=-10,high=10,size=2)


    def __call__(self,img):
        newarray = np.zeros_like(img)
        x,y = img.shape[:2]

        x_shift = self.wiggle_idx[0]
        y_shift = self.wiggle_idx[1]
        

        if x_shift >= 0 and y_shift >= 0:
            newarray[:x-x_shift,:y-y_shift] = \
                 img[x_shift:,y_shift:]
        elif x_shift <= 0 and y_shift >= 0:
            
            newarray[-x_shift:,:y-y_shift] = \
                 img[:x_shift,y_shift:]
        elif x_shift >= 0 and y_shift <= 0:
            
            newarray[:x-x_shift,-y_shift:] = \
                 img[x_shift:,:y_shift]
        else:
            
            newarray[-x_shift:,-y_shift:] = \
                 img[:x_shift,:y_shift]

        return newarray


class cutOut(object):
    """docstring for cutOut"""
    def __init__(self,slices):
        super(cutOut, self).__init__()
        #assert type(slices) is slice, "Parameter slices needs to be type of slice!"
        self.idx = slices
        self.slices = [slice(slices[0],slices[1]),slice(slices[2],slices[3])]

    def __call__(self,img):
        return img[tuple(self.slices)]

    def __str__(self):
        savefolder=str(self.idx[0])+"x"+str(self.idx[1])+"_"+str(self.idx[2])+"x"+str(self.idx[3])
        return savefolder


class ImageToPatches(object):
    """docstring for ImageToPatches"""
    def __init__(self,outputsize,inputsize = None,stride=(0,0)):
        super(ImageToPatches, self).__init__()
        self.stride = stride
        self.outputsize = outputsize
        self.inputsize = inputsize


        if self.inputsize is None:
            self.inputsize = self.outputsize

        self.offset_x = (self.inputsize[0] - self.outputsize[0]) // 2
        self.offset_y = (self.inputsize[1] - self.outputsize[1]) // 2


    def get_y_by_index(self,y,i,j):

        
        patch_y  = np.zeros(self.outputsize,dtype=np.uint8)

        start_x = i * self.outputsize[0] - self.stride[0]
        end_x   = start_x + self.outputsize[0] 

        start_y = j * self.outputsize[1] - self.stride[1]
        end_y   = start_y + self.outputsize[1]


        start_x = start_x if start_x > 0 else 0
        start_y = start_y if start_y > 0 else 0
        
        end_x = end_x if end_x < self.outputsize[0] else self.outputsize[0]
        end_y = end_y if end_y < self.outputsize[1] else self.outputsize[1]


    def __call__(self,img):
        x = img
        y = img

        input_matrix  = [] 
        output_matrix = []

        
        for index_i in range(0,x.shape[0],self.outputsize[0]-self.stride[0]):
            patch_in = []
            patch_ou = []
            start_x = index_i - self.offset_x
            end_x   = index_i + self.outputsize[0] + offset_x
            start_x = start_x if start_x >= 0 else 0
            end_x = end_x if end_x <= x.shape[0] else x.shape[0]

            for index_j in range(0,y.shape[1],self.outputsize[1]-self.stride[1]):
                
                patch_x  = np.zeros(self.inputsize,dtype=np.uint8)
                patch_y  = np.zeros(self.outputsize,dtype=np.uint8)


                start_y = index_j - self.offset_y
                end_y   = index_j+self.outputsize[1] + self.offset_y

                start_y = start_y if start_y >= 0 else 0
                end_y = end_y if end_y <= x.shape[1] else x.shape[1]
                
                patchx = x[start_x:end_x,start_y:end_y]
                patchy = y[index_i:index_i+self.outputsize[0],index_j:index_j+self.outputsize[1]]

                patch_y[:patchy.shape[0],:patchy.shape[1]] = patchy
                
                start_x_in = 0
                end_x_in = patchx.shape[0]
                start_y_in = 0
                end_y_in = patchx.shape[1]
                if patchx.shape[0] != self.inputsize[0]:
                    diff = self.inputsize[0] - patchx.shape[0]
                    if start_x == 0:
                        start_x_in += diff
                        end_x_in += diff



                if patchx.shape[1] != self.inputsize[1]:
                    diff = self.inputsize[1] - patchx.shape[1]
                    if start_y == 0:
                        start_y_in += diff
                        end_y_in += diff




                
                patch_x[start_x_in:end_x_in,start_y_in:end_y_in] = patchx

                patch_in.append(patch_x)
                patch_ou.append(patch_y)

            input_matrix.append(patch_in)
            output_matrix.append(patch_ou)
        
        return np.array(input_matrix),np.array(output_matrix)


def processes(listOfFiles,savedir,transformations):

    while listOfFiles:
        file     = listOfFiles.pop()[0]
        filename = file.split('/')[-1]
        pathToWrite = os.path.join(savedir,filename)

        if os.path.exists(pathToWrite):
            print("File {} exists".format(pathToWrite))
            continue

        img = np.array(cv2.imread(file,0))

        for transformation in transformations:
            img = transformation(img)


        #print(img.shape)
        cv2.imwrite(pathToWrite,img)


def transformImages(listOfFiles,
                    transformation,
                    savedir,
                    target=processes,
                    area = None):

    if not os.path.exists(savedir):
        os.mkdir(savedir)

    else:
        pass

    nbrProcesses = cpu_count() * 2
    stepsize = len(listOfFiles) // nbrProcesses

    splittedList = [listOfFiles[i:i+stepsize] for i in range(0,len(listOfFiles),stepsize)]

    def startjob(payload):
        job = Process(target = target, args = (payload,savedir,transformation))
        job.start()
        return job

    jobs = [ startjob(payload) for payload in splittedList ]

    for job in jobs: 
        job.join()

    newListOfFiles = [ os.path.join(savedir,file[0].split('/')[-1])\
                       for file in listOfFiles ]

    data_info = []
    for i,path in enumerate(newListOfFiles):
        print("Creating CSV file: {:07d}/{}".format(i,len(newListOfFiles)),end="\r")
        img = cv2.imread(path,0)
        if area is not None:
            img = area(img)
        data_info.append([path,img.mean(),img.std(),img.max()])
        columns = ["path","mean","std","max"]

    dframe  = pd.DataFrame(np.array(data_info),columns=columns)
    #dframe.to_csv(path_to_csvfile, index = False, header=True)

    return dframe

