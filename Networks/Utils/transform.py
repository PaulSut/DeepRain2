import cv2
from multiprocessing import Process, cpu_count
import os
from time import sleep
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf


class fromCategorical(object):
    """docstring for fromCategorical"""

    def __init__(self, conditions):
        super(fromCategorical, self).__init__()
        self.conditions = conditions


class from_sparse_categorical_lin(object):

    def __init__(self, num_classes, multi_val):
        self.num_classes = num_classes
        self.multi_val = multi_val

    def __call__(self, array):
        array_shape = array.shape
        array = np.reshape(array, (array.shape[1] * array.shape[2], self.num_classes))

        new_vector = []
        for pixel in array:
            index_of_max_val = np.argmax(pixel)
            new_vector.append(index_of_max_val*self.multi_val)

        new_vector = np.asanyarray(new_vector)
        final_vector = np.reshape(new_vector, (array_shape[1], array_shape[2]))

        return final_vector


class from_sparse_categorical(object):

    def __call__(self, array):
        array_shape = array.shape
        target_value0 = 0
        target_value1 = 2
        target_value2 = 10
        target_value3 = 11

        #array = np.reshape(array, (array.shape[1] * array.shape[2], 4))
        array = np.reshape(array, (array.shape[0] * array.shape[1], 4))

        new_vector = []
        for pixel in array:
            index_of_max_val = np.argmax(pixel)

            if index_of_max_val == 0:
                new_vector.append(target_value0)
            elif index_of_max_val == 1:
                new_vector.append(target_value1)
                print('light Rain')
            elif index_of_max_val == 2:
                new_vector.append(target_value2)
                print('medium Rain')
            elif index_of_max_val == 3:
                new_vector.append(target_value3)

        new_vector = np.asanyarray(new_vector)
        if len( np.unique(new_vector, return_counts=True)[0])>1:
            print('From Sarse: ', np.unique(new_vector, return_counts=True))
        #final_vector = np.reshape(new_vector, (array_shape[1], array_shape[2]))
        final_vector = np.reshape(new_vector, (array_shape[0], array_shape[1]))
        return final_vector

class ToCategorical(object):
    """

        Map array values to values in conditions

        [1,50,60]

        values between 1 and 50 will be mapped to index 0
        => [1,0,0]

    """

    def __init__(self, conditions):
        super(ToCategorical, self).__init__()
        self.conditions = conditions
        self.numClasses = len(self.conditions) - 1

    def __call__(self, array):

        newVector = np.zeros((*array.shape, self.numClasses))

        for i in range(1, self.numClasses + 1):
            value = self.conditions[i]
            valuePrev = self.conditions[i - 1]
            idx = np.where((array <= value) & (array > valuePrev))
            classV = np.zeros((self.numClasses))
            classV[i - 1] = 1
            for x_idx, y_idx in zip(idx[0], idx[1]):
                newVector[x_idx, y_idx] = classV

        for i in newVector:
            if i.max() < 1:
                print(array[i])
                exit(-1)

        # newVector = newVector.flatten()
        #print('ToCategorical Shape', newVector.shape)
        #print(newVector[:10])
        return newVector


class Normalize(object):
    def __call__(self, array):
        new_array = array/255
        #print('unique val ', np.unique(new_array, return_counts=True))
        return new_array


class Binarize(object):
    """docstring for Binarize"""

    def __init__(self, threshold=0, value=255):
        super(Binarize, self).__init__()
        self.threshold = threshold
        self.value = value

    def __call__(self, img):
        img[np.where(img > self.threshold)] = self.value
        img[np.where(img <= self.threshold)] = 0
        return img


class Flatten(object):
    """docstring for Flatten"""

    def __init__(self):
        super(Flatten, self).__init__()

    def __call__(self, img):
        img = img.flatten()
        return img


class NormalDist(object):
    """docstring for Normaldistribution"""

    def __init__(self):
        super(NormalDist, self).__init__()

    def __call__(self, img):
        return img - 127.0


########################################################################
###                     PRETRANSFORMATIONS                           ###
########################################################################


class cutOut(object):
    """docstring for cutOut"""

    def __init__(self, slices):
        super(cutOut, self).__init__()
        # assert type(slices) is slice, "Parameter slices needs to be type of slice!"
        self.idx = slices
        self.slices = [slice(slices[0], slices[1]), slice(slices[2], slices[3])]

    def __call__(self, img):
        return img[self.slices]

    def __str__(self):
        savefolder = str(self.idx[0]) + "x" + str(self.idx[1]) + "_" + str(self.idx[0]) + "x" + str(self.idx[1])
        return savefolder


class resize(object):
    """

        resizes image to dimension dim
    """

    def __init__(self, dim):
        super(resize, self).__init__()
        self.dim = dim

    def __call__(self, img):
        x, y = self.dim
        img = cv2.resize(img, (y, x))
        return img

    def __str__(self):
        savefolder = ""
        for i in self.dim:
            savefolder += str(i) + "x"
        savefolder = savefolder[:-1]
        return savefolder


########################################################################
###                             WORKER                              ###
########################################################################


def fProcess(listOfFiles, savedir, transformations):
    while listOfFiles:
        file = listOfFiles.pop()

        filename = file.split('/')[-1]

        pathToWrite = os.path.join(savedir, filename)

        if os.path.exists(pathToWrite):
            print("File ", pathToWrite, " exists", len(listOfFiles))
            continue

        img = np.array(Image.open(file))

        for transformation in transformations:
            img = transformation(img)

        img = Image.fromarray(img)
        img.save(pathToWrite)


def transformImages(listOfFiles, transformation, savedir, saveListOfFiles):
    """
        Transforms images specified by parameter transformation
        transformation needs to be a class with functions __call__ and __str__
        __call__ will should perform the transformation.
                __call__ receives an image and returns the transformed image
        __str__ should return the name of the path where the images are stored
    """

    if not os.path.exists(savedir):
        os.mkdir(savedir)

    else:
        return list(pd.read_csv(os.path.join(savedir, saveListOfFiles))["colummn"])

    nbrProcesses = cpu_count() * 2
    splittedlist = []
    stepsize = len(listOfFiles) // nbrProcesses

    # splittedlist = [listOfFiles[i:i + stepsize] for i in range(0, len(listOfFiles), stepsize)]

    for i in range(0, len(listOfFiles), stepsize):
        splittedlist.append(listOfFiles[i:i + stepsize])

    jobs = []
    for i in range(len(splittedlist)):
        p = Process(target=fProcess, args=(splittedlist[i], savedir, transformation))
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()

    newListOfFiles = []
    for file in listOfFiles:
        newListOfFiles.append(os.path.join(savedir, file.split("/")[-1]))
    dataframe = pd.DataFrame(newListOfFiles, columns=["colummn"])
    dataframe.to_csv(os.path.join(savedir, saveListOfFiles), index=False)
    return list(pd.read_csv(os.path.join(savedir, saveListOfFiles))["colummn"])