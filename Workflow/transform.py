import cv2
from multiprocessing import Process, cpu_count
import os
from time import sleep
from PIL import Image
import numpy as np
import pandas as pd


class fromCategorical(object):
    """docstring for fromCategorical"""

    def __init__(self, conditions):
        super(fromCategorical, self).__init__()
        self.conditions = conditions

class Normalize(object):
    def __call__(self, array):
        new_array = array/255
        return new_array


# class to_int_categorical(object):

class ToSparseCategorical(object):
    """

    """

    def __call__(self, array):
        shape = np.asanyarray(array.shape)
        # shape = np.append(shape, 6)
        newVector = np.zeros(shape)
        # newVector = np.zeros(array.shape)

        threshold_value_calls1 = 2
        threshold_value_calls2 = 10


        target_class_0 = 0
        target_class_1 = 1 / 3
        target_class_2 = 2 / 3
        target_class_3 = 3 / 3

        newVector[array <= 0] = target_class_0
        newVector[(threshold_value_calls1 >= array) & (array > 0)] = target_class_1
        newVector[(array <= threshold_value_calls2) & (array > threshold_value_calls1)] = target_class_2
        newVector[(array > threshold_value_calls2)] = target_class_3
        return newVector

    '''
    def __call__(self, array):
        shape = np.asanyarray(array.shape)
        #shape = np.append(shape, 6)
        newVector = np.zeros(shape)
        # newVector = np.zeros(array.shape)

        # create 6 different classes
        threshold_value_calls1 = 2
        threshold_value_calls2 = 5
        threshold_value_calls3 = 10
        threshold_value_calls4 = 20

        target_class_0 = 1/6
        target_class_1 = 2/6
        target_class_2 = 3/6
        target_class_3 = 4/6
        target_class_4 = 5/6
        target_class_5 = 6/6
    '''


        #target_class_0 = [1, 0, 0, 0, 0, 0]
        #target_class_1 = [0, 1, 0, 0, 0, 0]
        #target_class_2 = [0, 0, 1, 0, 0, 0]
        #target_class_3 = [0, 0, 0, 1, 0, 0]
        #target_class_4 = [0, 0, 0, 0, 1, 0]
        #target_class_5 = [0, 0, 0, 0, 0, 1]

    '''

        newVector[array <= 0] = target_class_0
        newVector[(threshold_value_calls1 >= array) & (array > 0)] = target_class_1
        newVector[(array <= threshold_value_calls2) & (array > threshold_value_calls1)] = target_class_2
        newVector[(array <= threshold_value_calls3) & (array > threshold_value_calls2)] = target_class_3
        newVector[(array <= threshold_value_calls4) & (array > threshold_value_calls3)] = target_class_4
        newVector[(array > threshold_value_calls4)] = target_class_5

        #print(newVector.shape)
        newVector = np.reshape(newVector, (272, 224))
        # print(newVector.shape)
        # print(newVector.flatten().shape)
        # print(newVector)
        # print('\nUnique Values X: ', np.unique(newVector, return_counts=True))
        return newVector
        # return newVector.flatten()
    '''

class from_sparse_categorical(object):

    def __call__(self, array):
        array_shape = array.shape
        target_value0 = 0
        target_value1 = 2
        target_value2 = 10
        target_value3 = 11
        #target_value1 = 2
        #target_value2 = 5
        #target_value3 = 10
        #target_value4 = 20
        #target_value5 = 50


        array = np.reshape(array, (array.shape[1]*array.shape[2], 4))

        new_vector = []
        for pixel in array:
            index_of_max_val = np.argmax(pixel)
            #if index_of_max_val != 0:
                #print('It will rain')

            if index_of_max_val == 0:
                new_vector.append(target_value0)
            elif index_of_max_val == 1:
                new_vector.append(target_value1)
            elif index_of_max_val == 2:
                new_vector.append(target_value2)
            elif index_of_max_val == 3:
                new_vector.append(target_value3)
            elif index_of_max_val == 4:
                new_vector.append(target_value4)
            elif index_of_max_val == 5:
                new_vector.append(target_value5)

        new_vector = np.asanyarray(new_vector)
        #print('New Vec shape: ',new_vector.shape)
        final_vector = np.reshape(new_vector, (array_shape[1], array_shape[2]))

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

        for i in range(1, self.numClasses):
            value = self.conditions[i]
            valuePrev = self.conditions[i - 1]
            idx = np.where((array < value) & (array >= valuePrev))
            classV = np.zeros((self.numClasses))
            classV[i - 1] = 1
            newVector[idx, :] = classV

        for i in newVector:
            if i.max() < 1:
                print(i)
                exit(-1)

        return newVector.flatten()


class Binarize(object):
    """docstring for Binarize"""

    def __init__(self, threshold=0, value=255):
        super(Binarize, self).__init__()
        self.threshold = threshold
        self.value = value

    def __call__(self, img):
        # print('\nUnique Values X: ', np.unique(img, return_counts=True))
        img[np.where(img > self.threshold)] = self.value
        img[np.where(img <= self.threshold)] = 0

        # print('\nUnique Values X: ', np.unique(img, return_counts=True))
        return img


class Flatten(object):
    """docstring for Flatten"""

    def __init__(self):
        super(Flatten, self).__init__()

    def __call__(self, img):
        img = img.flatten()
        return img

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
    for i in range(nbrProcesses):
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





