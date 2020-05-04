from PIL import Image
import numpy as np
import pandas as pd
import re
import keras
import os
from .transform import transformImages, resize
import cv2
import PIL
from Utils.colors import *
from tensorflow.python.keras.utils.data_utils import Sequence
from Utils.loadset import getDataSet

CSVFILE = "./.listOfFiles.csv"
WRKDIR = "./Data"
TRAINSETFOLDER = os.path.join(WRKDIR, "train")
VALSETFOLDER = os.path.join(WRKDIR, "val")

DatasetFolder = "./Data/RAW"
PathToData = os.path.join(DatasetFolder, "MonthPNGData")


def provideData(dimension,
                batch_size,
                channels,
                transform_input=None,
                transform_output=None,
                preTransformation=None,
                flatten=False,
                year=[2017],
                onlyUseYears=None,
                DatasetFolder=DatasetFolder):
    """
        DatasetFolder       : Download Data to this Folder an use it as working directory
        preTransformation   : Transformation to perform on Data BEFORE loading it
        transform           : Transformation to perform on a single Image before hand it to the NN
        onlyUseYears        : List of years to use in Dataset

    """

    getDataSet(DatasetFolder, year=[2017])
    train, test = dataWrapper(PathToData,
                              dimension=dimension,
                              channels=channels,
                              batch_size=batch_size,
                              overwritecsv=True,
                              flatten=flatten,
                              onlyUseYears=onlyUseYears,
                              transform_input=transform_input,
                              transform_output=transform_output,
                              preTransformation=preTransformation)

    return train, test


def dataWrapper(path,
                dimension,
                channels,
                batch_size,
                csvFile=CSVFILE,
                workingdir=WRKDIR,
                split=0.25,
                flatten=False,
                sortOut=True,
                shuffle=True,
                overwritecsv=False,
                onlyUseYears=None,
                preTransformation=None,
                transform_input=None,
                transform_output=None, ):
    """
    
        Returns two Data objects, which can be used for training.
        The first returned object is the trainingdata, second is
        testdata.

        path        : path to Data
        channels    : number of channels used as input
        batch_size  : self-explanatory
        csvFile     : name of csvFile where path to files are stored
                      (the benefit is that it is not necessary to load
                      the whole set into RAM, which could be too large)
        workingdir  : Directory where Data is expanded (resized image are
                      stored)
        split       : split ratio (train/test)
        sortOut     : not used
        shuffle     : shuffle Data after epochs,
        onlyUseYears: [2016,2017] ... make sure to use years in list

    """

    data = prepareListOfFiles(path, sortOut=sortOut, overwritecsv=overwritecsv, onlyUseYears=onlyUseYears)
    trainingsSet, validationSet = splitData(data)

    if not os.path.exists(TRAINSETFOLDER):
        os.mkdir(TRAINSETFOLDER)
    if not os.path.exists(VALSETFOLDER):
        os.mkdir(VALSETFOLDER)

    filename, ext = os.path.splitext(csvFile)
    trainsetCSV = filename + "_train_" + ext
    valsetCSV = filename + "_val_" + ext

    train_dataframe = pd.DataFrame(trainingsSet, columns=["colummn"])
    train_dataframe.to_csv(os.path.join(TRAINSETFOLDER, trainsetCSV), index=False)
    val_dataframe = pd.DataFrame(validationSet, columns=["colummn"])
    val_dataframe.to_csv(os.path.join(VALSETFOLDER, valsetCSV), index=False)

    train = Dataset(TRAINSETFOLDER,
                    dim=dimension,
                    n_channels=channels,
                    batch_size=batch_size,
                    workingdir=TRAINSETFOLDER,
                    saveListOfFiles=trainsetCSV,
                    timeToPred=5,
                    flatten=flatten,
                    shuffle=shuffle,
                    transform_input=transform_input,
                    transform_output=transform_output,
                    preTransformation=preTransformation)

    val = Dataset(VALSETFOLDER,
                  dim=dimension,
                  n_channels=channels,
                  batch_size=batch_size,
                  workingdir=VALSETFOLDER,
                  saveListOfFiles=valsetCSV,
                  timeToPred=5,
                  flatten=flatten,
                  shuffle=shuffle,
                  transform_input=transform_input,
                  transform_output=transform_output,
                  preTransformation=preTransformation)

    return train, val


def splitData(data, split=0.25):
    dataLength = len(data)
    validation_length = int(np.floor(dataLength * split))

    validationSet = data[-validation_length:]
    trainingsSet = data[:-validation_length]

    return trainingsSet, validationSet


def prepareListOfFiles(path, workingdir=WRKDIR, nameOfCsvFile=CSVFILE, sortOut=False, overwritecsv=False,
                       onlyUseYears=None):
    if not os.path.exists(workingdir):
        os.mkdir(workingdir)

    if not os.path.exists(os.path.join(workingdir, nameOfCsvFile)) or overwritecsv or onlyUseYears is not None:

        listOfFiles = getListOfFiles(path)
        listOfFiles.sort()

        if onlyUseYears:
            newlist = []
            prefix = "YW2017.002_"
            for year in onlyUseYears:
                checkFile = prefix + str(year)
                for file in listOfFiles:
                    if checkFile in file:
                        newlist.append(file)
            listOfFiles = newlist

        dataframe = pd.DataFrame(listOfFiles, columns=["colummn"])
        dataframe.to_csv(os.path.join(workingdir, nameOfCsvFile), index=False)
        listOfFiles = dataframe

    listOfFiles = list(pd.read_csv(os.path.join(workingdir, nameOfCsvFile))["colummn"])

    return listOfFiles


def getListOfFiles(path):
    """
    
        stolen from :
        https://thispointer.com/python-how-to-get-list-of-files-in-directory-and-sub-directories/

    """

    directory_entries = os.listdir(path)
    files = []

    for entry in directory_entries:
        fullPath = os.path.join(path, entry)
        if os.path.isdir(fullPath):
            files = files + getListOfFiles(fullPath)
        else:
            files.append(fullPath)
    return files


def mergeLists(list1, list2):
    newlist = []
    for file in list1:
        name = file.split("/")[-1]
        for file2 in list2:
            name2 = file2.split("/")[-1]

            if name == name2:
                newlist.append(file)
    return newlist


class Dataset(Sequence):

    def __init__(self, path,
                 batch_size,
                 dim,
                 n_channels=4,
                 shuffle=True,
                 saveListOfFiles=CSVFILE,
                 workingdir=WRKDIR,
                 timeToPred=30,
                 timeSteps=5,
                 sequenceExist=False,
                 flatten=False,
                 sortOut=False,
                 lstm=True,
                 dtype=np.float32,
                 transform_input=None,
                 transform_output=None,
                 preTransformation=None):

        """
        
            timeToPred    : minutes forecast, default is 30 minutes
            timeSteps     : time between images, default is 5 minutes

        """
        assert batch_size > 0, "batch_size needs to be greater than 0"
        assert timeSteps % 5 == 0, "timesteps % 5 needs to be 0"
        assert timeToPred % 5 == 0, "timeToPred % 5 needs to be 0"

        self.path = path
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.workingdir = workingdir
        self.saveListOfFiles = saveListOfFiles
        self.timeToPred = timeToPred
        self.timeSteps = timeSteps
        self.steps = int(timeToPred / timeSteps)
        self.datatype = dtype
        self.flatten = flatten
        self.sortOut = sortOut
        self.lstm = lstm
        self.transform_input = transform_input
        self.transform_output = transform_output
        self.preTransformation = preTransformation

        if preTransformation is None:
            self.preTransformation = [resize(self.dim)]
        else:
            self.preTransformation = preTransformation

        # index offset
        self.label_offset = self.n_channels + self.steps - 1
        #print('Label Offset', self.label_offset)

        if not os.path.exists(self.workingdir):
            os.mkdir(self.workingdir)

        if not os.path.exists(os.path.join(self.workingdir, saveListOfFiles)):
            self.listOfFiles = getListOfFiles(self.path)
            self.listOfFiles.sort()
            dataframe = pd.DataFrame(self.listOfFiles, columns=["colummn"])
            dataframe.to_csv(os.path.join(self.workingdir, saveListOfFiles), index=False)
            self.listOfFiles = dataframe

        if type(self.preTransformation) is list:
            pathName = ""
            for i in range(0, len(self.preTransformation) - 1):
                transObj = self.preTransformation[i]
                pathName += str(transObj)
            pathName += str(self.preTransformation[-1])
            savefolder = pathName

        else:
            savefolder = str(self.preTransformation)

        self.listOfFiles = list(pd.read_csv(os.path.join(self.workingdir, saveListOfFiles))["colummn"])

        self.new_listOfFiles = transformImages(self.listOfFiles, self.preTransformation,
                                               os.path.join(workingdir, savefolder), saveListOfFiles)

        if len(self.new_listOfFiles) != len(self.listOfFiles):
            print(YELLOW + "WARNING: Length of lists does not match! " + RESET)
            print(YELLOW + "To stop this warning, delete {} folder and restart".format(
                os.path.join(workingdir, savefolder)) + RESET)
            print(YELLOW + "Trying to update..\n" + RESET)

            newlist = mergeLists(self.new_listOfFiles, self.listOfFiles)
            if len(newlist) == 0:
                print(RED + "Could not fix this Problem!!! abort" + RESET)
                exit(-1)

            print(GREEN + "Fixed! " + RESET)
            self.new_listOfFiles = newlist
            if len(self.new_listOfFiles) != len(self.listOfFiles):
                print(CYAN + "But length does not match again.... just delete the folders bruh" + RESET)

        self.listOfFiles = self.new_listOfFiles
        # self.listOfFiles = self.new_listOfFiles[416:716]
        len_list = len(self.new_listOfFiles)
        self.listOfFiles = self.new_listOfFiles[int(5/12*len_list):int(6/12*len_list)]

        self.indizes = np.arange(len(self))

        if self.sortOut:
            self.indizes = sortOutFiles(self.listOfFiles, self.indizes)

    def __data_generation(self, index):
        X = []
        Y = []

        for i, id in enumerate(range((self.batch_size * index), (self.batch_size * index) + self.n_channels)):
            img = np.array(Image.open(self.listOfFiles[id]))
            #image = PIL.Image.fromarray(img)
            #image.save('/home/paul/Documents/DeepRain/Simon_git/DeepRain2/Networks/testForDataPy/' + str(i) + '.png')
            if self.transform_input is not None:

                for operation in self.transform_input:
                    img = operation(img)

            assert img.shape == self.dim, \
                "[Error] (Data generation) Image shape {} does not match dimension {}".format(img.shape, self.dim)
            X.append(img)


        try:
            # print(np.asanyarray(label).shape)
            # print(np.unique(label, return_counts=True))
            label = np.array(Image.open(self.listOfFiles[(self.batch_size * index) + self.label_offset]))
            #image = PIL.Image.fromarray(label)
            #image.save('/home/paul/Documents/DeepRain/Simon_git/DeepRain2/Networks/testForDataPy/' + 'label' + '.png')
            #if np.any(label>130):
            #    print(np.unique(label, return_counts=True))
            #    exit(-1)
            #label = label
        except Exception as e:
            print("\n\n Error in __data_generation", e, index, self.label_offset, len(self.listOfFiles))
            exit(-1)

        if self.transform_output is not None:

            for operation in self.transform_output:
                label = operation(label)


        Y.append(label)

        return np.array(X), np.array(Y)

    def on_epoch_end(self):

        if self.shuffle == True:
            np.random.shuffle(self.indizes)

    def __len__(self):

        return int(np.floor((len(self.listOfFiles) - self.label_offset) / self.batch_size))

    def __getitem__(self, index):

        if index >= len(self) - 1:
            self.on_epoch_end()

        X = []
        Y = []
        id_list = self.indizes[index:index + self.batch_size]

        for idd in id_list:
            x, y = self.__data_generation(idd)

            X.append(x)
            Y.append(y)
            # Y = y

        X = np.array(X)
        Y = np.array(Y)

        # print(X.shape)
        # X = np.transpose(X, (0, 2, 3, 1))
        # Y = np.transpose(Y, (0, 2, 3, 1))
        # print(X.shape)
        # return X, Y
        if self.transform_input is not None or self.transform_output is not None:
            # print(X.shape)
            # print(Y.shape)
            X = np.transpose(X, (0, 2, 3, 1))
            # X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]*X.shape[3], X.shape[4] ))
            Y = np.reshape(Y, (Y.shape[0], self.dim[0], self.dim[1], 4))
            # X = dim
            # Y = np.reshape(Y, (Y.shape[0], 272, 224, 4))
            # print(X.shape)
            # print(Y.shape)
            return X, Y
        if not self.flatten:
            Y = np.transpose(Y, (0, 2, 3, 1))

        else:

            return X / 255, Y.flatten()

        return X / 255.0, Y / 1.0
