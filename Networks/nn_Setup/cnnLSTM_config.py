
from tensorflow.keras.optimizers import Adam
#from Utils.loss import *
from Utils.transform import *
import tensorflow as tf


def NLL(y_true, y_hat):
    return -y_hat.log_prob(y_true)

DIMENSION 					= (64,64)
INPUT_CHANNELS 				= 5
OPTIMIZER 					= Adam( lr= 1e+5 )
BATCHSIZE 					= 10

# Cut out the Area around Pxl
slices 						= [256,320,256,320]
cutOutFrame 				= cutOut(slices)


PRETRAINING_TRANSFORMATIONS = [cutOutFrame]
TRANSFORMATIONS 			= None
LOSSFUNCTION 				= NLL
METRICS 					= ["mse"]

# Custom objects for model loading

CUSTOM_OBJECTS={'exp':tf.exp,'NLL':LOSSFUNCTION}