import keras

import numpy as np
import os
import PIL
import cv2
from DeepRain.Utils.transform import *
import datetime

def predict_weather_several_models(models, model_prediction_time,number_of_input_images, path_to_input_images, path_to_forecast_grayscale, transform_input=None, transform_output=None):

    for index, model in enumerate(models):
        print(model_prediction_time[index] + ' prediction')

        #generate input
        input_data_filenames = os.listdir(path_to_input_images)
        input_data_filenames.sort(reverse=True)
        input_data = []
        for image_name in input_data_filenames[:number_of_input_images]:
            image = np.asanyarray(PIL.Image.open(path_to_input_images + image_name, mode='r'))/255
            if transform_input is not None:
                for operation in transform_input:
                    image = operation(image)
            input_data = [image] + input_data

        input_data = np.asanyarray([input_data])
        input_data = np.transpose(input_data, (0, 2, 3, 1))

        forecast = np.array(model(input_data).mean())[0,:,:,:]

        cv2.imwrite(path_to_forecast_grayscale+f'9{index}.png', forecast)

    return True


def predict_weather(model, number_of_input_images, number_of_predictions, path_to_forecast_grayscale, transform_input=None, transform_output=None):

    for prediction in range(number_of_predictions):
        print('Prediction '+str(prediction+1)+'/'+str(number_of_predictions))
        input_data_filenames = os.listdir(path_to_forecast_grayscale)
        input_data_filenames.sort(reverse=True)
        input_data = []
        for image_name in input_data_filenames[:number_of_input_images]:
            image = np.asanyarray(PIL.Image.open(path_to_forecast_grayscale+image_name, mode='r'))
            if transform_input is not None:
                for operation in transform_input:
                    image = operation(image)
            input_data = [image] + input_data


        final_data = []
        for a,b,c,d,e in zip(input_data[0], input_data[1], input_data[2], input_data[3], input_data[4]):
            line = []
            for i in range(len(a)):
                line.append([a[i],b[i],c[i],d[i],e[i]])
            final_data.append(line)
        input_data = np.asanyarray([final_data])


        forecast = model.predict(input_data)



        #if transform_output is not None:
        #    for operation in transform_output:
        #        forecast = operation(forecast)

        '''
        # for Unet cat entropy
        forecast_data = np.reshape(forecast, (60928, 2))
        #print(forecast_data.shape)
        forecast = []
        for prediction_sample in forecast_data:
            #print(prediction_sample)
            if prediction_sample[0]>prediction_sample[1]:
                forecast.append(0)
            else:
                #print('Mooooin')
                forecast.append(255)

        forecast = np.reshape(forecast, (1, 272, 224, 1))
        # until here 
        '''
        forecast[forecast>255] = 255
        forecast[forecast<0] = 0
        cv2.imwrite(path_to_forecast_grayscale+f'9{prediction}.png', forecast)

        #image = PIL.Image.fromarray(forecast[0], mode='L')
        #image.save(path_to_forecast_grayscale+f'1{prediction}.png')

    return True