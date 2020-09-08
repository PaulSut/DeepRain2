import numpy as np
from tensorflow.keras.models import load_model
from min30_LSTM_conv_znBinomial import *
from Utils.Workflow.dwd_data import get_new_radar_pngs
from  Utils.Workflow.CreateRainPNG import create_rain_image, resize_images, take_slice_of_image, create_rain_intensity_values
from Utils.Workflow.NeuralNetwork import predict_weather_several_models
from Utils.Workflow.Database import replace_forecast_in_firebase
import PIL
from time import sleep
import os
import pickle


#current_working_dir = os.path.abspath(os.getcwd())
PATH_TO_FORECAST_DIR = 'Data/DWD/forecast/'
HISTORICAL_PICTURE_PATH = 'Data/DWD/historical_data/images/'
SCLICE_PICTURE_PATH = 'Data/DWD/historical_data/slices/'
FORECAST_GRAYSCALE_PATH = 'Data/DWD/forecast_grayscale/'
HISTORICAL_RADAR_DATA_PATH = 'Data/DWD/historical_data/bin/'
RESIZE = None

# Konstanz: 56, 456
SLICES = [ 8, 104, 438, 534]


TARGET_DIMENSION = [900, 900]
IMAGE_POS = [8, 438]

NUMBER_OF_INPUT_IMAGES = 5
NUMBER_OF_HISTORICAL_IMAGES = 5

if __name__ == '__main__':

    # get the lists with coordinate data
    f = open('Utils/Workflow/listLatitudeComplete.pckl', 'rb')
    listLatitude = pickle.load(f)
    f.close()

    f = open('Utils/Workflow/listLongitudeComplete.pckl', 'rb')
    listLongitude = pickle.load(f)
    f.close()

    f = open( 'Utils/Workflow/listCoordinates.pckl', 'rb')
    listCoordinates = pickle.load(f)
    f.close()

    coordinate_lists = [listLatitude, listLongitude, listCoordinates]

    # get the models
    models = []
    model_prediction_time = []
    get_LSTM_Model = getModel
    MODELNAME_LSTM = MODELNAME

    LSTM_model, checkpoint_lstm, modelpath_lstm, train, test = get_LSTM_Model()

    history_path = os.path.join(modelpath_lstm, MODELNAME_LSTM + "_history")
    laststate = getBestState(modelpath_lstm, history_path)
    epoch = laststate["epoch"]

    print(laststate)
    LSTM_model.load_weights(laststate["modelpath"])

    models.append(LSTM_model)
    model_prediction_time.append('30_minutes')

    new_radar_data_downloaded = False
    while True:
        while not new_radar_data_downloaded:
            new_radar_data_downloaded = get_new_radar_pngs(HISTORICAL_RADAR_DATA_PATH, HISTORICAL_PICTURE_PATH)
            if not new_radar_data_downloaded:
                print('New radar data is not available')
                sleep(5)
        new_radar_data_downloaded = False

        if SLICES is not None:
            take_slice_of_image(SLICES, HISTORICAL_PICTURE_PATH, SCLICE_PICTURE_PATH, NUMBER_OF_HISTORICAL_IMAGES)
        else:
            SCLICE_PICTURE_PATH = HISTORICAL_PICTURE_PATH
        if RESIZE is not None:
            resize_images(DIMENSION, SCLICE_PICTURE_PATH, FORECAST_GRAYSCALE_PATH, NUMBER_OF_HISTORICAL_IMAGES, reverse=SLICES is None)

        print('Predicting the weather')
        predict_weather_several_models(models, model_prediction_time, NUMBER_OF_INPUT_IMAGES,
                                       SCLICE_PICTURE_PATH, FORECAST_GRAYSCALE_PATH)

        print('Create forecast images')
        forecast_picture_list = os.listdir(FORECAST_GRAYSCALE_PATH)
        forecast_picture_list.sort()

        #get the historical images

        historical_picture_list = os.listdir(SCLICE_PICTURE_PATH)
        historical_picture_list.sort()

        historical_images_time = os.listdir(HISTORICAL_PICTURE_PATH)
        historical_images_time.sort()
        print(historical_images_time)
        historical_images_time = historical_images_time[-5:]
        print('Moin1')
        print(historical_images_time)


        historical_images = []
        rain_intensity_values = []
        time_stamps = []
        for i, path in enumerate(historical_picture_list[:5]):
            img = np.asanyarray(PIL.Image.open(SCLICE_PICTURE_PATH + path, mode='r'))
            img_rgba = np.asanyarray(PIL.Image.open(SCLICE_PICTURE_PATH + path, mode='r').convert(mode='RGBA'))
            historical_images.append(
                PIL.Image.fromarray(create_rain_image(img, img_rgba, TARGET_DIMENSION, IMAGE_POS), mode='RGBA'))
            rain_intensity_values.append(create_rain_intensity_values(img, TARGET_DIMENSION, IMAGE_POS))
            time_stamps.append(historical_images_time[i][15:-8])

        print(time_stamps)
        # historical_images.reverse()

        forecast_images = []
        for i, path in enumerate(forecast_picture_list):
            img = np.asanyarray(PIL.Image.open(FORECAST_GRAYSCALE_PATH + path, mode='r'))
            img_rgba = np.asanyarray(PIL.Image.open(FORECAST_GRAYSCALE_PATH + path, mode='r').convert(mode='RGBA'))
            forecast_images.append(
                PIL.Image.fromarray(create_rain_image(img, img_rgba, TARGET_DIMENSION, [8+16, 438+16]), mode='RGBA'))
            rain_intensity_values.append(create_rain_intensity_values(img, TARGET_DIMENSION, IMAGE_POS))
            #calc next time step (each time + 10 min)
            last_time_step = time_stamps[-1]
            if int(last_time_step[-2:]) + 10 > 60:
                hours = int(last_time_step[:-2])+1
                minutes = int(last_time_step[-2:]) - 50
            else:
                hours = int(last_time_step[:-2])
                minutes = int(last_time_step[-2:]) + 10

            time_stamps.append(str(hours)+str(minutes))


        print(time_stamps)




        print('Upload forecast images')
        replace_forecast_in_firebase(historical_images, forecast_images, PATH_TO_FORECAST_DIR)
        print('Done. Weather Forecast is up to date')