import numpy as np
from tensorflow.keras.models import load_model
from min30_LSTM_conv_znBinomial import *
from Utils.Workflow.dwd_data import get_new_radar_pngs
from  Utils.Workflow.CreateRainPNG import create_rain_image, resize_images, take_slice_of_image, create_rain_intensity_values
from Utils.Workflow.NeuralNetwork import predict_weather_several_models
from Utils.Workflow.Database import replace_forecast_in_firebase
import Utils.Workflow.forecast_database_uploader as rain_intense_uploader
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
#SLICES = [ 750, 846, 250, 346]


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
    model_paths = [    'Models_weights/10min_LSTM_znBinomial/10min_LSTM_znBinomial-033-0.647808-0.696726.h5', 'Models_weights/20min_LSTM_znBinomial/20min_LSTM_znBinomial-019-0.722603-0.775276.h5', 'Models_weights/30min_LSTM_znBinomial/30min_LSTM_znBinomial-026-0.760087-0.825096.h5']
    model_prediction_time = ['10_minutes','20_minutes','30_minutes']


    get_LSTM_Model = getModel
    MODELNAME_LSTM = MODELNAME

    LSTM_model, checkpoint_lstm, modelpath_lstm, train, test = get_LSTM_Model()

    for path in model_paths:
        LSTM_model.load_weights(path)
        models.append(LSTM_model)


    # get input data
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
        historical_images_time = historical_images_time[-5:]


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
            if int(last_time_step[-2:]) + 10 > 59:
                hours = int(last_time_step[:-2])+1
                minutes = int(last_time_step[-2:]) - 50
            else:
                hours = int(last_time_step[:-2])
                minutes = int(last_time_step[-2:]) + 10
            if minutes<10:
                time_stamps.append(str(hours) +'0' +str(minutes))
            else:
                time_stamps.append(str(hours)+str(minutes))


        print('Upload forecast images')
        replace_forecast_in_firebase(historical_images, forecast_images, PATH_TO_FORECAST_DIR)


        print('Upload rain intensity values')

        #uploade forecast rain intense data to firebase
        rain_intense_uploader.upload_data_to_firbase(rain_intensity_values, time_stamps, coordinate_lists)

        print('Done. Weather Forecast is up to date')
