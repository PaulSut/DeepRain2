import os
from ftplib import FTP
import pickle
#import cupy as cp
import numpy as np
import wradlib as wrl
import cv2
import PIL
from PIL import Image

def get_new_radar_pngs(path_to_radar_data, path_to_images):

    # download new radar data if available
    new_data_was_available = download_new_radar_data(path_to_radar_data)
    # create png
    if new_data_was_available:
        create_new_pngs(path_to_radar_data, path_to_images)

    return new_data_was_available

def  download_new_radar_data(path_to_radar_data):
    DWD_HOST = 'ftp-cdc.dwd.de'
    DWD_PATH = '/weather/radar/radolan/ry/'

    ftp = FTP(DWD_HOST)
    ftp.login()
    ftp.cwd(DWD_PATH)
    files_on_server = ftp.nlst()
    files_on_server.sort(reverse=True)

    radar_filenames = os.listdir(path_to_radar_data)

    new_file_available = False
    for file in files_on_server[1:11]:
        if file not in radar_filenames:
            print('Download new radar_file: ', file)
            new_file_available = True

            with open(path_to_radar_data+file, 'wb') as fp:
                ftp.retrbinary('RETR '+ file, fp.write)

    ftp.close
    return new_file_available


def create_new_pngs(path_to_radar_data, path_to_images):
    radar_filenames = os.listdir(path_to_radar_data)
    radar_filenames.sort(reverse=True)

    image_filenames = os.listdir(path_to_images)
    RadarValues99, maxOfRadarValues99, scaleFactor = initTranformRadarToPng()
    for radar_file in radar_filenames[:10]:
        if radar_file[:-6]+'.png' not in image_filenames:
            radar_data, attrs = wrl.io.read_radolan_composite(path_to_radar_data+radar_file)
            png_data= transformRadarDataToPng(radar_data, RadarValues99, maxOfRadarValues99, scaleFactor)
            save_png_grayscale_8bit(png_data, path_to_images, radar_file[:-6])

    return True

def initTranformRadarToPng ():
    RadarValues99 = np.asanyarray(pickle.load(open("RadarValues99PercentDatapoints.p", "rb")))
    maxValue = 40 # actual maxVal = maxVal + 99% Quanil (which is 1,39)
    remaining_range = 255-len(RadarValues99)
    scaleFactor = remaining_range/maxValue
    maxOfRadarValues99 = max(RadarValues99[:,0])
    RadarValues99 = np.sort(RadarValues99[:,0])
    return RadarValues99, maxOfRadarValues99, scaleFactor

def transformRadarDataToPng(radarData, RadarValues99, maxOfRadarValues99, scaleFactor):
        # scale values which are larger than maxOfRadarValues99
        radarData = np.asanyarray(radarData)
        radarData = np.where(radarData > maxOfRadarValues99,
                             ((radarData - maxOfRadarValues99) * scaleFactor + len(RadarValues99)).astype(int),
                             radarData)
        # set values that are larger than 255 to 255
        radarData[radarData > 255] = 255
        radarData[radarData < 0] = 0

        for idx in range(1, len(RadarValues99)):
            radarData[(radarData <= RadarValues99[-idx]) & (radarData > RadarValues99[-idx - 1])] = len(
                RadarValues99) - idx

        radarData[(radarData > 0) & (radarData < RadarValues99[0])] = 1
        return radarData


def save_png_grayscale_8bit(image_data, filepath, filename, factor=1):
    image_data_8bit = image_data.astype(np.uint8)
    image_data_8bit *= int(factor)
    image_data_8bit = np.asanyarray(image_data_8bit)

    # image_data_8bit = np.asanyarray(image_data_8bit)
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    full_filename = filepath + filename + ".png"
    cv2.imwrite(full_filename, image_data_8bit)
    #image = PIL.Image.fromarray(image_data_8bit, mode='L')
    #image.save(full_filename)



#if gpu and cp available:
'''
def transformRadarDataToPng(radarData, RadarValues99, maxOfRadarValues99, scaleFactor):
        # scale values which are larger than maxOfRadarValues99
        radarData = cp.asanyarray(radarData)
        radarData = cp.where(radarData > maxOfRadarValues99,
                             ((radarData - maxOfRadarValues99) * scaleFactor + len(RadarValues99)).astype(int),
                             radarData)
        # set values that are larger than 255 to 255
        radarData[radarData > 255] = 255

        for idx in range(1, len(RadarValues99)):
            radarData[(radarData <= RadarValues99[-idx]) & (radarData > RadarValues99[-idx - 1])] = len(
                RadarValues99) - idx

        radarData[(radarData > 0) & (radarData < RadarValues99[0])] = 1
        return radarData

def save_png_grayscale_8bit(image_data, filepath, filename, factor=1):
    image_data_8bit = image_data.astype(np.uint8)
    image_data_8bit *= int(factor)
    image_data_8bit = cp.asnumpy(image_data_8bit)

    # image_data_8bit = np.asanyarray(image_data_8bit)
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    full_filename = filepath + filename + ".png"
    cv2.imwrite(full_filename, image_data_8bit)
'''