import wradlib as wrl
import numpy as np
import tarfile
import os
import pickle
import matplotlib.pyplot as plt
import argparse
import os
import sys
import wradlib as wrl
import numpy as np
import warnings
import csv
import cv2

def get_timestamp_for_bin_filename(bin_file_name):
    split = bin_file_name.split('-')
    timestamp = split[2]
    return timestamp

def read_radolan(radfile):
    return wrl.io.read_radolan_composite(radfile)

# Array-Like, max of all data
def normalize(data, absolute_max):
    factor = float(254)/absolute_max
    data_scaled = []
    for row in data:
        scaled_row = []
        for pixel in row:
            if pixel == 0:
                scaled_row.append(pixel)
            else:
                scaled_row.append(pixel*factor+1)
        data_scaled.append(scaled_row)

    return np.asanyarray(data_scaled)

def save_png_grayscale_8bit(image_data, filename, factor=1):
    image_data_8bit = image_data.astype(np.uint8)
    image_data_8bit *= int(factor)
    full_filename = filename + ".png"
    cv2.imwrite(full_filename, image_data_8bit)
    print("Saved image file: " + full_filename)


if __name__ == '__main__':

    abs_path = os.path.abspath('.')
    path_of_radar_zip = abs_path + "/Data/RadDataZIP"
    list_of_zips = os.listdir(path_of_radar_zip)

    '''
        tar_year = tarfile.open(path_of_radar_zip + "/YW2017.002_201606.tar", "r:")
    tar_year.extractall(abs_path + "/Data/RadDataZipMonth")
    list_month = os.listdir(abs_path + "/Data/RadDataZipMonth")
    for month in list_month:
        tar_month = tarfile.open(abs_path + "/Data/RadDataZipMonth/" + month, "r:gz")
        tar_month.extractall(abs_path + "/Data/RadData")
    '''


    total_files = 0
    counter = 0
    out_dir = abs_path + "/Data/pngDATA"

    for subdir, dirs, files in os.walk(abs_path + "/Data/RadData"):
        total_files += len(files)
        for file in files:
            image_file_path = out_dir + '/' + "scaled_" + get_timestamp_for_bin_filename(file)
            if '.png' in file:
                print("Skipping png (" + str(counter) + '/' + str(len(files)) + ")")
                total_files -= 1
                continue
            if os.path.isfile(image_file_path + ".png"):
                total_files -= 1
                continue
            data, attrs = read_radolan(subdir + '/' + file)

            data = normalize(data, 5)
            print(np.max(data))
            save_png_grayscale_8bit(data, image_file_path, factor=1)
            counter += 1