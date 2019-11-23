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
import logging



if __name__ == '__main__':
    total_files = 0
    counter = 0
    abs_min, abs_max = query_metadata_file(metadata_file)

    for subdir, dirs, files in os.walk(os.environ["WRADLIB_DATA"]):
        total_files += len(files)
        for file in files:
            image_file_path = out_dir + '/' + "scaled_" + get_timestamp_for_bin_filename(file)
            if '.png' in file:
                logger.info("Skipping png (" + str(counter) + '/' + str(len(files)) + ")")
                total_files -= 1
                continue
            if os.path.isfile(image_file_path + ".png"):
                total_files -= 1
                continue
            data, attrs = read_radolan(subdir + '/' + file)

            data = normalize(data, abs_max)
            save_png_grayscale_8bit(data, image_file_path, factor)
            counter += 1