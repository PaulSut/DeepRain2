import numpy as np
import os
import cv2
import PIL
import shutil


def create_rain_image(prediction, prediction_rgba, target_dim, image_pos):
    LIGHT_RAININTENSITY = 2
    MEDIUM_RAININTENSITY = 10

    TRANSPARENT = [0, 0, 0, 0]
    COLOR_LIGHT = [125, 150, 230, 175]
    COLOR_MEDIUM = [70, 100, 235, 175]
    COLOR_STRONG = [45, 5, 255, 175]

    img_hight = len(prediction)
    img_width = len(prediction[0])

    prediction = np.reshape(prediction, img_hight * img_width)
    prediction_rgba = np.reshape(prediction_rgba, (img_hight * img_width, 4))
    prediction_rgba_copy = prediction_rgba.copy()

    transparent_index = np.where(prediction == 0)[0].tolist()
    light_index = np.where((prediction <= LIGHT_RAININTENSITY) & (prediction != 0))[0].tolist()
    medium_index = np.where((prediction <= MEDIUM_RAININTENSITY) & (prediction > LIGHT_RAININTENSITY))[0].tolist()
    strong_index = np.where(prediction > MEDIUM_RAININTENSITY)[0].tolist()

    prediction_rgba_copy[transparent_index] = TRANSPARENT
    prediction_rgba_copy[light_index] = COLOR_LIGHT
    prediction_rgba_copy[medium_index] = COLOR_MEDIUM
    prediction_rgba_copy[strong_index] = COLOR_STRONG

    prediction_rgba_copy = np.reshape(prediction_rgba_copy, (img_hight, img_width, 4))

    final_image = fit_image_to_map(target_dim, image_pos, prediction_rgba_copy)
    #print(final_image.shape)
    #print(prediction_rgba_copy.shape)
    # Flip Picture
    final_image = np.flip(final_image, axis=0)
    #final_image = np.flip(final_image, axis=1)

    return final_image

def create_rain_intensity_values(prediction, target_dimension, image_pos):
    TRANSPARENT = 0

    above_y = target_dimension[0] - (target_dimension[0] - image_pos[0])
    above_x = target_dimension[1]

    left_y = prediction.shape[0]
    left_x = target_dimension[1] - (target_dimension[1] - image_pos[1])

    right_y = prediction.shape[0]
    right_x = target_dimension[1] - (image_pos[1] + prediction.shape[1])

    below_y = target_dimension[0] - (image_pos[0] + prediction.shape[0])
    below_x = target_dimension[1]

    above_image = np.full((above_y, above_x), TRANSPARENT)
    left_of_image = np.full((left_y, left_x), TRANSPARENT)
    right_of_image = np.full((right_y, right_x), TRANSPARENT)
    below_of_image = np.full((below_y, below_x), TRANSPARENT)

    middle_section_with_image = np.concatenate((left_of_image, prediction, right_of_image), axis=1)
    rain_intensity_values = np.asanyarray(np.concatenate((above_image, middle_section_with_image, below_of_image)),
                                  dtype=np.uint8)

    rain_intensity_values = np.reshape(rain_intensity_values, (target_dimension[0], target_dimension[1]))

    rain_intensity_values_img = PIL.Image.fromarray(rain_intensity_values)
    rain_intensity_values_img_final = np.asanyarray(rain_intensity_values_img.resize((900,900)))

    return rain_intensity_values_img_final



def fit_image_to_map(target_dimension, image_pos, image_array):
    TRANSPARENT = np.asanyarray([0, 0, 0, 0])

    above_y = target_dimension[0] - (target_dimension[0]-image_pos[0])
    above_x = target_dimension[1]

    left_y = image_array.shape[0]
    left_x = target_dimension[1] - (target_dimension[1] - image_pos[1])

    right_y = image_array.shape[0]
    right_x = target_dimension[1] - (image_pos[1] + image_array.shape[1])

    below_y = target_dimension[0] - (image_pos[0] + image_array.shape[0])
    below_x = target_dimension[1]

    above_image = np.full((above_y, above_x, 4), TRANSPARENT)
    left_of_image = np.full((left_y, left_x, 4), TRANSPARENT)
    right_of_image = np.full((right_y, right_x, 4), TRANSPARENT)
    below_of_image = np.full((below_y, below_x, 4), TRANSPARENT)

    middle_section_with_image = np.concatenate((left_of_image, image_array, right_of_image), axis=1)
    final_picture = np.asanyarray(np.concatenate((above_image, middle_section_with_image, below_of_image)), dtype=np.uint8)

    final_picture = np.reshape(final_picture, (target_dimension[0], target_dimension[1], 4))

    return final_picture

    for i in range(1, self.numClasses + 1):
        value = self.conditions[i]
        valuePrev = self.conditions[i - 1]
        idx = np.where((array <= value) & (array > valuePrev))
        classV = np.zeros((self.numClasses))
        classV[i - 1] = 1
        for x_idx, y_idx in zip(idx[0], idx[1]):
            newVector[x_idx, y_idx] = classV


def resize_images(dim, image_dir, save_dir, num_of_historical_imag, reverse):
    image_filenames = os.listdir(image_dir)
    image_filenames.sort(reverse=reverse)

    clear_save_dir(save_dir)

    for i, image in enumerate(image_filenames[:num_of_historical_imag]):
        img = np.asanyarray(PIL.Image.open(image_dir + image))
        x, y = dim
        img = cv2.resize(img, (y, x))
        img = PIL.Image.fromarray(img)
        img.save(save_dir + str(i) + '.png')


def take_slice_of_image(slices, image_dir, save_dir, num_of_historical_imag):
    image_filenames = os.listdir(image_dir)
    image_filenames.sort(reverse=True)

    clear_save_dir(save_dir)

    images = []
    for i, image in enumerate(image_filenames[:num_of_historical_imag]):
        img = np.asanyarray(PIL.Image.open(image_dir + image))
        img = img[[slice(slices[0], slices[1]), slice(slices[2], slices[3])]]
        images.append(img)
    images.reverse()
    save_images(images, save_dir)


def save_images(images, save_dir):
    for i, img in enumerate(images):
        img = PIL.Image.fromarray(img)
        img.save(save_dir + str(i) + '.png')


def clear_save_dir(save_dir):
    # clear save dir
    folder = save_dir
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
