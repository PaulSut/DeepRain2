import pickle
import firebase_admin
from firebase_admin import credentials, firestore, storage
from datetime import datetime
from PIL import Image
import numpy as np
import time
import os

# list for all latitudes and longitudes which are already calulated
# [latitude, longitude, pixels]
latitude_longitude_pixels = {}

#return the cordinates of the pixel by [y (lng), x (lat)]
#TODO es kommen bei einem 900x900 grid natürlich andere Pixel raus als bei 1100x900, aber sind diese Werte richtig?
def return_pixel_from_coordinates(latitude, longitude, coordinate_lists):
    global latitude_longitude_pixels

    listLatitude = coordinate_lists[0]
    listLongitude = coordinate_lists[1]
    listCoordinates = coordinate_lists[2]

    if str(latitude + longitude) in latitude_longitude_pixels:
        return latitude_longitude_pixels[str(latitude+longitude)]

    dist_to_pixel = []
    for idx in range(len(listLatitude)):
        dist_to_pixel.append(np.linalg.norm([latitude - listLatitude[idx], longitude - listLongitude[idx]]))
    index_of_min_dist = dist_to_pixel.index(min(dist_to_pixel))

    pixel_coordinates = listCoordinates[index_of_min_dist]
    latitude_longitude_pixels[str(latitude + longitude)] = pixel_coordinates

    return pixel_coordinates

def return_rain_intense_from_forecast_by_latlng(latitude, longitude, image, coordinate_lists):

    #get the pixel coordinate for this long and latitude in format [y,x]
    pixel_cordinate = return_pixel_from_coordinates(latitude, longitude, coordinate_lists)

    #get the value of the pixel
    rain_intense = image[pixel_cordinate[0], pixel_cordinate[1]]

    return rain_intense

def upload_time_steps(time_steps, firestore_client):
    # unique, sortable ID for the time steps
    abc = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't']

    time_collection = firestore_client.collection('TimeSteps')
    id = 0
    for time in time_steps:
        doc_ref = time_collection.document(abc[id])
        doc_ref.set({
            'time': time
        })
        id = id + 1

def upload_data_to_firbase(forecast_images, time_of_forecasts, coordinate_lists):
    # Never, ever upload this Certificate file to git
    cred = credentials.Certificate(os.path.join(os.path.abspath(os.getcwd()), 'Utils/Workflow/ServiceAccountKey.json'))
    #cred = credentials.Certificate('./deeprain-firebase-adminsdk-xpcbj-bcbc99b37e.json')
    default_app = firebase_admin.initialize_app(cred, {'storageBucket': 'deeprain.appspot.com'})
    bucket = storage.bucket()
    db = firestore.client()

    # list for all latitudes and longitudes which are already calulated
    # [latitude, longitude, pixels]
    global latitude_longitude_pixels
    f = open('Utils/Workflow/latitude_longitude_pixels.pckl', 'rb')
    latitude_longitude_pixels = pickle.load(f)
    f.close()

    # Counter for the ID of Data
    ID = 0

    # all regions where are users
    regions = db.collection(u'Regions').stream()
    regions = list(regions)

    # For each region where are users, (Device Tokens in the Regions/Region/tokens collection), it need to be checked, if a push notification need to be send
    for region in regions:
        ID = 0
        #get the latitude and longitude of the current region
        region_lat_lng = region.to_dict()
        region_latitude = region_lat_lng['Latitude']
        region_longitude = region_lat_lng['Longitude']

        is_pushnotification_sended = False

        print('Start with: ', region.id)

        formatted_time_steps = []

        #calculate for each image the rain intense and send push notification if it is needed
        for image in range(len(forecast_images)):
            # the unique id for the documents of database.
            documentID = 'deeprain_' + time_of_forecasts[image] + '_' + str(ID)

            rainIntense = return_rain_intense_from_forecast_by_latlng(region_latitude, region_longitude, forecast_images[image], coordinate_lists)

            formatted_time_steps.append(str(time_of_forecasts[image][-4:-2] + ":" + time_of_forecasts[image][-2:]))

            ID = ID +1

            # send push notifications to devices.
            if is_pushnotification_sended == False:
                if(rainIntense > 90):
                    doc_ref = db.collection('RainWarningPushNotification').document(str(documentID))
                    doc_ref.set({
                        'title': 'Es gibt eine Regenwarnung!',
                        'body': 'Nehmen Sie besser Ihren Regenschirm mit, es wird in 30 Minuten regenen!',
                        'time_before_raining': '30',
                        'region': region.id
                    })
                    is_pushnotification_sended = True
        is_pushnotification_sended == False

        # the time which will be used for the slider.
        upload_time_steps(formatted_time_steps, db)

    #store the already calculated latitude longitude pixel context
    with open('Utils/Workflow/latitude_longitude_pixels.pckl', 'wb') as f:
        pickle.dump(latitude_longitude_pixels, f)

    firebase_admin.delete_app(default_app)