import pickle

import firebase_admin
from firebase_admin import credentials, firestore, storage
from datetime import datetime
from PIL import Image
import numpy as np

#
# upload the forecast rainintense to firebase for each region
#

if __name__ == '__main__':
    #Never, ever upload this Certificate file to git
    cred = credentials.Certificate('./deeprain-firebase-adminsdk-xpcbj-bcbc99b37e.json')
    default_app = firebase_admin.initialize_app(cred, {'storageBucket': 'deeprain.appspot.com'})
    bucket = storage.bucket()
    db = firestore.client()

    #get the lists with coordinate data
    f = open('listLatitudeComplete.pckl', 'rb')
    listLatitude = pickle.load(f)
    f.close()

    f = open('listLongitudeComplete.pckl', 'rb')
    listLongitude = pickle.load(f)
    f.close()

    f = open('listCoordinates.pckl', 'rb')
    listCoordinates = pickle.load(f)
    f.close()

    # Counter for the ID of Data
    ID = 0

    #list for all latitudes and longitudes which are already calulated
    #[latitude, longitude, pixels]
    latitude_longitude_pixels = []

    #return the cordinates of the pixel by [y (lng), x (lat)]
    #TODO es kommen bei einem 900x900 grid natürlich andere Pixel raus als bei 1100x900, aber sind diese Werte richtig?
    def return_pixel_from_coordinates(latitude, longitude):
        #check, if the pixel coordinates for this latitude and longitude is already calculated
        for latitude_index in range(len(latitude_longitude_pixels)):
            if latitude_longitude_pixels[latitude_index][0] == latitude:
                if latitude_longitude_pixels[latitude_index][1] == longitude:
                    print('Found the return value in List')
                    return latitude_longitude_pixels[latitude_index][2]

        dist_to_pixel = []
        for idx in range(len(listLatitude)):
            dist_to_pixel.append(np.linalg.norm([latitude - listLatitude[idx], longitude - listLongitude[idx]]))
        index_of_min_dist = dist_to_pixel.index(min(dist_to_pixel))

        pixel_coordinates = listCoordinates[index_of_min_dist]

        latitude_longitude_pixels.append([latitude, longitude, pixel_coordinates])

        return pixel_coordinates

    def return_rain_intense_from_forecast_by_latlng(latitude, longitude, image):

        #get the pixel coordinate for this long and latitude in format [y,x]
        pixel_cordinate = return_pixel_from_coordinates(latitude, longitude)

        print(pixel_cordinate)

        #get the value of the pixel
        rain_intense = image.getpixel((pixel_cordinate[1], pixel_cordinate[0]))

        return rain_intense


    def delete_collection(coll_ref):
        docs = coll_ref.stream()

        for doc in docs:
            print(u'Deleting doc {} => {}'.format(doc.id, doc.to_dict()))
            doc.reference.delete()

    #def upload_data_to_firbase(image0, image1, image2, image3, image4, image5, image6, image7, image8, image9, image10, image11, image12, image13, image14, image15, image16, image17, image18, image19):
    def upload_data_to_firbase(forecast_images):
        # all regions where are users
        regions = db.collection(u'Regions').stream()
        regions = list(regions)
        # For each region where are users, (Device Tokens in the Regions/Region/tokens collection), a forecast need to be pushed
        for region in regions:
            global ID

            #TODO hier muss die tatsächliche Zeit genommen werden, die zu der Regenvorhersage passt
            # the current time for firebase document ID
            now = datetime.now()
            current_time = now.strftime("%H:%M")

            # # the unique id for the documents of database.
            # documentID = 'deeprain_' + current_time + '_' + str(ID)

            # random dummy rainintense
            #rainIntense = randrange(100)
            #needed for pushnotification tests
            #rainIntense = 94

            #needed for test of real dataflow

            #get the latitude and longitude of the current region
            region_lat_lng = region.to_dict()
            region_latitude = region_lat_lng['Latitude']
            region_longitude = region_lat_lng['Longitude']

            #delete the old data
            delete_collection(db.collection('Regions').document(region.id).collection('forecast'))

            is_pushnotification_sended = False

            #calculate for each image the rain intense and load it to firebase
            for image in forecast_images:
                #TODO hier sollte immer die Uhrzeit des aktuellen Bildes eingesetzt werden
                # the unique id for the documents of database.
                documentID = 'deeprain_' + current_time + '_' + str(ID)

                rainIntense = return_rain_intense_from_forecast_by_latlng(region_latitude, region_longitude, image)

                #upload the forecast data to firebase
                doc_ref = db.collection('Regions').document(region.id).collection('forecast').document(str(documentID))
                doc_ref.set({
                    'rainIntense': rainIntense,
                    'time': current_time
                })

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
            print('Upload erfolgeich. ID:' + str(ID) + '. rainIntense: ' + str(rainIntense) + '. time: ' + str(current_time))

            #db.collection('Regions').document(region.id).collection('forecast').document(forecasts[0].id).delete()

    image0 = Image.open("assets/1701020100.png")
    image1 = Image.open("assets/1701020105.png")
    image2 = Image.open("assets/1701020110.png")
    image3 = Image.open("assets/1701020115.png")
    image4 = Image.open("assets/1701020120.png")
    image5 = Image.open("assets/1701020125.png")
    image6 = Image.open("assets/1701020130.png")
    image7 = Image.open("assets/1701020135.png")
    image8 = Image.open("assets/1701020140.png")
    image9 = Image.open("assets/1701020145.png")
    image90 = Image.open("assets/1701020150.png")
    image91 = Image.open("assets/1701020155.png")
    image92 = Image.open("assets/1701020200.png")
    image93 = Image.open("assets/1701020205.png")
    image94 = Image.open("assets/1701020210.png")
    image95 = Image.open("assets/1701020215.png")
    image96 = Image.open("assets/1701020220.png")
    image97 = Image.open("assets/1701020225.png")
    image98 = Image.open("assets/1701020230.png")
    image99 = Image.open("assets/1701020235.png")

    forecast_images = [
        image0,
        image1,
        image2,
        image3,
        image4,
        image5,
        image6,
        image7,
        image8,
        image9,
        image90,
        image91,
        image92,
        image93,
        image94,
        image95,
        image96,
        image97,
        image98,
        image99,
    ]

    upload_data_to_firbase(forecast_images)



