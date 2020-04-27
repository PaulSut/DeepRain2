import pickle

import firebase_admin
from firebase_admin import credentials, firestore, storage
import sched
import time
from random import randrange
from datetime import datetime
from PIL import Image

#
# creats example rain forecast data. upload this data to firebase. every 5 minutes one dataset.
#

if __name__ == '__main__':
    #Never, ever upload this Certificate file to git
    cred = credentials.Certificate('./deeprain-firebase-adminsdk-xpcbj-bcbc99b37e.json')
    default_app = firebase_admin.initialize_app(cred, {'storageBucket': 'deeprain.appspot.com'})
    bucket = storage.bucket()
    db = firestore.client()

    #to upload new forecast data from Time to time
    s = sched.scheduler(time.time, time.sleep)

    # Counter for the ID of Data
    ID = 0

    #return the cordinates of the pixel by [x (lat), y (lng)]
    def return_pixel_from_coordinates(latitude, longitude):
        f = open('listLatitude.pckl', 'rb')
        listLatitude = pickle.load(f)
        f.close()

        f = open('listLongitude.pckl', 'rb')
        listLongitude = pickle.load(f)
        f.close()

        minLat = 0;
        lastLatValue = 0;
        currentLatMin = 1000;

        for var in range(1100):
            lastLatValue = abs(listLatitude[var] - latitude)
            if (lastLatValue < currentLatMin):
                currentLatMin = lastLatValue;
                minLat = var;

        minLng = 0;
        lastLngValue = 0;
        currentLngMin = 1000;
        for var in range(900):
            lastLngValue = abs(listLongitude[var] - longitude)
            if (lastLngValue < currentLngMin):
                currentLngMin = lastLngValue;
                minLng = var;

        return [minLat, minLng]

    def return_rain_intense_from_forecast_by_latlng(latitude, longitude):
        #get the pixel coordinate for this long and latitude
        pixel_cordinate = return_pixel_from_coordinates(latitude, longitude)

        #at a later point, this will be the last forecast image which was generated
        img = Image.open("assets/example_forecast.png")
        #get the value of the pixel
        rain_intense = img.getpixel((pixel_cordinate[0], pixel_cordinate[1]))

        return rain_intense;

    def upload_data_to_firbase(sc):
        # all regions where are users
        regions = db.collection(u'Regions').stream()
        regions = list(regions)

        # For each region where are users, (Device Tokens in the Regions/Region/tokens collection), a forecast need to be pushed
        for region in regions:
            global ID
            # the current time for firebase document ID
            now = datetime.now()
            current_time = now.strftime("%H:%M")

            # the unique id for the documents of database.
            documentID = 'deeprain_' + current_time + '_' + str(ID)

            # random dummy rainintense
            rainIntense = randrange(100)

            #needed for pushnotification tests
            rainIntense = 94

            #needed for test of real dataflow
            #rainIntense = return_rain_intense_from_forecast_by_latlng(47.66033, 9.17582)
            #print(rainIntense)

            # upload forecast to firebase
            #doc_ref = db.collection('forecast').document(str(documentID))
            #doc_ref.set({
            #    'rainIntense': rainIntense,
            #    'time': current_time
            #})

            doc_ref = db.collection('Regions').document(region.id).collection('forecast').document(str(documentID))
            doc_ref.set({
                'rainIntense': rainIntense,
                'time': current_time
            })

            for doc in regions:
                print(u'{} => {}'.format(doc.id, doc.to_dict()))

            # send push notifications to devices.
            if(rainIntense > 90):
                doc_ref = db.collection('RainWarningPushNotification').document(str(documentID))
                doc_ref.set({
                    'title': 'Es gibt eine Regenwarnung!',
                    'body': 'Nehmen Sie besser Ihren Regenschirm mit, es wird in 30 Minuten regenen!',
                    'time_before_raining': '30',
                    'region': region.id
                })

            # increase the ID
            ID = ID + 1
            print('Upload erfolgeich. ID:' + str(ID) + '. rainIntense: ' + str(rainIntense) + '. time: ' + str(current_time))

            # all regions where are users
            forecasts = db.collection(u'Regions').document(region.id).collection('forecast').stream()
            forecasts = list(forecasts)
            number_of_forecasts = len(forecasts)
            # if there are 20 items in the db, alway remove the oldest one
            if number_of_forecasts > 19:
                # delete in firebase
                db.collection('Regions').document(region.id).collection('forecast').document(forecasts[0].id).delete()

            s.enter(10, 1, upload_data_to_firbase, (sc,))
    s.enter(10, 1, upload_data_to_firbase, (s,))
    s.run()



