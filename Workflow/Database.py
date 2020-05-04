import firebase_admin
import google.cloud
from firebase_admin import credentials, firestore, storage
import os



def replace_forecast_in_firebase(historical_images, forecast_images, path_to_forecast_dir):
    app, bucket = init_firebase()
    for index, image in enumerate(historical_images):
        replace_image_in_firebase(image, index, mode='historical', bucket= bucket, path_to_forecast_dir=path_to_forecast_dir)

    for index, image in enumerate(forecast_images):
        replace_image_in_firebase(image, index, mode='forecast', bucket= bucket, path_to_forecast_dir=path_to_forecast_dir)
    firebase_admin.delete_app(app)
    return True

def replace_image_in_firebase(image, index, mode, bucket, path_to_forecast_dir):

    if mode == 'historical':
        minutes = 45 - index*5
        filename = f'{index+1}'
    elif mode == 'forecast':
        minutes = (index+1)*5
        filename = f'{index+11}'
    else:
        print('ERROR in replace_image_in_firebase, mode is wrong: ',mode)

    # replace in local dir
    try:
        os.remove(path_to_forecast_dir + filename + '.png')
    except Exception as e:
        print('Warning while removing forecast picture local: ', e)

    image.save(path_to_forecast_dir + filename + '.png')

    # replace in firebase
    blob = bucket.blob('photos/'+filename+'.png')
    try:
        blob.delete()
    except Exception as e:
        print('Warning while removing forecast picture remote: ', e)

    blob = bucket.blob('photos/'+filename+'.png')
    blob.upload_from_filename(path_to_forecast_dir + filename + '.png')

    return True


def init_firebase():
    cred = credentials.Certificate('/home/paul/Documents/Master1/DeepRain_Teamproject/HomeOffice/Workflow/ServiceAccountKey.json')
    app = firebase_admin.initialize_app(cred, {'storageBucket': 'deeprain.appspot.com'
                                               })
    bucket = storage.bucket()
    return app, bucket