import 'dart:typed_data';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:deep_rain/DataObjects/DataHolder.dart';
import 'package:deep_rain/DataObjects/ForecastListItem.dart';
import 'package:deep_rain/global/GlobalValues.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:flutter/cupertino.dart';

/*
The communication with the firesbase is handled in this class
 */

class DatabaseService{
  final String uid;
  DatabaseService({this.uid});

  // collection reference for the forecast data
  final CollectionReference ForecastCollection = Firestore.instance.collection('forecast');

  // forecast list from snapshot
  List<ForecastListItem> _forecastListFromSnapshot(QuerySnapshot snapshot){
    return snapshot.documents.map((doc){
      return ForecastListItem(
        time: doc.data['time'] ?? '',
        rainIntense: doc.data['rainIntense'] ?? 0
      );
    }).toList();
  }
  //get forecast stream
  Stream<List<ForecastListItem>> get Forecast{
    return ForecastCollection.snapshots()
    .map(_forecastListFromSnapshot);
  }

  //if the image is not already stored in the DataHolder, it will be downloaded from firebase
  Uint8List getImage(int division){
    StorageReference photosReference = FirebaseStorage.instance.ref().child('photos');

    if (!requestedIndexes.contains(division)) {
      int MAX_SIZE = 7 * 1024 * 1024;
      photosReference.child('$division.png').getData(MAX_SIZE).then((data){
        requestedIndexes.add(division);
        imageData.putIfAbsent(division, () {
          return data;
        });
        return data;
      }).catchError((onError) {
        debugPrint(onError.toString());
      });
    }
  }

  //If the user change the time of rain warning, this function will update the device token in the firebase.
  void updatePushNotificationTime() async{
    GlobalValues _globalValues = GlobalValues();

    //Check if the default time of pushnotification is already changed
    if(_globalValues.getAppLastDeviceTokenDocument() != null){
      //Delete the old setting of pushnotificationtime
      deactivatePushNotification();
    }
    //Set the new setting of pushnotificationtime
    if(_globalValues.getAppSwitchRainWarning()){
      activatePushNotification();
    }
  }

  //if the user deactivate the pushnotification
  void deactivatePushNotification() async{
    GlobalValues _globalValues = GlobalValues();
    //Check if the default time of pushnotification is already changed
    if(_globalValues.getAppLastDeviceTokenDocument() != null){
      //Delete the old setting of pushnotificationtime
      Firestore.instance.collection('TimeBeforeRaining').document(_globalValues.getAppLastDeviceTokenDocument()).collection('tokens').document(_globalValues.getDeviceToken()).delete();
    }
  }

  //if the user activate his pushnotification again
  void activatePushNotification() async{
    GlobalValues _globalValues = GlobalValues();
    //Check if the default time of pushnotification is already changed
    if(_globalValues.getAppLastDeviceTokenDocument() != null){
      String timeBeforeWarningDocument = _globalValues.getTimeBeforeWarning().inMinutes.toString() + '_min';
      String deviceToken = _globalValues.getDeviceToken();

      //Store push notificationtime in firebase
      final CollectionReference collectionReference = Firestore.instance.collection('TimeBeforeRaining');
      collectionReference.document(timeBeforeWarningDocument).setData({'activateDocument' : 'isAcitivated'});
      collectionReference.document(timeBeforeWarningDocument).collection('tokens').document(deviceToken).setData({'token' : deviceToken});

      //need to be stored local to update or delete it later on
      _globalValues.setAppLastDeviceTokenDocument(timeBeforeWarningDocument);
    }
  }

  void updateRegion(){
    GlobalValues _globalValues = GlobalValues();

    //Check if the default region is already changed
    if(_globalValues.getAppLastRegionDocument() != null){
      //Delete the old setting of region
      Firestore.instance.collection('Regions').document(_globalValues.getAppLastRegionDocument()).collection('tokens').document(_globalValues.getDeviceToken()).delete();
     }
    //Set the new setting of region
    if(_globalValues.getAppLastRegionDocument() != null){
      String newCity = _globalValues.getAppRegionCity();
      String deviceToken = _globalValues.getDeviceToken();

      //Store push region in firebase
      final CollectionReference collectionReference = Firestore.instance.collection('Regions');
      collectionReference.document(newCity).setData({'activateDocument' : 'isAcitivated'});
      collectionReference.document(newCity).collection('tokens').document(deviceToken).setData({'token' : deviceToken});

      //need to be stored local to update or delete it later on
      _globalValues.setAppLastRegionDocument(newCity);
    }
  }
  void storeRegion(){
    GlobalValues _globalValues =  GlobalValues();
    //Set the new setting of region
    if(_globalValues.getAppLastRegionDocument() == null){
      Firestore.instance.collection('Regions').document('Konstanz').collection('tokens').document(_globalValues.getDeviceToken()).setData({'token' : _globalValues.getDeviceToken()});
      Firestore.instance.collection('Regions').document('Konstanz').setData({'activateDocument' : 'isAcitivated'});

      //need to be stored local to update or delete it later on
      _globalValues.setAppLastRegionDocument('Konstanz');
    }else{
      Firestore.instance.collection('Regions').document(_globalValues.getAppLastRegionDocument()).collection('tokens').document(_globalValues.getDeviceToken()).setData({'token' : _globalValues.getDeviceToken()});
      Firestore.instance.collection('Regions').document(_globalValues.getAppLastRegionDocument()).setData({'activateDocument' : 'isAcitivated'});
    }
  }


}