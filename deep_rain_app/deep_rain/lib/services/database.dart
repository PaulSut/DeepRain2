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
      Firestore.instance.collection(_globalValues.getAppLastDeviceTokenDocument()).document(_globalValues.getDeviceToken()).delete();
    }

    //Set the new setting of pushnotificationtime
    _globalValues.setAppLastDeviceTokenDocument('DeviceTokens_' + _globalValues.getTimeBeforeWarning().inMinutes.toString() + '_min');
    if(_globalValues.getAppSwitchRainWarning()){
      final CollectionReference ForecastCollection = Firestore.instance.collection('DeviceTokens_' + _globalValues.getTimeBeforeWarning().inMinutes.toString() + '_min');
      await ForecastCollection.document(_globalValues.getDeviceToken()).setData({'token' : _globalValues.getDeviceToken()});
    }
  }

  //if the user deactivate the pushnotification
  void deactivatePushNotification() async{
    GlobalValues _globalValues = GlobalValues();
    //Check if the default time of pushnotification is already changed
    if(_globalValues.getAppLastDeviceTokenDocument() != null){
      //Delete the old setting of pushnotificationtime
      Firestore.instance.collection(_globalValues.getAppLastDeviceTokenDocument()).document(_globalValues.getDeviceToken()).delete();
    }
  }

  //if the user activate his pushnotification again
  void activatePushNotification() async{
    GlobalValues _globalValues = GlobalValues();
    //Check if the default time of pushnotification is already changed
    if(_globalValues.getAppLastDeviceTokenDocument() != null){
      //Set the setting of pushnotificationtime
      _globalValues.setAppLastDeviceTokenDocument('DeviceTokens_' + _globalValues.getTimeBeforeWarning().inMinutes.toString() + '_min');
      final CollectionReference ForecastCollection = Firestore.instance.collection('DeviceTokens_' + _globalValues.getTimeBeforeWarning().inMinutes.toString() + '_min');
      await ForecastCollection.document(_globalValues.getDeviceToken()).setData({'token' : _globalValues.getDeviceToken()});
    }
  }

}