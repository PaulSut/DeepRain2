import 'dart:typed_data';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:deep_rain/DataObjects/DataHolder.dart';
import 'package:deep_rain/DataObjects/ForecastListItem.dart';
import 'package:deep_rain/global/PushNotifications.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:flutter/cupertino.dart';

class DatabaseService{

  final String uid;
  DatabaseService({this.uid});

  // collection reference
  final CollectionReference ForecastCollection = Firestore.instance.collection('forecast');

  Future updateUserData(String time, String rainIntense) async{
    return await ForecastCollection.document(uid).setData({
      'time' : time,
      'rainIntense' : rainIntense,
    });
  }

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
  
  void updatePushNotificationTime() async{
    PushNotifications _pushNotifications = PushNotifications();

    //Check if the default time of pushnotification is already changed
    if(_pushNotifications.getAppLastDeviceTokenDocument() != null){
      //Delete the old setting of pushnotificationtime
      Firestore.instance.collection(_pushNotifications.getAppLastDeviceTokenDocument()).document(_pushNotifications.getDeviceToken()).delete();
    }

    //Set the new setting of pushnotificationtime
    _pushNotifications.setAppLastDeviceTokenDocument('DeviceTokens_' + _pushNotifications.getTimeBeforeWarning().inMinutes.toString() + '_min');
    final CollectionReference ForecastCollection = Firestore.instance.collection('DeviceTokens_' + _pushNotifications.getTimeBeforeWarning().inMinutes.toString() + '_min');
    await ForecastCollection.document(_pushNotifications.getDeviceToken()).setData({'token' : _pushNotifications.getDeviceToken()});
  }

  void deactivatePushNotification() async{
    PushNotifications _pushNotifications = PushNotifications();
    //Check if the default time of pushnotification is already changed
    if(_pushNotifications.getAppLastDeviceTokenDocument() != null){
      //Delete the old setting of pushnotificationtime
      Firestore.instance.collection(_pushNotifications.getAppLastDeviceTokenDocument()).document(_pushNotifications.getDeviceToken()).delete();
    }
  }

  void activatePushNotification() async{
    PushNotifications _pushNotifications = PushNotifications();
    //Check if the default time of pushnotification is already changed
    if(_pushNotifications.getAppLastDeviceTokenDocument() != null){
      //Set the setting of pushnotificationtime
      _pushNotifications.setAppLastDeviceTokenDocument('DeviceTokens_' + _pushNotifications.getTimeBeforeWarning().inMinutes.toString() + '_min');
      final CollectionReference ForecastCollection = Firestore.instance.collection('DeviceTokens_' + _pushNotifications.getTimeBeforeWarning().inMinutes.toString() + '_min');
      await ForecastCollection.document(_pushNotifications.getDeviceToken()).setData({'token' : _pushNotifications.getDeviceToken()});
    }
  }

}