import 'dart:typed_data';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:deep_rain/DataObjects/DataHolder.dart';
import 'package:deep_rain/DataObjects/ForecastListItem.dart';
import 'package:deep_rain/global/GlobalValues.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:flutter/cupertino.dart' as imageloader;
import 'package:flutter/material.dart';
import 'package:image/image.dart' as image_libary;
import 'dart:io' as dart_io;
import 'dart:convert';
import 'dart:async' show Future;
import 'package:flutter/services.dart' show rootBundle;

/*
The communication with the firesbase is handled in this class
 */

class DatabaseService{
  final String uid;
  DatabaseService({this.uid});
  GlobalValues _globalValues = GlobalValues();

  // collection reference for the forecast data
  //final CollectionReference ForecastCollection = Firestore.instance.collection('forecast');
  final CollectionReference ForecastCollection = Firestore.instance.collection('Regions').document(AppRegionCity).collection('forecast');

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

  //collection reference for the timestep data (which will be shown in the label of the slider in forecastmap)
  final CollectionReference TimeStepCollection = Firestore.instance.collection('TimeSteps');
  // timestep list from snapshot
  List<String> _timeStepListFromSnapshot(QuerySnapshot snapshot){
    return snapshot.documents.map((doc){

      return doc.data['time'].toString();
    }).toList();
  }

  Stream<List<String>> get TimeSteps{
    return TimeStepCollection.snapshots()
        .map(_timeStepListFromSnapshot);
  }

  //if the image is not already stored in the DataHolder, it will be downloaded from firebase
  Future<int> getImage(int division) async{
    StorageReference photosReference =  await FirebaseStorage.instance.ref().child('photos');

    int pixel_value = 0;
    if (!requestedIndexes.contains(division)) {
      print('Ich bin hier' + division.toString());
      int MAX_SIZE = 7 * 1024 * 1024;
      await photosReference.child('$division.png').getData(MAX_SIZE).then((data) async{
        requestedIndexes.add(division);
        pixel_value = await calculate_pixel_value(data, division);
        imageData.putIfAbsent(division, (){
          return data;
        });
        return data;
      }).catchError((onError) {
        imageloader.debugPrint(onError.toString());
      });
    }
    return pixel_value;
  }

  Future<int> calculate_pixel_value(data, division) async{
    var pixel_of_current_location = await _globalValues.getAppPixel();
    print('Current App Pixel: ' + pixel_of_current_location.toString());
    //Example for
    int y = 900 - pixel_of_current_location[1];
    int x = 900 - pixel_of_current_location[0];
    image_libary.Image image = image_libary.decodeImage(data);
    int pixel = image.getPixel(y, x);
    print('Pixel value: ' + pixel.toString() + ' In division: ' + division.toString());
    return pixel;
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

  Future<void> updateRegion() async {
    GlobalValues _globalValues = GlobalValues();

    //Check if the default region is already changed
    if(_globalValues.getAppLastRegionDocument() != null){
      //Delete the old setting of region
      Firestore.instance.collection('Regions').document(_globalValues.getAppLastRegionDocument()).collection('tokens').document(_globalValues.getDeviceToken()).delete();

      //check if there are still some devicetokens in the old region. if not, delete the document.
      QuerySnapshot querySnapshot = await Firestore.instance.collection("Regions").document(_globalValues.getAppLastRegionDocument()).collection('tokens').getDocuments();
      var list = querySnapshot.documents;
      if(list.length == 0){
        Firestore.instance.collection('Regions').document(_globalValues.getAppLastRegionDocument()).delete();
      }
     }

    //Set the new setting of region
    if(_globalValues.getAppLastRegionDocument() != null){
      String newCity = _globalValues.getAppRegionCity();
      String deviceToken = _globalValues.getDeviceToken();

      //Store push region in firebase
      final CollectionReference collectionReference = Firestore.instance.collection('Regions');
      collectionReference.document(newCity).setData({'Latitude' : _globalValues.getAppRegion().latitude, 'Longitude' : _globalValues.getAppRegion().longitude});
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
      Firestore.instance.collection('Regions').document('Konstanz').setData({'Latitude' : _globalValues.getAppRegion().latitude, 'Longitude' : _globalValues.getAppRegion().longitude});

      //need to be stored local to update or delete it later on
      _globalValues.setAppLastRegionDocument('Konstanz');
    }else{
      Firestore.instance.collection('Regions').document(_globalValues.getAppLastRegionDocument()).collection('tokens').document(_globalValues.getDeviceToken()).setData({'token' : _globalValues.getDeviceToken()});
      Firestore.instance.collection('Regions').document(_globalValues.getAppLastRegionDocument()).setData({'Latitude' : _globalValues.getAppRegion().latitude, 'Longitude' : _globalValues.getAppRegion().longitude});
    }
  }


}