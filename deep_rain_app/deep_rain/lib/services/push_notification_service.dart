import 'package:deep_rain/screens/ForecastMap.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'dart:io';
import 'package:flutter/material.dart';

class PushNotificationService{
  final FirebaseMessaging _fcm = FirebaseMessaging();

  Future initialise() async{

    _fcm.getToken().then((String token) => {
      print('Das ist ein Token: $token')
    });

    if(Platform.isIOS){
      _fcm.requestNotificationPermissions(IosNotificationSettings());
    }

    _fcm.configure(
      //called wehn app is in the foreground and we recieve a push notification
      onMessage: (Map<String, dynamic> message) async {
        print('onMessage: $message');
      },
      //
      onResume: (Map<String, dynamic> message) async{
        print('onResume: $message');
      },
      onLaunch: (Map<String, dynamic> message) async{
        print('onLaunch: $message');
      }
    );
  }

  void _serialiseAndNavigate(Map<String, dynamic> message){
    var notificationData = message['data'];
    var view = notificationData['view'];

    if(view != null){
      if(view == 'eins'){
        //Hier muss navigiert werden
      }
    }
  }
}