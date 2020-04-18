import 'package:firebase_messaging/firebase_messaging.dart';
import 'dart:io';

/*
Handle the pushnotifications
 */
class PushNotificationService{
  final FirebaseMessaging _fcm = FirebaseMessaging();

  Future initialise() async{
    _fcm.getToken().then((String token) => {
      //There is a token! (Devicetoken)
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
}