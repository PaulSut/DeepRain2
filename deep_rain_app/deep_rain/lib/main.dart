import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:deep_rain/global/PushNotifications.dart';
import 'package:deep_rain/global/UIText.dart';
import 'package:deep_rain/screens/ForecastList.dart';
import 'package:deep_rain/screens/ImageGrid.dart';
import 'package:deep_rain/screens/Settings.dart';
import 'package:deep_rain/screens/loading.dart';
import 'package:deep_rain/services/push_notification_service.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter/material.dart';
import 'package:deep_rain/screens/ForecastMap.dart';
import 'package:firebase_database/firebase_database.dart';


void main() => runApp(MaterialApp(
  initialRoute: '/',
  routes: {
    '/' : (context) => Loading(),
    '/MainApp' : (context) => MainApp(),
  },
));

//void main() => runApp(new MainApp());
class MainApp extends StatefulWidget {
  @override
  State<StatefulWidget> createState() {
    return MainAppState();
  }
}
class MainAppState extends State<MainApp> {

  UIText _uiText;
  MainAppState() {
    _uiText = UIText();
    final PushNotificationService _pushNotificationService = PushNotificationService();
    _pushNotificationService.initialise();

    //The Devicetoken is needed to send Pushnotifications.
    //If the device token of the current Device already exists in the DataBase, it will be overwritten.
    //If not, it will be added.
    pushDeviceTokenToDB();

  }

  int _selectedTab = 0;
  final _pageOptions = [
    ForecastList(),
    //ImagesScreen(),
    ForecastMap(),
    Settings(),
  ];


  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: ThemeData(
          primarySwatch: Colors.blueGrey,
          primaryTextTheme: TextTheme(
            title: TextStyle(color: Colors.white),
          )),
      home: Scaffold(
        body: _pageOptions[_selectedTab],
        bottomNavigationBar: BottomNavigationBar(
          currentIndex: _selectedTab,
          onTap: (int index) {
            setState(() {
              _selectedTab = index;
            });
          },
          items: [
            BottomNavigationBarItem(
              icon: Icon(Icons.view_quilt),
              title: Text(_uiText.List),
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.map),
              title: Text(_uiText.Map),
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.settings),
              title: Text(_uiText.Settings),
            ),
          ],
        ),
      ),
    );
  }

  final FirebaseMessaging _fcm = FirebaseMessaging();
  PushNotifications _pushNotifications = PushNotifications();

  pushDeviceTokenToDB() async {
    final CollectionReference ForecastCollection = Firestore.instance.collection('DeviceTokens');
    await _fcm.getToken().then((token) async{
      await ForecastCollection.document(token).setData({'token' : token});
      _pushNotifications.setDeviceToken(token);
    });
  }
}