import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:deep_rain/global/GlobalValues.dart';
import 'package:deep_rain/global/UIText.dart';
import 'package:deep_rain/screens/ForecastList.dart';
import 'package:deep_rain/screens/Settings.dart';
import 'package:deep_rain/screens/Loading.dart';
import 'package:deep_rain/services/PushNotification.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter/material.dart';
import 'package:deep_rain/screens/ForecastMap.dart';

//First screen is the Loadingscreen, then in navigate to MainApp
void main() => runApp(MaterialApp(
  initialRoute: '/',
  routes: {
    '/' : (context) => Loading(),
    '/MainApp' : (context) => MainApp(),
  },
));

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
  }

  //The screens for the bottomNavigiation
  int _selectedTab = 0;
  final _pageOptions = [
    ForecastList(),
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
}