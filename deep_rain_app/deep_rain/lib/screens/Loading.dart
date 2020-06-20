import 'dart:async';
import 'dart:convert';
import 'dart:ffi';
import 'package:animated_text_kit/animated_text_kit.dart';
import 'package:deep_rain/global/GlobalValues.dart';
import 'package:deep_rain/main.dart';
import 'package:deep_rain/services/Database.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:latlong/latlong.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'dart:async' show Future;
import 'package:flutter/services.dart' show rootBundle;
import 'dart:io' show File;
import 'dart:typed_data';

//Is called as first screen at appstart. All forecast images get downloaded from firebase. All app settings from shared preferences will be set.

class Loading extends StatefulWidget {
  @override
  _LoadingState createState() => _LoadingState();
}

class _LoadingState extends State<Loading> {

  Future<String> loadListCoordinates() async {
    final ByteData data = await rootBundle.load('assets/data/listCoordinates.json');
    String jsonContent = utf8.decode(data.buffer.asUint8List());
    return jsonContent;

  }
  Future<String> loadListLatitude() async {
    final ByteData data = await rootBundle.load('assets/data/listLatitudeComplete.json');
    String jsonContent = utf8.decode(data.buffer.asUint8List());
    return jsonContent;
  }
  Future<String> loadListLongitude() async {
    final ByteData data = await rootBundle.load('assets/data/listLongitudeComplete.json');
    String jsonContent = utf8.decode(data.buffer.asUint8List());
    return jsonContent;
  }

  //Download images. Set settings.
  Future<bool> setupApp() async{
    // The local stored settings will be set again.
    final prefs = await SharedPreferences.getInstance();
    GlobalValues _globalValues = GlobalValues();
    await FirebaseAuth.instance.signInAnonymously();

    //The Images will be downloaded
    DatabaseService instance = DatabaseService();

    if(_globalValues.getAppLastRegionDocument() == null){
      instance.storeRegion();
      _globalValues.setAppLastRegionDocument('Konstanz');
    } else{
      _globalValues.setAppLastRegionDocument(_globalValues.getAppLastRegionDocument());
    }

    _globalValues.setAppLanguage(prefs.getString('AppLanguage') == null ? 'Deutsch' : prefs.getString('AppLanguage'));
    _globalValues.setAppLastDeviceTokenDocument(prefs.getString('AppLastDeviceTokenDocument') == null ? '20_min' : prefs.getString('AppLastDeviceTokenDocument'));
    _globalValues.setDeviceToken(prefs.getString('AppDeviceToken'));
    _globalValues.setAppSwitchRainWarning(prefs.getBool('AppSwitchRainWarning'));
    _globalValues.setAppRegionCity(prefs.getString('AppRegionCity') == null ? 'Konstanz' : prefs.getString('AppRegionCity'));

    int minutes = prefs.getInt('AppTimeBeforeWarning');
    Duration _duration = Duration(minutes: minutes == null ? 20 : minutes);
    _globalValues.setTimeBeforeWarning(_duration);

    instance.activatePushNotification();

    double latitude = prefs.getDouble('AppRegionLatitude');
    double longitude = prefs.getDouble('AppRegionLongitude');
    LatLng _latLng = LatLng(latitude == null ? 47.66033 : latitude, longitude == null ? 9.17582 : longitude);
    _globalValues.setAppRegion(_latLng);

    int appPixel_x = prefs.getInt('AppPixel_X');
    int appPixel_y = prefs.getInt('AppPixel_Y');
    if(appPixel_y == null && appPixel_x == null){
      print('die Pixel waren null');
      await _globalValues.getAppPixel();
    }else{
      await _globalValues.setAppPixel([appPixel_x, appPixel_y]);
    }


    for(var i = 1; i <= 20; i++){
      print('Ich hole Bilder');
      await instance.getImage(i);
    }

    final FirebaseMessaging _fcm = FirebaseMessaging();
    await _fcm.getToken().then((token) async{
      _globalValues.setDeviceToken(token);
    });

    Navigator.of(context).pushReplacement(MaterialPageRoute(builder: (BuildContext context) => MainApp()));
  }

  @override
  Future<void> initState() {
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    return FutureBuilder(
      future: setupApp(),
      builder: (context, snap){
        return  Scaffold(
          backgroundColor: Colors.blueGrey,
          body: Center(
            child: TypewriterAnimatedTextKit(
              totalRepeatCount: 1,
              pause: Duration(milliseconds:  1000),
              text: ["deepRain", "stay dry", "HTWG Konstanz"],
              textStyle: TextStyle(fontSize: 32.0, fontWeight: FontWeight.bold),
              displayFullTextOnTap: true,
              stopPauseOnTap: true,
              onFinished:(){
                //Navigator.of(context).pushReplacement(MaterialPageRoute(builder: (BuildContext context) => MainApp()));
          },
          ),/*
            SpinKitRotatingPlain(
              color: Colors.white,
              size: 50.0,
            ),*/
          ),
        );
      },
    );
  }
}
