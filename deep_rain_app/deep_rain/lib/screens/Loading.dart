import 'dart:async';
import 'package:deep_rain/global/GlobalValues.dart';
import 'package:deep_rain/main.dart';
import 'package:deep_rain/services/database.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:latlong/latlong.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:animated_text_kit/animated_text_kit.dart';

//Is called as first screen at appstart. All forecast images get downloaded from firebase. All app settings from shared preferences will be set.

class Loading extends StatefulWidget {
  @override
  _LoadingState createState() => _LoadingState();
}

class _LoadingState extends State<Loading> {
  //Download images. Set settings.
  void setupApp() async{

    //The Images will be downloaded
    DatabaseService instance = DatabaseService();
    for(var i = 1; i <= 20; i++){
      await instance.getImage(i);
    }

    // The local stored settings will be set again.
    final prefs = await SharedPreferences.getInstance();
    GlobalValues _globalValues = GlobalValues();

    _globalValues.setAppLanguage(prefs.getString('AppLanguage'));
    _globalValues.setAppLastDeviceTokenDocument(prefs.getString('AppLastDeviceTokenDocument'));
    _globalValues.setDeviceToken(prefs.getString('AppDeviceToken'));
    _globalValues.setTimeBeforeWarning(Duration(minutes: prefs.getInt('AppTimeBeforeWarning')));
    _globalValues.setAppSwitchRainWarning(prefs.getBool('AppSwitchRainWarning'));
    _globalValues.setAppRegionCity(prefs.getString('AppRegionCity'));
    _globalValues.setAppRegion(new LatLng(prefs.getDouble('AppRegionLatitude'), prefs.getDouble('AppRegionLongitude')));

  }

  @override
  void initState(){
    super.initState();
    setupApp();
  }

  @override
  Widget build(BuildContext context) {
    //Stay 2,5 seconds on the loadingscreen.
    Timer(
        Duration(seconds: 3, milliseconds: 500),
        () => Navigator.of(context).pushReplacement(MaterialPageRoute(builder: (BuildContext context) => MainApp()))
    );

    return Scaffold(
      backgroundColor: Colors.blueGrey,
      body: Center(
        child: TypewriterAnimatedTextKit(
            totalRepeatCount: 4,
            pause: Duration(milliseconds:  1000),
            text: ["deepRain", "stay dry", "HTWG Konstanz"],
            textStyle: TextStyle(fontSize: 32.0, fontWeight: FontWeight.bold),
            displayFullTextOnTap: true,
            stopPauseOnTap: true
        ),
        /*SpinKitRotatingPlain(
          color: Colors.white,
          size: 50.0,
        ),*/
      ),
    );
  }
}
