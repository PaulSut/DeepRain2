import 'dart:async';
import 'package:deep_rain/global/GlobalValues.dart';
import 'package:deep_rain/main.dart';
import 'package:deep_rain/services/database.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:latlong/latlong.dart';
import 'package:shared_preferences/shared_preferences.dart';

class Loading extends StatefulWidget {
  @override
  _LoadingState createState() => _LoadingState();
}

class _LoadingState extends State<Loading> {
  void setupForecastMap() async{
    DatabaseService instance = DatabaseService();
    for(var i = 1; i <= 20; i++){
      await instance.getImage(i);
    }

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
    setupForecastMap();
  }

  @override
  Widget build(BuildContext context) {
    Timer(
        Duration(seconds: 2, milliseconds: 500),
        () => Navigator.of(context).pushReplacement(MaterialPageRoute(builder: (BuildContext context) => MainApp()))
    );

    return Scaffold(
      backgroundColor: Colors.blueGrey,
      body: Center(
        child: SpinKitRotatingPlain(
          color: Colors.white,
          size: 50.0,
        ),
      ),
    );
  }
}
