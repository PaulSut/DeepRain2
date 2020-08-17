import 'dart:async';
import 'dart:convert';
import 'dart:ffi';
import 'package:animated_text_kit/animated_text_kit.dart';
import 'package:deep_rain/DataObjects/ForecastListItem.dart';
import 'package:deep_rain/global/GlobalValues.dart';
import 'package:deep_rain/main.dart';
import 'package:deep_rain/services/Database.dart';
import 'package:deep_rain/services/ProvideForecastData.dart';
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

  //Is called with every start of the app. Download all Forecast images. Set the stored Settings.
  Future<bool> setupApp() async{
    // The local stored settings will be set again.
    final prefs = await SharedPreferences.getInstance();
    GlobalValues _globalValues = GlobalValues();
    await FirebaseAuth.instance.signInAnonymously();

    //The instance for the Databasehandler
    DatabaseService instance = DatabaseService();

    // restore the region
    if(_globalValues.getAppLastRegionDocument() == null){
      instance.storeRegion();
      _globalValues.setAppLastRegionDocument('Konstanz');
    } else{
      _globalValues.setAppLastRegionDocument(_globalValues.getAppLastRegionDocument());
    }

    // restore the language. Default Deutsch, otherwise the language which the User set.
    _globalValues.setAppLanguage(prefs.getString('AppLanguage') == null ? 'Deutsch' : prefs.getString('AppLanguage'));

    // restore the device Token. If null, put the token default in 20_min, otherwise in the time which was from user choosed.
    _globalValues.setAppLastDeviceTokenDocument(prefs.getString('AppLastDeviceTokenDocument') == null ? '20_min' : prefs.getString('AppLastDeviceTokenDocument'));
    _globalValues.setDeviceToken(prefs.getString('AppDeviceToken'));

    // restore the push notification switch (get notifications or not)
    _globalValues.setAppSwitchRainWarning(prefs.getBool('AppSwitchRainWarning'));

    // restore the cityname (only needed for UI)
    _globalValues.setAppRegionCity(prefs.getString('AppRegionCity') == null ? 'Konstanz' : prefs.getString('AppRegionCity'));

    // set the time when the push notification should be sended
    int minutes = prefs.getInt('AppTimeBeforeWarning');
    Duration _duration = Duration(minutes: minutes == null ? 20 : minutes);
    _globalValues.setTimeBeforeWarning(_duration);

    //activate Pushnotifications by default
    instance.activatePushNotification();

    // Set the region of the app. Is needed to get the Data for the Forecastlist. Default: Konstanz.
    double latitude = prefs.getDouble('AppRegionLatitude');
    double longitude = prefs.getDouble('AppRegionLongitude');
    LatLng _latLng = LatLng(latitude == null ? 47.66033 : latitude, longitude == null ? 9.17582 : longitude);
    _globalValues.setAppRegion(_latLng);

    // Restore the coordinate of the region in the forecast image (x and y pixel). If the pixel is not calculated, it will need to calculate it (take some time)
    int appPixel_x = prefs.getInt('AppPixel_X');
    int appPixel_y = prefs.getInt('AppPixel_Y');
    if(appPixel_y == null && appPixel_x == null){
      print('die Pixel waren null');
      await _globalValues.getAppPixel();
    }else{
      await _globalValues.setAppPixel([appPixel_x, appPixel_y]);
    }

    List<String> time_steps = [];
    instance.TimeSteps.listen((event) {
      time_steps = event;
    });

    // download the forecast images
    int pixel_value;
    List<ForecastListItem> forecast_list = [];
    for(var i = 1; i <= 20; i++){
      print('Ich hole Bilder');
      pixel_value = await instance.getImage(i);
      forecast_list.add(ForecastListItem(rainIntense: pixel_value, time: time_steps[i-1]));
    }
    ProvideForecastData provider = ProvideForecastData();
    provider.setForecast(forecast_list);

    final FirebaseMessaging _fcm = FirebaseMessaging();
    await _fcm.getToken().then((token) async{
      _globalValues.setDeviceToken(token);
    });

    // Navigate to the next screen.
    Navigator.of(context).pushReplacement(MaterialPageRoute(builder: (BuildContext context) => MainApp()));
  }

  @override
  Future<void> initState() {
    super.initState();
  }

  // UI Widget
  @override
  Widget build(BuildContext context) {
    return FutureBuilder(
      future: setupApp(),
      builder: (context, snap){
        return  Scaffold(
          backgroundColor: Colors.blueGrey,
          body: Center(
            child: TypewriterAnimatedTextKit(
              totalRepeatCount: 3,
              pause: Duration(milliseconds:  1000),
              text: ["deepRain", "stay dry", "HTWG Konstanz"],
              textStyle: TextStyle(fontSize: 32.0, fontWeight: FontWeight.bold),
              displayFullTextOnTap: true,
              stopPauseOnTap: true,
            ),
          ),
        );
      },
    );
  }
}
