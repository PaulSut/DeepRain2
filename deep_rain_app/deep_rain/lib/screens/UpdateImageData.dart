import 'package:animated_text_kit/animated_text_kit.dart';
import 'package:deep_rain/DataObjects/DataHolder.dart';
import 'package:deep_rain/DataObjects/ForecastListItem.dart';
import 'package:deep_rain/global/GlobalValues.dart';
import 'package:deep_rain/screens/ForecastMap.dart';
import 'package:deep_rain/services/Database.dart';
import 'package:deep_rain/services/ProvideForecastData.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';

class UpdateImageData extends StatefulWidget {
  @override
  _UpdateImageDataState createState() => _UpdateImageDataState();
}

class _UpdateImageDataState extends State<UpdateImageData> {
  void setupForecastMap() async{
    DatabaseService instance = DatabaseService();

    List<String> time_steps = [];
    instance.TimeSteps.listen((event) {
      time_steps = event;
    });

    // download the forecast images
    int pixel_value;
    List<ForecastListItem> forecast_list = [];
    for(var i = 1; i <= 8; i++){
      print('Ich hole Bilder');
      pixel_value = await instance.getImage(i);
      forecast_list.add(ForecastListItem(rainIntense: pixel_value, time: time_steps[i-1]));
    }
    ProvideForecastData provider = ProvideForecastData();
    provider.setForecast(forecast_list);
    provider.setTimeSteps(time_steps);

    GlobalValues _globalValues = GlobalValues();
    _globalValues.setAppSwitchDemoMode(true);
    _globalValues.setAppSwitchDemoMode(false);
  }

  @override
  void initState(){
    super.initState();
    imageData = {};
    requestedIndexes = [];
    setupForecastMap();
  }


  @override
  Widget build(BuildContext context) {

    return Scaffold(
      backgroundColor: Colors.blueGrey,
      body: Center(
        child: TypewriterAnimatedTextKit(
          totalRepeatCount: 1,
          pause: Duration(milliseconds:  500),
          text: ["Vorhersage", "wird", "aktualisiert"],
          textStyle: TextStyle(fontSize: 32.0, fontWeight: FontWeight.bold),
          displayFullTextOnTap: true,
          stopPauseOnTap: true,
          onFinished:(){
            Navigator.of(context).pop(MaterialPageRoute(builder: (BuildContext context) => ForecastMap()));
          },
        ),
      ),
    );
  }
}
