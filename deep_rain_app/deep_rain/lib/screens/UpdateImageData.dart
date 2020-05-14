import 'dart:async';
import 'package:animated_text_kit/animated_text_kit.dart';
import 'package:deep_rain/DataObjects/DataHolder.dart';
import 'package:deep_rain/screens/ForecastMap.dart';
import 'package:deep_rain/services/Database.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';

class UpdateImageData extends StatefulWidget {
  @override
  _UpdateImageDataState createState() => _UpdateImageDataState();
}

class _UpdateImageDataState extends State<UpdateImageData> {
  void setupForecastMap() async{
    DatabaseService instance = DatabaseService();
    for(var i = 1; i <= 20; i++){
      await instance.getImage(i);
    }
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
