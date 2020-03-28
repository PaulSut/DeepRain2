import 'dart:async';
import 'package:deep_rain/DataObjects/DataHolder.dart';
import 'package:deep_rain/screens/ForecastMap.dart';
import 'package:deep_rain/services/database.dart';
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
    Timer(
        Duration(seconds: 2, milliseconds: 500),
            () => Navigator.of(context).pop(MaterialPageRoute(builder: (BuildContext context) => ForecastMap()))
    );

    return Scaffold(
      backgroundColor: Colors.blueGrey,
      body: Center(
        child: SpinKitRotatingPlain(
          color: Colors.white,
          size: 50.0,
          itemBuilder: (BuildContext context, int index) {
            return Text('Bilder werden runtergeladen...');
          },
        ),
      ),
    );
  }
}
