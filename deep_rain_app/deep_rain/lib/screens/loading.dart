import 'dart:async';
import 'dart:io';

import 'package:deep_rain/main.dart';
import 'package:deep_rain/services/database.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';

class Loading extends StatefulWidget {
  @override
  _LoadingState createState() => _LoadingState();
}

class _LoadingState extends State<Loading> {
  void setupForecastMap() async{
    DatabaseService instance = DatabaseService();
    for(var i = 1; i <= 18; i++){
      await instance.getImage(i);
    }
    //Navigator.pushReplacementNamed(context, '/MainApp');
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
