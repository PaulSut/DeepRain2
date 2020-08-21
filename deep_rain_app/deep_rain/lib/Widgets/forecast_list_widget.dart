import 'dart:async';

import 'package:deep_rain/DataObjects/DemoData.dart';
import 'package:deep_rain/DataObjects/ForecastListItem.dart';
import 'package:deep_rain/Widgets/forecast_tile.dart';
import 'package:deep_rain/global/GlobalValues.dart';
import 'package:deep_rain/services/Database.dart';
import 'package:deep_rain/services/ProvideForecastData.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

//The list for the forecast list screen. Get latest data from Provider.

class ForecastListWidget extends StatefulWidget {
  @override
  _ForecastListWidgetState createState() => _ForecastListWidgetState();
}

class _ForecastListWidgetState extends State<ForecastListWidget> {
  @override
  Widget build(BuildContext context) {

    ProvideForecastData provider = ProvideForecastData();
    final forecasts = provider.getForecast();

    if(AppSwitchDemoMode == false){
      return ListView.builder(
        itemCount: forecasts == null ? 0 : forecasts.length,
        itemBuilder: (context, index) {
          return ForecastTile(forecast: forecasts[index]);
        },
      );
    }
    else{
      return ListView.builder(
        itemCount: DemoList == null ? 0 : DemoList.length,
        itemBuilder: (context, index) {
          return ForecastTile(forecast: DemoList[index]);
        },
      );
    }
  }
}