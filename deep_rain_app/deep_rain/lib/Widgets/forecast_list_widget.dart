import 'package:deep_rain/DataObjects/ForecastListItem.dart';
import 'package:deep_rain/Widgets/forecast_tile.dart';
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

    final forecasts = Provider.of<List<ForecastListItem>>(context);

    return ListView.builder(
      itemCount: forecasts == null ? 0 : forecasts.length,
      itemBuilder: (context, index) {
        return ForecastTile(forecast: forecasts[index]);
      },
    );
  }
}
