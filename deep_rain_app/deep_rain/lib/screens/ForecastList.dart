import 'package:deep_rain/DataObjects/ForecastListItem.dart';
import 'package:deep_rain/global/GlobalValues.dart';
import 'package:deep_rain/global/UIText.dart';
import 'package:deep_rain/services/Database.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:deep_rain/Widgets/forecast_list_widget.dart';

/*
This screen displays the forecast data as list. Cause of StreamProvider, always the latest data will be displayed.
 */
class ForecastList extends StatelessWidget {
  GlobalValues _globalValues = GlobalValues();
  @override
  Widget build(BuildContext context) {
    UIText _uiText = UIText();
    return StreamProvider<List<ForecastListItem>>.value(
      value: DatabaseService().Forecast,
      child: Scaffold(
        backgroundColor: Colors.brown[50],
        appBar: AppBar(
          title: Text(_uiText.forecastListAppTitle + " " + _globalValues.getAppRegionCity()),
          elevation: 0.0,
        ),
        body: ForecastListWidget()
      ),
    );
  }
}
