import 'package:deep_rain/DataObjects/ForecastListItem.dart';
import 'package:flutter/material.dart';
import 'package:deep_rain/services/database.dart';
import 'package:provider/provider.dart';
import 'package:deep_rain/Widgets/forecast_list_widget.dart';

class ForecastList extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return StreamProvider<List<ForecastListItem>>.value(
      value: DatabaseService().Forecast,
      child: Scaffold(
        backgroundColor: Colors.brown[50],
        appBar: AppBar(
          title: Text('Regen Vorhersage'),
          elevation: 0.0,
        ),
        body: ForecastListWidget()
      ),
    );
  }
}
