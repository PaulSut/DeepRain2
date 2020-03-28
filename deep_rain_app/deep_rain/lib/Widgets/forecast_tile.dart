import 'package:deep_rain/global/UIText.dart';
import 'package:flutter/material.dart';
import '../DataObjects/ForecastListItem.dart';

class ForecastTile extends StatelessWidget {

  final ForecastListItem forecast;
  ForecastTile({this.forecast});

  UIText _uiText = UIText();

  String getRainIntenseString(int rainIntense){
    if(rainIntense < 5){
      return _uiText.forecastListRainIntenseNoRain;
    }
    else if(rainIntense < 30){
      return _uiText.forecastListRainIntenseSomeRain;
    }
    else if(rainIntense < 60){
      return _uiText.forecastListRainIntenseMediumRain;
    }
    else if(rainIntense < 100){
      return _uiText.forecastListRainIntenseStrongRain;
    }
    return _uiText.forecastListRainIntenseError;
  }

  AssetImage getIcon(int rainIntese){
    if(rainIntese < 5){
      return AssetImage('assets/ForecastList/tagundnacht.png');
    }
    else{
      return AssetImage('assets/ForecastList/regenschirm.png');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: EdgeInsets.only(top: 3.0),
      child: Card(
        margin: EdgeInsets.fromLTRB(12.0, 4, 12.0, 0.0),
        child: ListTile(
          leading: CircleAvatar(
            radius: 25.0,
            backgroundColor: Colors.blueGrey[(((forecast.rainIntense)/10).round())*100],
            backgroundImage: getIcon(forecast.rainIntense),
          ),
          title: Text(getRainIntenseString(forecast.rainIntense)),
          subtitle: Text('${forecast.time}'),
        ),
      )
    );
  }
}
