import 'package:deep_rain/global/UIText.dart';
import 'package:flutter/material.dart';
import '../DataObjects/ForecastListItem.dart';

//The elements which are in the forecast list. Depending on the data of the forecast a other text and icon color will be changed.

class ForecastTile extends StatelessWidget {

  final ForecastListItem forecast;
  ForecastTile({this.forecast});

  UIText _uiText = UIText();

  //returns a string with the rain intense
  // these big values are the pixel values of the png forecast images
  String getRainIntenseString(int rainIntense){
    if(rainIntense < 2951124605){
      return _uiText.forecastListRainIntenseNoRain;
    }
    else if(rainIntense == 2951124605){
      return _uiText.forecastListRainIntenseSomeRain;
    }
    else if(rainIntense == 2951439430){
      return _uiText.forecastListRainIntenseMediumRain;
    }
    else if(rainIntense == 2952725805){
      return _uiText.forecastListRainIntenseStrongRain;
    }
    return _uiText.forecastListRainIntenseError;
  }

  //returns a string with the rain intense
  // these big values are the pixel values of the png forecast images
  int getRainColor(int rainIntense){
    if(rainIntense < 2951124605){
      return 200;
    }
    else if(rainIntense == 2951124605){
      return 400;
    }
    else if(rainIntense == 2951439430){
      return 600;
    }
    else if(rainIntense == 2952725805){
      return 800;
    }
    return 500;
  }

  //Return the image for the list entry depending on the rain intense
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
            backgroundColor: Colors.blueGrey[getRainColor(forecast.rainIntense)],
            backgroundImage: getIcon(forecast.rainIntense),
          ),
          title: Text(getRainIntenseString(forecast.rainIntense)),
          subtitle: Text('${forecast.time}'),
        ),
      )
    );
  }
}
