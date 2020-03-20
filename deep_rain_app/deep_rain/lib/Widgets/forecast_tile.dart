import 'package:flutter/material.dart';
import '../DataObjects/ForecastListItem.dart';

class ForecastTile extends StatelessWidget {

  final ForecastListItem forecast;
  ForecastTile({this.forecast});

  String getRainIntenseString(int rainIntense){
    if(rainIntense < 5){
      return 'Kein Regen';
    }
    else if(rainIntense < 30){
      return 'Leichter Regen';
    }
    else if(rainIntense < 60){
      return 'Mittlerer Regen';
    }
    else if(rainIntense < 100){
      return 'Starker Regen';
    }
    return 'Es gab einen Fehler';
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
