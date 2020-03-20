import 'package:deep_rain/DataObjects/ForecastListItem.dart';
import 'package:deep_rain/services/SliderService.dart';
import 'package:flutter/material.dart';
import '../my_flutter_app_icons.dart';

class ForecastData extends StatefulWidget {
  @override
  _ForecastDataState createState() => _ForecastDataState();
}

class _ForecastDataState extends State<ForecastData> {

  SliderService sliderService = SliderService();

  List rainIntense = [40, 20, 0, 0, 10, 20, 70, 80, 50, 30, 10, 0, 0, 0, 0, 10, 30];

  List rainIntense1 = ["Mittlerer Regen", "Leichter Regen", "Kein Regen", "Kein Regen", "Leichter Regen", "Leichter Regen", "Mittlerer Regen", "Schwerer Regen", "Schwerer Regen", "Sehr Schwerer Regen", "Mittlerer Regen", "Mittlerer Regen", "Leichter Regen", "Kein Regen", "Kein Regen", "Kein Regen", "Kein Regen"];

  int numberOfDevisions = 17;

  List getTimeStepsForList(){
    List timeSteps = [];

    for(double i = 0; i < 17; i++){
      timeSteps.add(sliderService.getTime(numberOfDevisions, i/17));
    }
    return timeSteps;
  }

  List generateListItems(){
    List itemList = [];
    for(int i = 0; i < 17; i++){
      String time = sliderService.getTime(numberOfDevisions, i/17);
      ForecastListItem listItem = ForecastListItem(time: time, rainIntense: rainIntense[i]);
      itemList.add(listItem);
    }
    return itemList;
  }

  Color getColor(int rainIntense){
    if(rainIntense == 0){
      return Colors.blueGrey[50];
    }
    else if(rainIntense < 30){
      return Colors.blueGrey[200];
    }
    else if(rainIntense < 60){
      return Colors.blueGrey[300];
    }
    else if(rainIntense < 100){
      return Colors.blueGrey[600];
    }
    else{
      return Colors.yellow;
    }
  }

  String getRainIntenseString(int rainIntense){
    if(rainIntense == 0){
      return "Kein Regen";
    }
    else if(rainIntense < 30){
      return "Leichter Regen";
    }
    else if(rainIntense < 60){
      return "Mittlerer Regen";
    }
    else if(rainIntense < 100){
      return "Schwerer Regen";
    }
    else{
      return "Keine gültigen Vorhersagedaten vorhanden";
    }
  }

  @override
  Widget build(BuildContext context) {

    List timeSteps = getTimeStepsForList();
    List itemList = generateListItems();

    return Scaffold(
      appBar: AppBar(
        title: Text("Vorhersageliste"),
      ),
      body: ListView.builder(
        itemBuilder: (context, position) {
          return Column(
            children: <Widget>[
              Container(
                decoration: new BoxDecoration(
                  color: getColor(itemList[position].rainIntense)
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: <Widget>[
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: <Widget>[
                        Padding(
                          padding:
                          const EdgeInsets.fromLTRB(12.0, 12.0, 12.0, 6.0),
                          child: Text(
                            getRainIntenseString(itemList[position].rainIntense),
                            style: TextStyle(
                                fontSize: 18.0, fontWeight: FontWeight.w600),
                          ),
                        ),
                        Padding(
                          padding:
                          const EdgeInsets.fromLTRB(12.0, 6.0, 12.0, 12.0),
                          child: Text(
                            itemList[position].time,
                            style: TextStyle(fontSize: 18.0),
                          ),
                        ),
                      ],
                    ),
                    Padding(
                      padding: const EdgeInsets.all(8.0),
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                        children: <Widget>[
                          Padding(
                            padding: const EdgeInsets.all(8.0),
                            child: Icon(
                              Icons.cloud_queue,
                              size: 35.0,
                              color: Colors.black,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
              Divider(
                height: 2.0,
                color: Colors.grey,
              )
            ],
          );
        },
        itemCount: rainIntense.length,
      ),
    );
  }
}
/*
void main() => runApp(MaterialApp(
  home: NinjaCard(),
));

class NinjaCard extends StatefulWidget {
  @override
  _NinjaCardState createState() => _NinjaCardState();
}

 */





