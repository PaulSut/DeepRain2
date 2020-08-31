import 'package:deep_rain/DataObjects/ForecastListItem.dart';
//This class provide all the dataobjects for the forecast list. It is called, when new images are downloaded.

//The list which is shown in the list of forecasts
List<ForecastListItem> forecast_list;

class ProvideForecastData {
  //to set the
  setForecast(List<ForecastListItem> list){
    forecast_list = list;
  }
  List<ForecastListItem> getForecast() {
    print(forecast_list);
    return forecast_list;
  }
}