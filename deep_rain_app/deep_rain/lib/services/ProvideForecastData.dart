import 'package:deep_rain/DataObjects/ForecastListItem.dart';

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


