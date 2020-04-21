import 'package:latlong/latlong.dart';
import 'package:shared_preferences/shared_preferences.dart';

/*
All variables which need to be accesed are stored here. If a variable changes, the set method will be called.
The new value will be stored on the global variable which starts with "App...". It also will be stored in SharedPreferences (Local Key Value DB).
On Appstart the global values will be set from the data which is stored in shared preferences.
 */

Duration AppTimeBeforeWarning;
String AppDeviceToken;
String AppLastDeviceTokenDocument;
bool AppSwitchRainWarning;
String AppLanguage;
LatLng AppRegion;
String AppRegionCity;

class GlobalValues{

  //How long before the rain the user want to get his push notifivation
  setTimeBeforeWarning(Duration TimeBeforeWarning) async{
    AppTimeBeforeWarning = TimeBeforeWarning;
    final prefs = await SharedPreferences.getInstance();
    prefs.setInt('AppTimeBeforeWarning', TimeBeforeWarning.inMinutes);
  }
  Duration getTimeBeforeWarning(){
    return AppTimeBeforeWarning;
  }

  //A uniq devicetoken wich also is stored in the firestorage. It is needed from cloud functions to send push notification.
  setDeviceToken(String DeviceToken) async{
    AppDeviceToken = DeviceToken;
    final prefs = await SharedPreferences.getInstance();
    prefs.setString('AppDeviceToken', DeviceToken);
  }
  String getDeviceToken(){
    return AppDeviceToken;
  }

  //If the time of rain warning changes, the old devicetoken need to be deleted and stored in a other collection.
  //The document which need to be replaced is stored here.
  setAppLastDeviceTokenDocument(String LastDeviceTokenDocument) async{
    AppLastDeviceTokenDocument = LastDeviceTokenDocument;
    final prefs = await SharedPreferences.getInstance();
    prefs.setString('AppLastDeviceTokenDocument', LastDeviceTokenDocument);
  }
  String getAppLastDeviceTokenDocument(){
    return AppLastDeviceTokenDocument;
  }

  //Variable which store if push notification is activated or deactivated
  setAppSwitchRainWarning(bool SwitchRainWarning) async{
    AppSwitchRainWarning = SwitchRainWarning;
    final prefs = await SharedPreferences.getInstance();
    prefs.setBool('AppSwitchRainWarning', AppSwitchRainWarning);

  }
  getAppSwitchRainWarning(){
    if(AppSwitchRainWarning != null){
      return AppSwitchRainWarning;
    }
    return true;
  }

  //The zu language of the app.
  setAppLanguage(String Language) async {
    AppLanguage = Language;
    final prefs = await SharedPreferences.getInstance();
    prefs.setString('AppLanguage', AppLanguage);
  }
  String getAppLanguage(){
    if(AppLanguage != null){
      return AppLanguage;
    }
    return "Deutsch";
  }

  //The region of the app.
  setAppRegion(LatLng Region) async {
    AppRegion = Region;
    final prefs = await SharedPreferences.getInstance();
    prefs.setDouble('AppRegionLatitude', AppRegion.latitude);
    prefs.setDouble('AppRegionLongitude', AppRegion.longitude);
  }
  LatLng getAppRegion(){
    if(AppRegion != null){
      return AppRegion;
    }
    return LatLng(47.666947, 9.170982);
  }

  //The name of the city which the user choosed
  setAppRegionCity(String City) async {
    AppRegionCity = City;
    final prefs = await SharedPreferences.getInstance();
    prefs.setString('AppRegionCity', AppRegionCity);
  }
  String getAppRegionCity(){
    if(AppRegionCity != null){
      return AppRegionCity;
    }
    return "Konstanz";
  }
}