import 'package:shared_preferences/shared_preferences.dart';

Duration AppTimeBeforeWarning;
String AppDeviceToken;
String AppLastDeviceTokenDocument;
bool AppSwitchRainWarning;
String AppLanguage;

class GlobalValues{

  GlobalValues(){
  }

  setTimeBeforeWarning(Duration TimeBeforeWarning) async{
    AppTimeBeforeWarning = TimeBeforeWarning;
    final prefs = await SharedPreferences.getInstance();
    prefs.setInt('AppTimeBeforeWarning', TimeBeforeWarning.inMinutes);
  }
  Duration getTimeBeforeWarning(){
    return AppTimeBeforeWarning;
  }

  setDeviceToken(String DeviceToken) async{
    AppDeviceToken = DeviceToken;
    final prefs = await SharedPreferences.getInstance();
    prefs.setString('AppDeviceToken', DeviceToken);
  }

  String getDeviceToken(){
    return AppDeviceToken;
  }

  setAppLastDeviceTokenDocument(String LastDeviceTokenDocument) async{
    AppLastDeviceTokenDocument = LastDeviceTokenDocument;

    final prefs = await SharedPreferences.getInstance();
    prefs.setString('AppLastDeviceTokenDocument', LastDeviceTokenDocument);
  }

  String getAppLastDeviceTokenDocument(){
    return AppLastDeviceTokenDocument;
  }

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

  loadDataFromSharedPreferences() async{
    final prefs = await SharedPreferences.getInstance();

    AppSwitchRainWarning = prefs.getBool('AppSwitchRainWarning');
    AppTimeBeforeWarning = Duration(minutes: prefs.getInt('AppTimeBeforeWarning'));
    AppDeviceToken = prefs.getString('AppDeviceToken');
    AppLastDeviceTokenDocument = prefs.getString('AppLastDeviceTokenDocument');
    AppLanguage = prefs.getString('AppLanguage');

  }
}