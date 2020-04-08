import 'package:shared_preferences/shared_preferences.dart';

Duration AppTimeBeforeWarning;
String AppDeviceToken;
String AppLastDeviceTokenDocument;
bool AppSwitchRainWarning;

class PushNotifications{

  PushNotifications(){
    if(AppSwitchRainWarning == null){
      AppSwitchRainWarning = true;
    }
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
    return AppSwitchRainWarning;
  }

  loadDataFromSharedPreferences() async{
    final prefs = await SharedPreferences.getInstance();

    AppSwitchRainWarning = prefs.getBool('AppSwitchRainWarning');
    AppTimeBeforeWarning = Duration(minutes: prefs.getInt('AppTimeBeforeWarning'));
    AppDeviceToken = prefs.getString('AppDeviceToken');
    AppLastDeviceTokenDocument = prefs.getString('AppLastDeviceTokenDocument');

  }
}