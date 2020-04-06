Duration AppTimeBeforeWarning;
String AppDeviceToken;
String AppLastDeviceTokenDocument;

class PushNotifications{

  setTimeBeforeWarning(Duration TimeBeforeWarning){
    AppTimeBeforeWarning = TimeBeforeWarning;
  }
  Duration getTimeBeforeWarning(){
    return AppTimeBeforeWarning;
  }

  setDeviceToken(String DeviceToken){
    AppDeviceToken = DeviceToken;
  }

  String getDeviceToken(){
    return AppDeviceToken;
  }

  setAppLastDeviceTokenDocument(String LastDeviceTokenDocument){
    AppLastDeviceTokenDocument = LastDeviceTokenDocument;
  }

  String getAppLastDeviceTokenDocument(){
    return AppLastDeviceTokenDocument;
  }

}