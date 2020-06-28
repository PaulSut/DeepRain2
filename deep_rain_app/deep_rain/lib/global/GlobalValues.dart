import 'dart:convert';

import 'package:deep_rain/services/FindPixel.dart';
import 'package:flutter/services.dart';
import 'package:latlong/latlong.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:ml_linalg/linalg.dart';
import 'dart:math';

/*
All variables which need to be accesed are stored here. If a variable changes, the set method will be called.
The new value will be stored on the global variable which starts with "App...". It also will be stored in SharedPreferences (Local Key Value DB).
On Appstart the global values will be set from the data which is stored in shared preferences.
 */

Duration AppTimeBeforeWarning;
String AppDeviceToken;
String AppLastDeviceTokenDocument;
String AppLastRegionDocument;
bool AppSwitchRainWarning;
String AppLanguage;
LatLng AppRegion;
String AppRegionCity;
List<String> AppCoordinateList;
List<String> AppLatitudeList;
List<String> AppLongitudeList;
var AppPixel;

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

  //If the region changes, the old region need to be deleted and stored in a other collection.
  //The document which need to be replaced is stored here.
  setAppLastRegionDocument(String LastRegionDocument) async{
    AppLastRegionDocument = LastRegionDocument;
    final prefs = await SharedPreferences.getInstance();
    prefs.setString('AppLastRegionDocument', LastRegionDocument);
  }
  String getAppLastRegionDocument(){
    return AppLastRegionDocument;
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

  setCoordinateLists(String CoordinateList) async{
//    AppCoordinateList = CoordinateList.map((s) => s as String).toList();
    final prefs = await SharedPreferences.getInstance();
    prefs.setString('AppCoordinateList', CoordinateList);
  }
  List<dynamic> getCoordinateLists(){
    return AppCoordinateList;
  }

  setLatitudeList(String LatitudeList) async{
    //AppLatitudeList = LatitudeList.map((s) => s as String).toList();
    final prefs = await SharedPreferences.getInstance();
    prefs.setString('AppLatitudeList', LatitudeList);
  }
  List<dynamic> getLatitudeList(){
    return AppLatitudeList;
  }

  setLongitudeList(String LongitudeList) async{
    //AppLongitudeList = LongitudeList.map((s) => s as String).toList();
    final prefs = await SharedPreferences.getInstance();
    prefs.setString('AppLongitudeList', LongitudeList);
  }
  List<dynamic> getLongitudeList(){
    return AppLongitudeList;
  }

  Future<String> loadListCoordinates() async {
    final ByteData data = await rootBundle.load('assets/data/listCoordinates.json');
    String jsonContent = utf8.decode(data.buffer.asUint8List());
    return jsonContent;

  }
  Future<String> loadListLatitude() async {
    final ByteData data = await rootBundle.load('assets/data/listLatitudeComplete.json');
    String jsonContent = utf8.decode(data.buffer.asUint8List());
    return jsonContent;
  }
  Future<String> loadListLongitude() async {
    final ByteData data = await rootBundle.load('assets/data/listLongitudeComplete.json');
    String jsonContent = utf8.decode(data.buffer.asUint8List());
    return jsonContent;
  }

  setAppPixel(var Pixel) async{
    AppPixel = Pixel;
    final prefs = await SharedPreferences.getInstance();
    prefs.setInt('AppPixel_X', Pixel[0]);
    prefs.setInt('AppPixel_Y', Pixel[1]);
  }
  getAppPixel() async{
    if(AppPixel != null){
      return AppPixel;
    }
    if(AppPixel == null){
      changeAppPixel();
    }
  }

  changeAppPixel() async{
    print('Ich berechne jetzt den Pixel!');
    String coordinateListString = await loadListCoordinates();
    String latitudeListString = await loadListLatitude();
    String longitudeListString = await loadListLongitude();

    print('Ich berechne jetzt den Pixel! 2');
    var coordinateListVar = await json.decode(coordinateListString);
    print('Ich berechne jetzt den Pixel! 3');
    var longitudeListVar = await jsonDecode(longitudeListString);
    print('Ich berechne jetzt den Pixel! 4');
    var latitudeListVar = await jsonDecode(latitudeListString);

    print('TADAAA');
    print(coordinateListVar[100]);
    print(longitudeListVar[100]);
    print(latitudeListVar[100]);

    double latitude = AppRegion.latitude;
    double longitude = AppRegion.longitude;
    print(latitude);
    print(longitude);

    FindPixel hey = FindPixel();
//    var longitude_min = longitudeListVar.reduce((curr, next) => curr < next? curr: next);
//    var longitude_max = longitudeListVar.reduce((curr, next) => curr > next? curr: next);
//    var latitude_min = latitudeListVar.reduce((curr, next) => curr < next? curr: next);
//    var latitude_max = latitudeListVar.reduce((curr, next) => curr > next? curr: next);

    print(longitudeListVar.length);

    int row = 900;
    int col = 900;
    var longitude_list_2d = List.generate(row, (i) => List.generate(col, (k) => longitudeListVar[i * k + k]), growable: false);
    var latitude_list_2d = List.generate(row, (i) => List.generate(col, (k) => latitudeListVar[i * k + k]), growable: false);
    var coordinate_list_2d = List.generate(row, (i) => List.generate(col, (k) => coordinateListVar[i * k + k]), growable: false);


    //var pixels_index = hey.getClosest_Coordinate(longitude_list_2d, latitude_list_2d, 899, 0, 899, 0, AppRegion.longitude, AppRegion.latitude);
    //print(coordinate_list_2d[pixels_index[0]][pixels_index[1]]);

    var pixels = [300,200];

    print('HALELASJDLJLASJDLAJSDL');
    print(pixels);

    coordinateListVar = null;
    longitudeListVar = null;
    latitudeListVar  = null;

    await setAppPixel(pixels);

    print('Ich berechne jetzt den Pixel! 5');

    return pixels;
  }

}
