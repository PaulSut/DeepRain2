import 'package:deep_rain/global/GlobalValues.dart';
import 'package:deep_rain/global/UIText.dart';
import 'package:deep_rain/services/Database.dart';
import 'package:flutter/material.dart';
import 'package:flutter_duration_picker/flutter_duration_picker.dart';
import 'package:latlong/latlong.dart';
import 'package:nominatim_location_picker/nominatim_location_picker.dart';
import 'package:settings_ui/settings_ui.dart';
import 'package:geocoder/geocoder.dart';
import 'dart:io' show Platform;

import 'Impressum.dart';

//The screen for the settings. Every setting will be stored in the global values and shared preferences (local db).
//Some of them will be uploaded to firebase.
class Settings extends StatefulWidget {
  @override
  _LoadingState createState() => _LoadingState();
}

class _LoadingState extends State<Settings> {

  UIText _uiText = UIText();
  GlobalValues _globalValues = GlobalValues();

//Open a screen in which a city can be choosed. This city will always be zoomed in by default,
// The push notifications will only be sended for this location.
  Future getLocationWithNominatim() async {
    Map result = await showDialog(
        context: context,
        builder: (BuildContext ctx) {
          return NominatimLocationPicker(
            searchHint: _uiText.chooseRegionScreenSearchHint,
            awaitingForLocation: _uiText.chooseRegionScreenAwaitingForLocation,
          );
        });
    if (result != null) {
      LatLng coordinatesInLatLng = result["latlng"];
      Coordinates coordinates = new Coordinates(coordinatesInLatLng.latitude, coordinatesInLatLng.longitude);
      _globalValues.setAppRegion(coordinatesInLatLng);

      _globalValues.changeAppPixel();

      var addresses = await Geocoder.local.findAddressesFromCoordinates(coordinates);
      var first = addresses.first;
      _globalValues.setAppRegionCity(first.locality);
      setState(() {});
    } else {
      return;
    }
  }

  @override
  Widget build(BuildContext context) {
    //If the user did not changed the time of warning, it will be set to 20 min by default.
    Duration _duration = _globalValues.getTimeBeforeWarning();
    if(_duration == null){
      _duration = Duration(hours: 0, minutes: 20);
      _globalValues.setTimeBeforeWarning(_duration);
    }

    return Scaffold(
      appBar: AppBar(
        title: Text(_uiText.settingsAppTitle),
      ),
      body: SettingsList(
        sections: [
          SettingsSection(
            title: _uiText.settingsHeaderGeneral,
            tiles: [
              SettingsTile(
                title: _uiText.settingsLanguage,
                subtitle: AppLanguage,
                leading: Icon(Icons.language),
                onTap: () async {
                  return await showDialog(
                      context: context,
                      builder: (context) {
                        return AlertDialog(
                          title: Text(_uiText.chooseLanguageDialogHeader),
                          content: StatefulBuilder(
                            builder: (BuildContext context, StateSetter setState){
                              return DropdownButton<String>(
                              value: _globalValues.getAppLanguage(),
                              onChanged: (String newValue){
                                _globalValues.setAppLanguage(newValue);
                                setState(() {});
                              },
                              items: <String> ['Deutsch', 'English', 'Español'].map<DropdownMenuItem<String>>((String value){
                                return DropdownMenuItem<String>(
                                value: value,
                                child: Text(value),
                              );
                              }).toList(),
                              );
                            },
                          ),
                          actions: <Widget>[
                            FlatButton(
                              color: Colors.blueGrey,
                              textColor: Colors.white,
                              disabledColor: Colors.grey,
                              disabledTextColor: Colors.black,
                              padding: EdgeInsets.all(8.0),
                              splashColor: Colors.blueAccent,
                              onPressed: (){
                                Navigator.of(context).pop();

                              },
                              child: Text(_uiText.chooseLanguageDialogOkButton),
                            )
                          ],
                        );
                      }).then((value){
                        setState((){});
                      });
                },
              ),
              SettingsTile(
                title: _uiText.settingsRegion,
                subtitle: _globalValues.getAppRegionCity(),
                leading: Icon(Icons.location_on),
                onTap: () async {
                  getLocationWithNominatim().then((val){
                    DatabaseService _dbService = DatabaseService();
                    _dbService.updateRegion();
                  });
                },
              ),
            ],
          ),
          SettingsSection(
            title: _uiText.settingsHeaderNotifications,
            tiles: [
              SettingsTile.switchTile(
                title: _uiText.settingsRainWarning,
                leading: Icon(Icons.priority_high),
                switchValue: _globalValues.getAppSwitchRainWarning(),
                onToggle: (bool value) {
                  setState(() {
                    if(value == true){
                      DatabaseService _db = DatabaseService();
                      _db.activatePushNotification();
                    }else if(value == false){
                      DatabaseService _db = DatabaseService();
                      _db.deactivatePushNotification();
                    }
                    _globalValues.setAppSwitchRainWarning(!_globalValues.getAppSwitchRainWarning());
                  });
                },
              ),
              SettingsTile(
                title: _uiText.settingsTimeOfRainWarning,
                subtitle: Platform.isAndroid ? _globalValues.getTimeBeforeWarning().inMinutes.toString() + _uiText.settingsTimeOfRainWarningSubtitle : _globalValues.getTimeBeforeWarning().inMinutes.toString() + 'min.',
                leading: Icon(Icons.av_timer),
                onTap: () async{
                  return await showDialog(
                      context: context,
                      builder: (context) {
                        return AlertDialog(
                          title: Text(_uiText.chooseWarningTimeScreen),
                          content: StatefulBuilder(
                            builder: (BuildContext context, StateSetter setState){
                              return DurationPicker(
                                    duration: _globalValues.getTimeBeforeWarning(),
                                    onChange: (Duration val) {
                                      if(val.inMinutes<60){
                                        _globalValues.setTimeBeforeWarning(val);
                                        setState((){});
                                      }
                                    },
                                    snapToMins: 5.0,
                                  );
                            },
                          ),
                          actions: <Widget>[
                            FlatButton(
                              color: Colors.blueGrey,
                              textColor: Colors.white,
                              disabledColor: Colors.grey,
                              disabledTextColor: Colors.black,
                              padding: EdgeInsets.all(8.0),
                              splashColor: Colors.blueAccent,
                              onPressed: (){
                                Navigator.of(context).pop();
                              },
                              child: Text(_uiText.chooseWarningTimeDialogOkButton),
                            )
                          ],
                        );
                      }).then((value){
                        DatabaseService _db = DatabaseService();
                        _db.updatePushNotificationTime();
                    setState((){});
                  });
                },
              ),
            ],
          ),
          SettingsSection(
            title: _uiText.settingsHeaderMore,
            tiles: [
              SettingsTile.switchTile(
                title: _uiText.settingsDemoMode,
                leading: Icon(Icons.insert_chart),
                switchValue: _globalValues.getAppSwitchDemoMode(),
                onToggle: (bool value) {
                  setState(() {
                    if(value == true){
                      print('Set false');
                    }else if(value == false){
                      print('Set true');
                    }
                    _globalValues.setAppSwitchDemoMode(!_globalValues.getAppSwitchDemoMode());
                  });
                },
              ),
              SettingsTile(
                title: _uiText.settingsImpressum,
                leading: Icon(Icons.person_outline),
                onTap: (){
                  Navigator.push(
                    context,
                    MaterialPageRoute(builder: (context) => Impressum()),
                  );
                },
              )
            ],
          )
        ],
      ),
    );
  }
}