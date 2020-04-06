import 'package:deep_rain/global/PushNotifications.dart';
import 'package:deep_rain/global/UIText.dart';
import 'package:deep_rain/main.dart';
import 'package:deep_rain/services/SliderService.dart';
import 'package:deep_rain/services/database.dart';
import 'package:flutter/material.dart';
import 'package:flutter_duration_picker/flutter_duration_picker.dart';
import 'package:flutter_map/flutter_map.dart';
import 'dart:async';
import 'package:flutter/material.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart';
import 'package:map_overlay/google/options.dart';
import 'package:map_overlay/google/overlay.dart';
import 'package:map_overlay/google/widgetContainer.dart';
import 'package:photo_view/photo_view.dart';
import 'package:settings_ui/settings_ui.dart';
import 'package:shared_preferences_settings/shared_preferences_settings.dart';

class Settings extends StatefulWidget {
  @override
  _LoadingState createState() => _LoadingState();
}

class _LoadingState extends State<Settings> {

  bool switchRegenwarnungen = true;

  UIText _uiText = UIText();
  PushNotifications _pushNotifications = PushNotifications();


  @override
  Widget build(BuildContext context) {

    Duration _duration = _pushNotifications.getTimeBeforeWarning();
    if(_duration == null){
      _duration = Duration(hours: 0, minutes: 20);
      _pushNotifications.setTimeBeforeWarning(_duration);
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
                              value: _uiText.getLanguage(),
                              onChanged: (String newValue){
                                _uiText.setLanguage(newValue);
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
                subtitle: 'Konstanz',
                leading: Icon(Icons.location_on),
                onTap: () {
                  showAlertDialog(context, "Region", "Hier könnte man eine Region auswählen");
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
                switchValue: switchRegenwarnungen,
                onToggle: (bool value) {
                  setState(() {
                    if(value == true){
                      DatabaseService _db = DatabaseService();
                      _db.activatePushNotification();
                    }else if(value == false){
                      DatabaseService _db = DatabaseService();
                      _db.deactivatePushNotification();
                    }
                    switchRegenwarnungen = !switchRegenwarnungen;
                  });
                },
              ),
              SettingsTile(
                title: _uiText.settingsTimeOfRainWarning,
                subtitle: _pushNotifications.getTimeBeforeWarning().inMinutes.toString() + _uiText.settingsTimeOfRainWarningSubtitle,
                leading: Icon(Icons.av_timer),
                onTap: () async{
                  return await showDialog(
                      context: context,
                      builder: (context) {
                        return AlertDialog(
                          title: Text(_uiText.chooseLanguageDialogHeader),
                          content: StatefulBuilder(
                            builder: (BuildContext context, StateSetter setState){
                              return DurationPicker(
                                    duration: _pushNotifications.getTimeBeforeWarning(),
                                    onChange: (Duration val) {
                                      if(val.inMinutes<60){
                                        _pushNotifications.setTimeBeforeWarning(val);
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
                              child: Text(_uiText.chooseLanguageDialogOkButton),
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
        ],
      ),
    );
  }

  showAlertDialog(BuildContext context, String titel, String text) {
    // set up the button
    Widget okButton = FlatButton(
      child: Text("OK"),
      onPressed: () {
        Navigator.pop(context);
      },
    );

    // set up the AlertDialog
    AlertDialog alert = AlertDialog(
      title: Text(titel),
      content: Text(text),
      actions: [
        okButton,
      ],
    );

    // show the dialog
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return alert;
      },
    );
  }

}
