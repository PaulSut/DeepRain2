import 'package:deep_rain/services/SliderService.dart';
import 'package:flutter/material.dart';
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Einstellungen"),
      ),
      body: SettingsList(
        sections: [
          SettingsSection(
            title: 'Allgemeines',
            tiles: [
              SettingsTile(
                title: 'Sprache',
                subtitle: 'Deutsch',
                leading: Icon(Icons.language),
                onTap: () {
                  showAlertDialog(context, "Sprache", "Hier könnte man eine Sprache auswählen");
                },
              ),
              SettingsTile(
                title: 'Region',
                subtitle: 'Konstanz',
                leading: Icon(Icons.location_on),
                onTap: () {
                  showAlertDialog(context, "Region", "Hier könnte man eine Region auswählen");
                },
              ),
            ],
          ),
          SettingsSection(
            title: 'Benachrichtigungen',
            tiles: [
              SettingsTile.switchTile(
                title: 'Regenwarnungen',
                leading: Icon(Icons.priority_high),
                switchValue: switchRegenwarnungen,
                onToggle: (bool value) {
                  setState(() {
                    switchRegenwarnungen = !switchRegenwarnungen;
                  });
                },
              ),
              SettingsTile(
                title: 'Zeitpunkt Regenwarnung',
                subtitle: '30 Minuten',
                leading: Icon(Icons.av_timer),
                onTap: () {
                  showAlertDialog(context, "Benachrichtigung vor Regen in Minuten", "Hier könnte man eine Sprache auswählen");
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
