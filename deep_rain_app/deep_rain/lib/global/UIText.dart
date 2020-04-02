import 'dart:async';

import 'package:deep_rain/main.dart';
import 'package:flutter/cupertino.dart';

String AppLanguage = 'Deutsch';

class UIText{
  setLanguage(String Language){
    AppLanguage = Language;
  }
  String getLanguage(){
    return AppLanguage;
  }

  //ForecastList
  String forecastListAppTitle;
  String forecastListRainIntenseNoRain;
  String forecastListRainIntenseSomeRain;
  String forecastListRainIntenseMediumRain;
  String forecastListRainIntenseStrongRain;
  String forecastListRainIntenseError;

  //ForecastMap
  String forecastMapAppTitle;

  //Settings
  String settingsAppTitle;
  String settingsHeaderGeneral;
  String settingsLanguage;
  String settingsRegion;
  String settingsHeaderNotifications;
  String settingsRainWarning;
  String settingsTimeOfRainWarning;
  String settingsTimeOfRainWarningSubtitle;
    //Choose Language Dialog
    String chooseLanguageDialogHeader;
    String chooseLanguageDialogOkButton;

  //Bottom Navigation
  String List;
  String Map;
  String Settings;


  UIText(){
    if(AppLanguage == 'Deutsch'){
      //ForecastList
      forecastListAppTitle = 'Regenvorhersage';
      forecastListRainIntenseNoRain = 'Kein Regen';
      forecastListRainIntenseSomeRain = 'Leichter Regen';
      forecastListRainIntenseMediumRain = 'Mittlerer Regen';
      forecastListRainIntenseStrongRain = 'Starker Regen';
      forecastListRainIntenseError = 'Es gab einen Fehler';

      //ForecastMap
      forecastMapAppTitle = 'Regenkarte';

      //Settings
      settingsAppTitle = 'Einstellungen';
      settingsHeaderGeneral = 'Allgemeines';
      settingsLanguage = 'Sprache';
      settingsRegion = 'Region';
      settingsHeaderNotifications = 'Benachrichtigungen';
      settingsRainWarning = 'Regenwarnung';
      settingsTimeOfRainWarning = 'Zeitpunkt Regenwarnung';
      settingsTimeOfRainWarningSubtitle = ' Minuten vor dem Regen';
        //choose language dialog
        chooseLanguageDialogHeader = 'Sprache auswählen';
        chooseLanguageDialogOkButton = 'Bestätigen';

      //Bottom Naviagation
      List = 'Liste';
      Map = 'Karte';
      Settings = 'Einstellungen';


    }
    else if(AppLanguage == 'Español'){
      //ForecastList
      forecastListAppTitle = 'Previsión de lluvia';
      forecastListRainIntenseNoRain = 'no llueve';
      forecastListRainIntenseSomeRain = 'Lluvia ligera';
      forecastListRainIntenseMediumRain = 'Lluvia media';
      forecastListRainIntenseStrongRain = 'Lluvia fuerte';
      forecastListRainIntenseError = 'Hubo un error';

      //ForecastMap
      forecastMapAppTitle = 'Mapa de la lluvia';

      //Settings
      settingsAppTitle = 'Ajustes';
      settingsHeaderGeneral = 'Información general';
      settingsLanguage = 'Idioma';
      settingsRegion = 'Región';
      settingsHeaderNotifications = 'Notificaciones';
      settingsRainWarning = 'Advertencia de lluvia';
      settingsTimeOfRainWarning = 'La hora de la advertencia de la lluvia';
      settingsTimeOfRainWarningSubtitle = ' minutos antes de la lluvia';
        //choose language dialog
        chooseLanguageDialogHeader = 'Seleccione el idioma';
        chooseLanguageDialogOkButton = 'Confirme';

      //Bottom Naviagation
      List = 'Lista';
      Map = 'Mapa';
      Settings = 'Ajustes';


    }
    else if(AppLanguage == 'English'){
      //ForecastList
      forecastListAppTitle = 'Rainforecast';
      forecastListRainIntenseNoRain = 'No rain';
      forecastListRainIntenseSomeRain = 'A little rain';
      forecastListRainIntenseMediumRain = 'Medium rain';
      forecastListRainIntenseStrongRain = 'Strong rain';
      forecastListRainIntenseError = 'There was an error';

      //ForecastMap
      forecastMapAppTitle = 'Rainfall map';

      //Settings
      settingsAppTitle = 'Settings';
      settingsHeaderGeneral = 'General';
      settingsLanguage = 'Language';
      settingsRegion = 'Region';
      settingsHeaderNotifications = 'Notifications';
      settingsRainWarning = 'Rainwarning';
      settingsTimeOfRainWarning = 'Time of rainwarning';
      settingsTimeOfRainWarningSubtitle = ' minutes before the rain';
        //choose language dialog
        chooseLanguageDialogHeader = 'Choose Language';
        chooseLanguageDialogOkButton = 'Ok';

      //Bottom Naviagation
      List = 'List';
      Map = 'Map';
      Settings = 'Settings';
    }
  }
}

