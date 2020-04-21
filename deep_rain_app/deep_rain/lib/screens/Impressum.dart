import 'package:deep_rain/global/UIText.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';

class Impressum extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    UIText _uiText = UIText();
    return Scaffold(
          backgroundColor: Colors.brown[50],
          appBar: AppBar(
            title: Text(_uiText.impressumAppTitle),
            elevation: 0.0,
          ),
          body: Text(
              'Hier kommt ein Impressum hin.'
          ),
      );
  }
}
