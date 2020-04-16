import 'dart:typed_data';

import 'package:deep_rain/DataObjects/DataHolder.dart';
import 'package:deep_rain/global/GlobalValues.dart';
import 'package:deep_rain/global/UIText.dart';
import 'package:deep_rain/services/SliderService.dart';
import 'package:flutter/material.dart';
import 'package:latlong/latlong.dart';
import 'package:flutter_map/flutter_map.dart';

import 'UpdateImageData.dart';


class ForecastMap extends StatefulWidget {
  @override
  _ForecastMapState createState() => _ForecastMapState();
}

class _ForecastMapState extends State<ForecastMap> {

  UIText _uiText = UIText();
  GlobalValues _globalValues = GlobalValues();

  SliderService sliderService = SliderService();
  double rating = 0;
  int currentDivison = 1;
  int numberOfDivisions = 20;
  Uint8List imageFile;

  @override
  Widget build(BuildContext context) {
    //if the image for the current dvision is not stored in DataHolder
    if (!imageData.containsKey(currentDivison)) {
      //imageFile = dbInstance.getImage(currentDivison);
    } else {
      // set the imageFile to the Image from the DataHolder
      this.setState(() {
        imageFile = imageData[currentDivison];
      });
    }

    return Scaffold(
      appBar: AppBar(
        title: Text(_uiText.forecastMapAppTitle),
        actions: <Widget>[
          IconButton(
            icon: Icon(Icons.update),
            onPressed: () {
              Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => UpdateImageData()),
              );
            },
          ),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            flex: 9,
            child: FlutterMap(
              options: MapOptions(
                center: _globalValues.getAppRegion(),
                zoom: 12.0,
              ),
              layers: [
                TileLayerOptions(
                    urlTemplate: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
                    subdomains: ['a', 'b', 'c']
                ),
                OverlayImageLayerOptions(overlayImages: <OverlayImage>[
                  OverlayImage(
                    bounds: LatLngBounds(LatLng(54.469581, 5.457018), LatLng(47.321954, 13.781278)),
                    opacity: 0.8,
                    imageProvider: imageFile == null ? AssetImage('assets/error.png') : Image.memory(imageFile).image,
                  ),
                ],
                ),
              ],
            ),
          ),
          Expanded(
            flex: 1,
            child: Slider(
              value: rating,
              onChanged: (newRating) {
                setState(() {
                  rating = newRating;
                  currentDivison = newRating ~/ (1 / numberOfDivisions);
                });
              },
              divisions: numberOfDivisions,
              label: sliderService.getTime(numberOfDivisions, rating),
            ),
          ),
        ],
      ),
    );
  }
}

