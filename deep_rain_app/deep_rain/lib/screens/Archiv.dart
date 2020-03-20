//Die Google Maps karte über die ein Bild gelegt werden kann
/*
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

class Settings extends StatefulWidget {
  @override
  _LoadingState createState() => _LoadingState();
}

class _LoadingState extends State<Settings> {
  GoogleMapController mapController;

  SliderService sliderService = SliderService();
  double rating = 0;
  int currentDivison = 0;
  int numberOfDivisions = 17;

  final LatLng _center = const LatLng(51.165691, 10.451526);

  void _onMapCreated(GoogleMapController controller) {
    mapController = controller;
  }

  @override
  Widget build(BuildContext context) {
      return Scaffold(
        body: GoogleMapOverlay(
          mapOptions: GoogleMapOptions(
            initialCameraPosition: CameraPosition(
              target: _center,
              zoom: 5.0,
            ),
            // Your typical google map options here (will pass through all events - even if used)
            zoomGesturesEnabled: true,
          ),
          overlays: <MapOverlayWidgetContainer>[
            // All widgets you wish to display on the map (they must have some form of default size)
            MapOverlayWidgetContainer(
              offset: Offset(50, 15),
              position: LatLng(51.165691, 10.451526),
              child: SizedBox(
                width: 200,
                height: 200,
                child: new PhotoView(
                    imageProvider: AssetImage('assets/1.png'),
                    minScale: PhotoViewComputedScale.contained * 1.25,
                    maxScale: 4.0,
                  ),
              ),
            ),
            MapOverlayWidgetContainer(
              position: LatLng(51.165691, 10.451526),
              child: Slider(
                value: rating,
                onChanged: (newRating){
                  setState(() {
                    rating = newRating;
                    currentDivison = newRating~/ (1/numberOfDivisions);
                  });
                },
                divisions: numberOfDivisions,
                label: sliderService.getTime(numberOfDivisions, rating),
              ),
            )
          ],
        )


        /*GoogleMap(
          onMapCreated: _onMapCreated,

          initialCameraPosition: CameraPosition(
            target: _center,
            zoom: 5.0,
          ),
        ),*/
      );
  }
}

 */