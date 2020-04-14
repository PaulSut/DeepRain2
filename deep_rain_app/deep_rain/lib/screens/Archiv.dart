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


/*
Die alte ForecastMap
 */
import 'dart:typed_data';

import 'package:deep_rain/DataObjects/DataHolder.dart';
import 'package:deep_rain/global/UIText.dart';
import 'package:deep_rain/screens/UpdateImageData.dart';
import 'package:deep_rain/services/SliderService.dart';
import 'package:deep_rain/services/database.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:flutter/material.dart';
import 'package:photo_view/photo_view.dart';


class ForecastMap extends StatefulWidget {
  @override
  _ForecastMapState createState() => _ForecastMapState();
}

class _ForecastMapState extends State<ForecastMap> {

  UIText _uiText = UIText();

  SliderService sliderService = SliderService();
  double rating = 0;
  int currentDivison = 1;
  int numberOfDivisions = 20;

  Uint8List imageFile;
  DatabaseService dbInstance = new DatabaseService();
  StorageReference photosReference = FirebaseStorage.instance.ref().child('photos');

  //if the image is not already stored in the DataHolder, it will be downloaded from firebase
  getImage(int division) {
    division = division + 1;
    if (!requestedIndexes.contains(division)) {
      int MAX_SIZE = 7 * 1024 * 1024;
      photosReference.child('$division.png').getData(MAX_SIZE).then((data) {
        this.setState(() {
          imageFile = data;
        });
        imageData.putIfAbsent(division, () {
          return data;
        });
      }).catchError((onError) {
        debugPrint(onError.toString());
      });
      requestedIndexes.add(division);
    }
  }

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
      body: SafeArea(
        child: Container(
          /*decoration: BoxDecoration(
                image: DecorationImage(
                  image: AssetImage('assets/$backgroundImage'),
                  fit: BoxFit.cover,
                )
            ),*/
          child: Column(
            children: <Widget>[
              Expanded(
                flex: 9,
                child: Container(
                  //child: imageFile == null ? Center(child: Text('Keine Daten')) : Image.memory(imageFile, fit: BoxFit.cover),
                  child: new PhotoView(
                    imageProvider: imageFile == null ? AssetImage('assets/error.png') : Image.memory(imageFile).image,
                    minScale: PhotoViewComputedScale.contained * 1.25,
                    maxScale: 16.0,
                  ),
                ),
              ),
              /*Expanded(
                  flex: 9,
                  child: Text(''),
                ),*/
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
        ),
      ),
    );
  }
}

