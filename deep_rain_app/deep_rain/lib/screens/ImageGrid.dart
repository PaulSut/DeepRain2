import 'dart:typed_data';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:flutter/material.dart';
import 'package:deep_rain/DataObjects/DataHolder.dart';

class ImagesScreen extends StatelessWidget {

  Widget makeImagesGrid(){
    return GridView.builder(
        itemCount: 18,
        gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(crossAxisCount: 3),
        itemBuilder: (context, index){
          return ImageGridItem(index);
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Bilder Gallerie"),
      ),
      body: Container(
        child: makeImagesGrid(),
      ),
    );
  }
}


class ImageGridItem extends StatefulWidget {

  int _index;
  ImageGridItem(int index){
    this._index = index + 1;
  }

  @override
  _ImageGridItemState createState() => _ImageGridItemState();
}

class _ImageGridItemState extends State<ImageGridItem> {

  Uint8List imageFile;
  StorageReference photosReference = FirebaseStorage.instance.ref().child('photos');

  getImage(){
    if(!requestedIndexes.contains(widget._index)){
      int MAX_SIZE = 7 * 1024 * 1024;
      photosReference.child('${widget._index}.png').getData(MAX_SIZE).then((data){
        this.setState((){
          imageFile = data;
        });
        imageData.putIfAbsent(widget._index, (){
          return data;
        });
      }).catchError((onError){
        debugPrint(onError.toString());
      });
      requestedIndexes.add(widget._index);

    }
  }

  Widget decideGridTileWidget(){
    if(imageFile == null){
      return Center(child: Text('Keine Daten'));
    } else{
      return Image.memory(imageFile, fit: BoxFit.cover);
    }
  }

  @override
  void initState(){
    super.initState();
    if(!imageData.containsKey(widget._index)){
      getImage();
    }else{
      this.setState((){
        imageFile = imageData[widget._index];
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return GridTile(child: decideGridTileWidget());
  }
}
