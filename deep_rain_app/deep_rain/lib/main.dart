import 'package:deep_rain/screens/ForecastList.dart';
import 'package:deep_rain/screens/ImageGrid.dart';
import 'package:deep_rain/screens/Settings.dart';
import 'package:deep_rain/screens/loading.dart';
import 'package:flutter/material.dart';
import 'package:deep_rain/screens/ForecastMap.dart';

void main() => runApp(MaterialApp(
  initialRoute: '/',
  routes: {
    '/' : (context) => Loading(),
    '/MainApp' : (context) => MainApp(),
  },
));

//void main() => runApp(new MainApp());
class MainApp extends StatefulWidget {
  @override
  State<StatefulWidget> createState() {
    return MainAppState();
  }
}
class MainAppState extends State<MainApp> {
  int _selectedTab = 0;
  final _pageOptions = [
    ForecastList(),
    //ImagesScreen(),
    ForecastMap(),
    Settings(),
  ];
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: ThemeData(
          primarySwatch: Colors.blueGrey,
          primaryTextTheme: TextTheme(
            title: TextStyle(color: Colors.white),
          )),
      home: Scaffold(
        body: _pageOptions[_selectedTab],
        bottomNavigationBar: BottomNavigationBar(
          currentIndex: _selectedTab,
          onTap: (int index) {
            setState(() {
              _selectedTab = index;
            });
          },
          items: [
            BottomNavigationBarItem(
              icon: Icon(Icons.view_quilt),
              title: Text('Zahlen'),
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.map),
              title: Text('Karte'),
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.settings),
              title: Text('Einstellungen'),
            ),
          ],
        ),
      ),
    );
  }
}