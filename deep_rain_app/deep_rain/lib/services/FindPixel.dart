import 'package:matrix2d/matrix2d.dart';

/*
  includes the function to calculate the pixel in the forecast png. It will be used, to calculate the forecast for the list from the forecast pngs.
 */

class FindPixel {
  get_pixel_in_forecast_png(List longitude_list, List latitude_list, var device_position_long,var device_position_lat){
    var longitude = new List();
    longitude.add(longitude_list);

    var latitude = new List();
    latitude.add(latitude_list);

    longitude = longitude.reshape(900,900);
    latitude = latitude.reshape(900, 900);

    var current_pixel_coordinate_lat = 0;
    var current_pixel_coordinate_long = 0;

    while(true){
      //der Pixel in dem sich der 'Curser' aktuell befindet
      var current_coordinate_lat = latitude[current_pixel_coordinate_lat][current_pixel_coordinate_long];
      var current_coordinate_long = longitude[current_pixel_coordinate_lat][current_pixel_coordinate_long];
      //die Distanz zu den Koordinaten des Gerätes von dem Pixel in dem sich der Cursor aktuell befindet
      var current_distance = (current_coordinate_lat - device_position_lat)*(current_coordinate_lat - device_position_lat) + (current_coordinate_long - device_position_long)*(current_coordinate_long - device_position_long);

      // nach rechts gehen?
      // die Distanz des rechten nachbarpixels zu den Koordinaten des Gerätes
      var next_coordinate_lat = latitude[current_pixel_coordinate_lat + 1][current_pixel_coordinate_long];
      var next_coordinate_long = longitude[current_pixel_coordinate_lat + 1][current_pixel_coordinate_long];
      var next_distance = (next_coordinate_lat - device_position_lat)*(next_coordinate_lat - device_position_lat) + (next_coordinate_long - device_position_long)*(next_coordinate_long - device_position_long);

      if(next_distance < current_distance){
        current_pixel_coordinate_lat = current_pixel_coordinate_lat + 1;
      }else{
        // nach oben gehen?
        var next_coordinate_lat = latitude[current_pixel_coordinate_lat][current_pixel_coordinate_long + 1];
        var next_coordinate_long = longitude[current_pixel_coordinate_lat][current_pixel_coordinate_long + 1];
        var next_distance = (next_coordinate_lat - device_position_lat)*(next_coordinate_lat - device_position_lat) + (next_coordinate_long - device_position_long)*(next_coordinate_long - device_position_long);
        if(next_distance < current_distance){
          current_pixel_coordinate_long = current_pixel_coordinate_long + 1;
        }
        else{
          // nach links gehen?
          var next_coordinate_lat = latitude[current_pixel_coordinate_lat - 1][current_pixel_coordinate_long];
          var next_coordinate_long = longitude[current_pixel_coordinate_lat - 1][current_pixel_coordinate_long];
          var next_distance = (next_coordinate_lat - device_position_lat)*(next_coordinate_lat - device_position_lat) + (next_coordinate_long - device_position_long)*(next_coordinate_long - device_position_long);

          if(next_distance < current_distance){
            current_pixel_coordinate_lat = current_pixel_coordinate_lat - 1;
          }
          else{
            // nach unten gehen?
            var next_coordinate_lat = latitude[current_pixel_coordinate_lat][current_pixel_coordinate_long - 1];
            var next_coordinate_long = longitude[current_pixel_coordinate_lat][current_pixel_coordinate_long] - 1;
            var next_distance = (next_coordinate_lat - device_position_lat)*(next_coordinate_lat - device_position_lat) + (next_coordinate_long - device_position_long)*(next_coordinate_long - device_position_long);

            if(next_distance < current_distance){
              current_pixel_coordinate_long = current_pixel_coordinate_long - 1;
            }
            else{
              break;
            }
          }
        }
      }
    }
    return [current_pixel_coordinate_lat, current_pixel_coordinate_long];
  }
}