import'package:intl/intl.dart';

class FindPixel {
  getClosest_Coordinate(List<dynamic> x,List<dynamic> y,var x_range_max,var x_range_min,var y_range_max,var y_range_min,var location_x,var location_y){
    if(x_range_min == x_range_max && y_range_max == y_range_min){
      print('Y_RANGE_MIN');
      print(y_range_min);
      return [y[y_range_min], x[x_range_min]];
    }
    var x_range = (x_range_max + 1 - x_range_min);
    var y_range = (y_range_max + 1 - y_range_min);

    var box_location = getBoxLocation(x_range, y_range, x_range_min, y_range_min, x, y, location_x, location_y);
    print('Box Location');
    print(box_location);

    var x_range_uneven = (x_range) % 2 == 1 ? 1 : false;
    var y_range_uneven = (y_range) % 2 == 1 ? 1 : false;

    var x_range_min_;
    var x_range_max_;
    var y_range_min_;
    var y_range_max_;

    if (box_location == 'top_left'){
      x_range_min_ = x_range_min;
      x_range_max_ = (x_range_min + x_range / 2 + 0.5 - 1).round();
      y_range_min_ = y_range_min;
      y_range_max_ = (y_range_min + y_range / 2 + 0.5 - 1).round();
    }


    else if (box_location == 'top_right'){
      x_range_min_ = (x_range_min + x_range / 2).round();
      x_range_max_ = x_range_min + x_range - 1;
      y_range_min_ = y_range_min;
      y_range_max_ = (y_range_min + y_range / 2 + 0.5 - 1).round();
    }

    else if (box_location == 'bottom_left'){
      x_range_min_ = x_range_min;
      x_range_max_ = (x_range_min + x_range / 2 + 0.5 - 1).round();
      y_range_min_ = (y_range_min + y_range / 2).round();
      y_range_max_ = y_range_min + y_range - 1;
    }


    else if (box_location == 'bottom_right'){
      x_range_min_ = (x_range_min + x_range / 2).round();
      x_range_max_ = x_range_min + x_range - 1;
      y_range_min_ = (y_range_min + y_range / 2).round();
      y_range_max_ = y_range_min + y_range - 1;
    }
    else{
      print('Error');
    }

    return getClosest_Coordinate(x, y, x_range_min_, x_range_max_, y_range_min_, y_range_max_, location_x, location_y);
  }

  getBoxLocation(var x_range,var y_range,var x_range_min,var y_range_min, List<dynamic> x, List<dynamic> y,var location_x,var location_y){
    var x_range_uneven = (x_range) % 2 == 1 ? 1 : false;
    var y_range_uneven = (y_range) % 2 == 1 ? 1 : false;

    var x_min_left = x[y_range_min][x_range_min] + ((x[y_range_min][x_range_min] - x[y_range_min][x_range_min + 1]) / 2);
    var x_min_right;
    var y_min_bottom;

    print('x_min_left');
    print(x_min_left);

    if (x_range_uneven){
      x_min_right = x[y_range_min][(x_range_min + x_range / 2).round()];
    }else{
      x_min_right = x[y_range_min][(x_range_min + x_range / 2).round()] + ((x[y_range_min][(x_range_min + x_range / 2).round() - 1] - (x_range_min + x_range / 2).round()) / 2);
    }

    print('x_min_right');
    print(x_min_right);

    var y_min_top = y[y_range_min][x_range_min] + ((y[y_range_min][x_range_min]- y[y_range_min + 1][x_range_min]) / 2);
    if (y_range_uneven){
      y_min_bottom = y[(y_range_min + y_range / 2).round()][x_range_min];
    }
    else{
      y_min_bottom = y[(y_range_min + y_range / 2).round()][x_range_min] + ((y[(y_range_min + y_range / 2).round() - 1][x_range_min]- (y_range_min + y_range / 2).round()) / 2);
    }

    //top left
    if (location_is_in_box(location_x, location_y, x_min_left, x_min_right, y_min_top, y_min_bottom)){
     return 'top_left';
    }

    var x_max_right = x[y_range_min][x_range_min + x_range - 1] + ((x[y_range_min][x_range_min + x_range - 1] - x[y_range_min][x_range_min + x_range - 2]) / 2);

    //top right
    if (location_is_in_box(location_x, location_y, x_min_right, x_max_right, y_min_top, y_min_bottom)){
      return 'top_right';
    }

    var y_max_bottom = y[y_range_min + y_range - 1][x_range_min] + ((y[y_range_min + y_range - 1][x_range_min] - y[y_range_min + y_range - 2][x_range_min]) / 2);

    //bottom left
    if (location_is_in_box(location_x, location_y, x_min_left, x_min_right, y_min_bottom, y_max_bottom)){
      return 'bottom_left';
    }

    //bottom right
    if (location_is_in_box(location_x, location_y, x_min_right, x_max_right, y_min_bottom, y_max_bottom)){
      return 'bottom_right';
    }
    else{
      print('Error in getBoxLocation');
    }


  }

  bool location_is_in_box(location_x, location_y, x_min, x_max, y_min, y_max){
    if ((x_max >= location_x) && (location_x >= x_min) && (y_max >= location_y) && (location_y >= y_min)){
      return true;
    }else{
      return false;
    }
  }


}