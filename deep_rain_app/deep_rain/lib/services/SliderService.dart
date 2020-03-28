import'package:intl/intl.dart';

class SliderService {
  ///return the represented time of each division
  ///the first 1/3 of divisions are divisions of the past, the last 2/3 divisions are for the future
  String getTime(int numberOfDivisions, double currentRating) {
    String time;

    double oneStepInRating = 1 / numberOfDivisions; // Stepsize in the Rating
    int currentDevisionOfSlider = currentRating~/ oneStepInRating; //Check, in which division the slider is
    int boarderPastFuture = numberOfDivisions~/3; // Check, where are the boarder between past and future.
    int newDevisionOfSlider = currentDevisionOfSlider - boarderPastFuture; //New Value on the new "Timeline". The 0 is now after the first 1/3 of all Divisions.

    DateTime now = DateTime.now();
    now = now.add(Duration(minutes: (newDevisionOfSlider * 5).toInt()));
    time = DateFormat.jm().format(now);
    return time;
  }

}