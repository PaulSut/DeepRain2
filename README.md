# DeepRain
<img src="https://github.com/PaulIVI/DeepRain2/blob/master/deep_rain_app/deep_rain/assets/regenschirm.png" width="200">

This project is a team project which was carried out during the MSI Master at the [HTWG Konstanz - University of Applied Sciences](https://www.htwg-konstanz.de). 

The  goal  of  the  present  work  is  to  examine  whether  it  is  possible  to  calculate  a  rainfallforecast with limited resources and to make it available to users. For the calculation of therain forecast neural networks were used. The required historical and current radar data wereobtained from the German Weather Service and then analyzed and processed.  
Furthermore, an app was developed in which the rain forecasts are visualized. 
It also offers the possibility tonotify the user in case of imminent rain.

## Results & Documentation
* Detailed documentation in german: [here](https://github.com/PaulIVI/DeepRain2/blob/master/Documentation/src/Hauptdatei.pdf)
* Summary of the work in english: [here](https://github.com/PaulIVI/DeepRain2/blob/master/Documentation/paper/paper/report.pdf)

## DeepRain App 
The apk for the app is stored here: [DeepRain App](https://github.com/PaulIVI/DeepRain2/blob/master/deep_rain_app/DeepRain.apk) <br>
Store it local on an android smartphone and just click on the file to install it. 

## Directory structure
* [deep_rain_app](https://github.com/PaulIVI/DeepRain2/tree/master/deep_rain_app): The code for the DeepRain App, the server simulator,  and the corresponding cloudfunction 
* [Documentation](https://github.com/PaulIVI/DeepRain2/tree/master/Documentation): All the Latex Documentation directorys. 
* [dwd_radardata_inspection](https://github.com/PaulIVI/DeepRain2/tree/master/dwd_radardata_inspection): Radardata inspection and convertion from binary data to PNG. 
* [DeepRain](https://github.com/PaulIVI/DeepRain2/tree/master/DeepRain): The code of the neuronal networks and the final pipeline. 
* [opticFlow](https://github.com/PaulIVI/DeepRain2/tree/master/opticFlow): approach for a baseline with optical flow

## Team
Simon Christofzik, Paul Sutter, Till Reitlinger, Prof. Dr. Oliver Dürr

10.09.2020, Konstanz
