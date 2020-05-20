import numpy as np
from Utils.Data import dataWrapper
from tensorflow.keras.models import load_model
from Utils.Data import dataWrapper
from Utils.transform import *
from Utils.loadset import getDataSet
from Utils.Data import dataWrapper
import imageio
import PIL.Image
import os
import sklearn.metrics
import pickle
import codecs
from xhtml2pdf import pisa
import time
# import plotly.plotly as py
# from plotly.graph_objs import *
import plotly.express as px
import tensorflow as tf


class Evaluation(object):

    def __init__(self,
                 model_name,
                 Data,
                 batch_size,
                 channels,
                 dimension,
                 number_of_predictions,
                 time_steps,
                 model=False,
                 transform_input=None,
                 transform_predictions=None,
                 flatten=False):
        self.model = model
        self.model_name = model_name
        self.batch_size = batch_size
        self.channels = channels
        self.dimension = dimension
        self.number_of_predictions = number_of_predictions
        self.time_steps = time_steps
        self.transform_predictions = transform_predictions
        self.transform_input = transform_input
        self.EVAL_DIR = 'Evaluation'

        self.eval_model_dir = os.path.join(self.EVAL_DIR, self.model_name)
        self.path_to_actual_weather = os.path.join(self.EVAL_DIR, self.model_name, 'actual_weather')
        self.path_to_predictions = os.path.join(self.EVAL_DIR, self.model_name, 'predictions')
        self.path_to_video = os.path.join(self.EVAL_DIR, self.model_name, 'video')
        self.path_to_video_frames = os.path.join(self.path_to_video, 'frames')

        self.train, self.test = dataWrapper(Data,
                                            dimension=dimension,
                                            channels=channels,
                                            batch_size=batch_size,
                                            flatten=flatten)

    def make_predictions(self):
        # create dir structur
        try:

            os.makedirs(os.path.join(self.EVAL_DIR, self.model_name), )
            os.makedirs(self.path_to_actual_weather)
            os.makedirs(self.path_to_predictions)
        except Exception as e:
            print('Prediction already made.\n', e)
            return
        print('Start to make the predictions. This might take a while')
        for point_of_time in range(self.time_steps):
            print('Currently at time step', point_of_time + 1, 'of', self.time_steps)
            actual_weather = np.reshape(self.train[point_of_time][1], self.dimension)
            actual_weather = np.asanyarray(actual_weather, dtype=np.int8)
            actual_weather = PIL.Image.fromarray(actual_weather, mode='L')
            actual_weather.save(
                os.path.join(self.EVAL_DIR, self.model_name, 'actual_weather', str(point_of_time) + '.png'))
            weather_history = self.train[point_of_time][0]
            os.makedirs(os.path.join(self.path_to_predictions, str(point_of_time)))
            for prediction in range(self.number_of_predictions):
                if not self.model:
                    # Baseline "Wheater does not change"
                    forecast = weather_history[:, :, :, 4:]
                    forecast = np.reshape(forecast, (self.dimension))
                else:
                    forecast = self.model.predict(weather_history)

                if self.transform_predictions is not None:
                    for operation in self.transform_predictions:
                        forecast = operation(forecast)
                else:
                    forecast *= 255
                    forecast[forecast < 0] = 0
                    forecast[forecast > 255] = 255
                    forecast = np.reshape(forecast, self.dimension)

                #print(np.unique(forecast, return_counts=True))

                forecast_img = PIL.Image.fromarray(np.asanyarray(forecast, dtype=np.int8), mode='L')
                forecast_img.save(os.path.join(self.path_to_predictions, str(point_of_time), str(prediction) + '.png'))

                if self.transform_input is not None:
                    for operation in self.transform_input:
                        forecast = operation(forecast)

                forecast = np.reshape(forecast, (1, self.dimension[0], self.dimension[1], 1))
                weather_history = np.concatenate((weather_history, forecast), axis=3)
                weather_history = weather_history[:, :, :, 1:]

        for point_of_time in range(self.time_steps, self.time_steps + self.number_of_predictions):
            actual_weather = np.reshape(self.train[point_of_time][1], self.dimension)
            actual_weather = np.asanyarray(actual_weather, dtype=np.int8)
            actual_weather = PIL.Image.fromarray(actual_weather, mode='L')
            actual_weather.save(
                os.path.join(self.EVAL_DIR, self.model_name, 'actual_weather', str(point_of_time) + '.png'))

    def create_grayscale_gif(self, time_steps=None):
        '''
        Creates a grayscale gif from prediction and actual weather.
        time_steps defines how many frames the gif has (None -> all time steps which are available)
        '''

        BAR_DIAMETER = 2
        if time_steps is None:
            time_steps = self.time_steps

        # create dirs
        try:
            os.makedirs(self.path_to_video_frames)
        except Exception as e:
            print('Grayscale GIF already made.\n', e)
            return

        # create frames
        print('Create Frames for Video')
        y_shape, x_shape = np.asanyarray(PIL.Image.open(
            os.path.join(self.path_to_predictions, str(0), str(0) + '.png'), mode='r')).shape

        for point_of_time in range(time_steps):
            forecast_array = np.full((y_shape, BAR_DIAMETER), 255)
            for image_name in range(self.number_of_predictions):
                img_array = np.asanyarray(PIL.Image.open(
                    os.path.join(self.path_to_predictions, str(point_of_time), str(image_name) + '.png'), mode='r'))
                forecast_array = np.concatenate(
                    (forecast_array, img_array, np.full((img_array.shape[0], BAR_DIAMETER), 255)), axis=1)

            actual_weather = np.full((y_shape, BAR_DIAMETER), 255)
            for image_name_offset in range(self.number_of_predictions):
                img_array = np.asanyarray(PIL.Image.open(
                    os.path.join(self.path_to_actual_weather, str(point_of_time + image_name_offset) + '.png'),
                    mode='r'))
                actual_weather = np.concatenate(
                    (actual_weather, img_array, np.full((img_array.shape[0], BAR_DIAMETER), 255)), axis=1)

            frame = np.concatenate(
                (forecast_array, np.full((BAR_DIAMETER, forecast_array.shape[1]), 255), actual_weather))

            frame = np.asanyarray(frame, dtype=np.int8)
            PIL.Image.fromarray(frame, mode='L').save(
                os.path.join(self.path_to_video_frames, str(point_of_time) + '.png'))

        # create video
        print('Done.\nStart to create Video')

        with imageio.get_writer(os.path.join(self.path_to_video, self.model_name + '.gif'), mode='I') as writer:
            for frame_name in range(time_steps):
                frame = imageio.imread(os.path.join(self.path_to_video_frames, str(frame_name) + '.png'))
                writer.append_data(frame)
        print('Done')

    def evaluation(self, evaluation_time_steps=None, rain_no_rain=False, four_categories=False,
                   weather_changes_two_categories=False):
        '''
        Evaluates the predictions

        evaluation_time_steps defines how many frames get evaluated (None -> all time steps which are available)
        '''

        if evaluation_time_steps is None:
            evaluation_time_steps = self.time_steps

        # create dirs
        try:
            os.makedirs(os.path.join(self.eval_model_dir, 'evaluation'))
        except FileExistsError:
            print('Evaluation already made.\n')
            return
        except Exception as e:
            print('Error: ', e)

        for prediction_time_step in range(self.number_of_predictions):
            print('Currently evaluating prediction time step', prediction_time_step + 1, 'of',
                  self.number_of_predictions)
            path_to_current_eval_time_step = os.path.join(self.eval_model_dir, 'evaluation',
                                                          str((1 + prediction_time_step) * 5) + '_minute_prediction')
            os.makedirs(path_to_current_eval_time_step)

            # different confusion matrices
            rain_no_rain_confusion = np.zeros((2, 2))
            four_categories_confusion = np.zeros((4, 4))
            weather_changes_two_cat_confusion = np.zeros((3, 3))
            previous_predictions, previous_actual = np.asanyarray([]), np.asanyarray([])
            for evaluation_step in range(evaluation_time_steps):
                prediction_array = np.asanyarray(PIL.Image.open(
                    os.path.join(self.path_to_predictions, str(evaluation_step), str(prediction_time_step) + '.png'),
                    mode='r'))

                actual_array = np.asanyarray(PIL.Image.open(
                    os.path.join(self.path_to_actual_weather, str(evaluation_step + prediction_time_step) + '.png'),
                    mode='r'))

                # different evaluations
                if rain_no_rain:
                    rain_no_rain_confusion += self.rain_no_rain_eval(prediction_array, actual_array)

                if four_categories:
                    four_categories_confusion += self.four_categories_eval(prediction_array, actual_array)

                if weather_changes_two_categories:
                    weather_changes_two_cat_confusion += self.weather_changes_two_categories_eval(previous_predictions,
                                                                                                  prediction_array,
                                                                                                  previous_actual,
                                                                                                  actual_array)
                previous_actual = np.asanyarray(actual_array)
                previous_predictions = np.asanyarray(prediction_array)

            # save confusion matrices
            if rain_no_rain:
                self.save_confusion_matrix(rain_no_rain_confusion, path_to_current_eval_time_step, 'rain_no_rain')

            if four_categories:
                self.save_confusion_matrix(four_categories_confusion, path_to_current_eval_time_step, 'four_categories')

            if weather_changes_two_categories:
                self.save_confusion_matrix(weather_changes_two_cat_confusion, path_to_current_eval_time_step,
                                           'weather_changes_two_categories')

    def create_report(self, normalize=True, rain_no_rain=True, four_categories=True,
                      weather_changes_two_categories=True):
        print('Start to create the report')
        if normalize:
            file_ending = '_normalized'
        else:
            file_ending = ''

        for prediction_time_step in range(self.number_of_predictions):
            path_to_current_eval_time_step = os.path.join(self.eval_model_dir, 'evaluation',
                                                          str((1 + prediction_time_step) * 5) + '_minute_prediction')

            try:
                os.makedirs(os.path.join(path_to_current_eval_time_step, 'diagramms'))

            except FileExistsError:
                print('Diagramms already made.\n')
            except Exception as e:
                print('Error: ', e)

            if rain_no_rain:
                data = pickle.load(
                    open(os.path.join(path_to_current_eval_time_step, 'rain_no_rain' + file_ending + '.p'), 'rb'))
                fig = px.imshow(data,
                                labels=dict(x="Actual Class", y="Predicted Class"),
                                x=['No Rain', 'Rain'],
                                y=['No Rain', 'Rain']
                                )
                # fig.update_xaxes(side="top")
                fig.update_layout(title=str((1 + prediction_time_step) * 5) + ' minute prediction', title_x=0.5,
                                  title_y=1)
                fig.write_html(
                    os.path.join(path_to_current_eval_time_step, 'diagramms', 'rain_no_rain' + file_ending + '.html'),
                    full_html=False)

            if four_categories:
                data = pickle.load(
                    open(os.path.join(path_to_current_eval_time_step, 'four_categories' + file_ending + '.p'), 'rb'))
                fig = px.imshow(data,
                                labels=dict(x="Actual Class", y="Predicted Class"),
                                x=['No Rain', 'light rain intensity', 'medium rain intensity', 'strong rain intensity'],
                                y=['No Rain', 'light rain intensity', 'medium rain intensity', 'strong rain intensity']
                                )
                # fig.update_xaxes(side="top")
                fig.update_layout(title=str((1 + prediction_time_step) * 5) + ' minute prediction', title_x=0.5,
                                  title_y=1)
                fig.write_html(
                    os.path.join(path_to_current_eval_time_step, 'diagramms',
                                 'four_categories' + file_ending + '.html'),
                    full_html=False)

            if weather_changes_two_categories:
                data = pickle.load(
                    open(os.path.join(path_to_current_eval_time_step,
                                      'weather_changes_two_categories' + file_ending + '.p'), 'rb'))
                fig = px.imshow(data,
                                labels=dict(x="Actual Class", y="Predicted Class"),
                                x=['Weather remains as it is', 'It will start to rain', 'Rain will stop'],
                                y=['Weather remains as it is', 'It will start to rain', 'Rain will stop']
                                )
                # fig.update_xaxes(side="top")
                fig.update_layout(title=str((1 + prediction_time_step) * 5) + ' minute prediction', title_x=0.5,
                                  title_y=1)
                fig.write_html(
                    os.path.join(path_to_current_eval_time_step, 'diagramms',
                                 'weather_changes_two_categories' + file_ending + '.html'),
                    full_html=False)

        head = '''
            <p style="margin-bottom: 0in; line-height: 100%;" align="center"><span style="font-family: Calibri, sans-serif;"><span style="font-size: xx-large;">DeepRain Evaluation Report</span></span></p>
            <p style="margin-bottom: 0in; line-height: 100%;">&nbsp;</p>
            <p style="margin-bottom: 0in; line-height: 100%;" align="center"><span style="font-family: Calibri, sans-serif;"><span style="font-size: xx-large;">''' + self.model_name + '''</span></span></p>
            <br>
            <br>
            <br>
            '''

        # create final report
        print('Create th final report')
        interactive_report = head
        if rain_no_rain:
            interactive_report = self.add_rain_no_rain_report(interactive_report, file_ending)
        if four_categories:
            interactive_report = self.add_four_categories_report(interactive_report, file_ending)
        if weather_changes_two_categories:
            interactive_report = self.add_weather_changes_two_categories_report(interactive_report, file_ending)

        with open(os.path.join(self.eval_model_dir, 'evaluation', 'final_report' + file_ending + '.html'), "w") as file:
            file.write(interactive_report)

        print('Done.')

        # self.convert_html_to_pdf(static_report,
        #                         os.path.join(self.eval_model_dir, 'evaluation', 'final_report' + file_ending + '.pdf'))

    def add_four_categories_report(self, interactive_report, file_ending):
        cm = pickle.load(
            open(os.path.join(self.eval_model_dir, 'evaluation', '5_minute_prediction', 'four_categories' + '.p'),
                 "rb"))

        number_of_datapoints = int(cm.sum())
        number_of_no_rain_datapoints = int(cm.T.sum(axis=0)[0])
        number_of_light_datapoints = int(cm.T.sum(axis=0)[1])
        number_of_medium_datapoints = int(cm.T.sum(axis=0)[2])
        number_of_strong_datapoints = int(cm.T.sum(axis=0)[3])

        rain_no_rain_intro = f'''<p style="margin-bottom: 0in; line-height: 100%;" align="justify"><span style="font-family: Calibri, sans-serif;"><span style="font-size: xx-large;">Four categories evaluation</span></span></p>
                                                    <p style="margin-bottom: 0in; line-height: 100%;" align="justify">&nbsp;</p>
                                                    <p style="margin-bottom: 0in; line-height: 100%;" align="justify"><span style="font-family: Calibri, sans-serif;"><span style="font-size: small;">In the following section the model performance is evaluated by dividing the weather into four categories: no rain, light rain intensity, medium rain intensity and strong rain intensity. </span></span><span style="font-family: Calibri, sans-serif;"><span style="font-size: small;">Therefore {number_of_datapoints} data points were analyzed out of which {number_of_no_rain_datapoints} were of the class no rain, {number_of_light_datapoints} were of the class light rain intensity, {number_of_medium_datapoints} were of the class medium rain intensity and {number_of_strong_datapoints} were of the class strong rain intensity</span></span></p>
                                                    <br>
                                                    <br>
                                                    <br>
                                                    '''
        interactive_report += rain_no_rain_intro
        for prediction_time_step in range(self.number_of_predictions):
            path_to_current_eval_time_step_diagramms = os.path.join(self.eval_model_dir, 'evaluation',
                                                                    str((
                                                                                1 + prediction_time_step) * 5) + '_minute_prediction')
            graph = codecs.open(os.path.join(path_to_current_eval_time_step_diagramms, 'diagramms',
                                             'four_categories' + file_ending + '.html'), 'r').read()

            _interactive_block = self.report_block_template(graph_html=graph, interactive=True, caption='')
            interactive_report += _interactive_block

        return interactive_report

    def add_rain_no_rain_report(self, interactive_report, file_ending):
        cm = pickle.load(
            open(os.path.join(self.eval_model_dir, 'evaluation', '5_minute_prediction', 'rain_no_rain' + '.p'), "rb"))

        number_of_datapoints = int(cm.sum())
        number_of_no_rain_datapoints = int(cm.T.sum(axis=0)[0])
        number_of_rain_datapoints = int(cm.T.sum(axis=0)[1])

        rain_no_rain_intro = f'''<p style="margin-bottom: 0in; line-height: 100%;" align="justify"><span style="font-family: Calibri, sans-serif;"><span style="font-size: xx-large;">Rain/ no rain evaluation</span></span></p>
                                            <p style="margin-bottom: 0in; line-height: 100%;" align="justify">&nbsp;</p>
                                            <p style="margin-bottom: 0in; line-height: 100%;" align="justify"><span style="font-family: Calibri, sans-serif;"><span style="font-size: small;">In the following section the model performance is evaluated by dividing the weather into two categories: rain/no rain. </span></span><span style="font-family: Calibri, sans-serif;"><span style="font-size: small;">Therefore {number_of_datapoints} data points were analyzed out of which {number_of_no_rain_datapoints} were of the class no rain and {number_of_rain_datapoints} were of the class rain</span></span></p>
                                            <br>
                                            <br>
                                            <br>
                                            '''
        interactive_report += rain_no_rain_intro
        for prediction_time_step in range(self.number_of_predictions):
            path_to_current_eval_time_step_diagramms = os.path.join(self.eval_model_dir, 'evaluation',
                                                                    str((
                                                                                1 + prediction_time_step) * 5) + '_minute_prediction')
            graph = codecs.open(os.path.join(path_to_current_eval_time_step_diagramms, 'diagramms',
                                             'rain_no_rain' + file_ending + '.html'), 'r').read()

            _interactive_block = self.report_block_template(graph_html=graph, interactive=True, caption='')
            interactive_report += _interactive_block

        return interactive_report

    def add_weather_changes_two_categories_report(self, interactive_report, file_ending):
        cm = pickle.load(
            open(os.path.join(self.eval_model_dir, 'evaluation', '5_minute_prediction',
                              'weather_changes_two_categories' + '.p'), "rb"))

        number_of_datapoints = int(cm.sum())
        number_of_weather_remains = int(cm.T.sum(axis=0)[0])
        number_of_start_rain = int(cm.T.sum(axis=0)[1])
        number_of_stop_rain = int(cm.T.sum(axis=0)[2])

        rain_no_rain_intro = f'''<p style="margin-bottom: 0in; line-height: 100%;" align="justify"><span style="font-family: Calibri, sans-serif;"><span style="font-size: xx-large;">Weather changes evaluation</span></span></p>
                                                    <p style="margin-bottom: 0in; line-height: 100%;" align="justify">&nbsp;</p>
                                                    <p style="margin-bottom: 0in; line-height: 100%;" align="justify"><span style="font-family: Calibri, sans-serif;"><span style="font-size: small;">In the following section the model performance is evaluated by dividing the weather into two categories: rain/no rain. </span></span><span style="font-family: Calibri, sans-serif;"><span style="font-size: small;">Therefore {number_of_datapoints} data points were analyzed out of which {number_of_weather_remains} were of the class "weather will remain the same", {number_of_start_rain} were of the class "weather changes to rain" and {number_of_stop_rain} were of the class "the rain will stop"</span></span></p>
                                                    <br>
                                                    <br>
                                                    <br>
                                                    '''
        interactive_report += rain_no_rain_intro
        for prediction_time_step in range(self.number_of_predictions):
            path_to_current_eval_time_step_diagramms = os.path.join(self.eval_model_dir, 'evaluation',
                                                                    str((
                                                                                1 + prediction_time_step) * 5) + '_minute_prediction')
            graph = codecs.open(os.path.join(path_to_current_eval_time_step_diagramms, 'diagramms',
                                             'weather_changes_two_categories' + file_ending + '.html'), 'r').read()

            _interactive_block = self.report_block_template(graph_html=graph, interactive=True, caption='')
            interactive_report += _interactive_block

        return interactive_report

    def report_block_template(self, graph_html, interactive=True, caption=''):
        if interactive:
            graph_block = graph_html
        else:
            print('Not implemented yet')
            graph_block = ('<img style="height: 400px;" src="{graph_html}.png">'
                           '</a>')

        report_block = ('' +
                        graph_block +
                        caption +  # Optional caption to include below the graph
                        '<br>' +  # Line break
                        '<br>' +
                        '<hr>')  # horizontal line

        return report_block

    def convert_html_to_pdf(self, source_html, output_filename):
        # open output file for writing (truncated binary)
        result_file = open(output_filename, "w+b")

        # convert HTML to PDF
        pisa_status = pisa.CreatePDF(
            source_html,  # the HTML to convert
            dest=result_file)  # file handle to recieve result

        # close output file
        result_file.close()  # close output file

        # return True on success and False on errors
        return pisa_status.err

    def save_confusion_matrix(self, cm, path_to_current_eval_time_step, name):
        pickle.dump(cm,
                    open(os.path.join(path_to_current_eval_time_step, name + '.p'), "wb"))
        cm_normalized = self.normalize_confusion_matrix(cm)
        pickle.dump(cm_normalized,
                    open(os.path.join(path_to_current_eval_time_step, name + '_normalized.p'), "wb"))

    def normalize_confusion_matrix(self, cm):
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        return cm

    def rain_no_rain_eval(self, prediction_array, actual_array):
        new_prediction_array = np.zeros(prediction_array.shape)
        new_actual_array = np.zeros(actual_array.shape)

        new_prediction_array[prediction_array > 0] = 1
        new_actual_array[actual_array > 0] = 1

        new_prediction_array = new_prediction_array.flatten()
        new_actual_array = new_actual_array.flatten()

        confusion_matrix = sklearn.metrics.confusion_matrix(new_actual_array, new_prediction_array)

        return confusion_matrix

    def four_categories_eval(self, prediction_array, actual_array):
        new_prediction_array = np.zeros(prediction_array.shape)
        new_actual_array = np.zeros(actual_array.shape)

        new_prediction_array[(prediction_array > 0) & (prediction_array <= 2)] = 1
        new_prediction_array[(prediction_array > 2) & (prediction_array <= 10)] = 2
        new_prediction_array[prediction_array > 10] = 3

        new_actual_array[(actual_array > 0) & (actual_array <= 2)] = 1
        new_actual_array[(actual_array > 2) & (actual_array <= 10)] = 2
        new_actual_array[actual_array > 10] = 3

        new_prediction_array = new_prediction_array.flatten()
        new_actual_array = new_actual_array.flatten()

        confusion_matrix = sklearn.metrics.confusion_matrix(new_actual_array, new_prediction_array)

        return confusion_matrix

    def weather_changes_two_categories_eval(self, previous_predictions, current_predictions, previous_actual,
                                            current_actual):

        if previous_actual.shape != self.dimension:
            return np.zeros((3, 3))
        # create new arrays
        new_previous_predictions_array = np.zeros(current_actual.shape)
        new_current_prediction_array = np.zeros(current_actual.shape)
        new_previous_actual = np.zeros(current_actual.shape)
        new_current_actual_array = np.zeros(current_actual.shape)

        # fill arrays with 1 for rain 0 no rain
        new_previous_predictions_array[previous_predictions > 0] = 1
        new_current_prediction_array[current_predictions > 0] = 1
        new_previous_actual[previous_actual > 0] = 1
        new_current_actual_array[current_actual > 0] = 1

        '''
        Weather stays as it is : 0 
        starts to rain: 1
        rain stops: 2
        '''
        prediction = np.zeros(current_actual.shape)
        prediction = np.where((new_previous_predictions_array == 0) & (new_current_prediction_array == 1),
                              1, prediction)
        prediction = np.where((new_previous_predictions_array == 1) & (new_current_prediction_array == 0),
                              2, prediction)

        actual = np.zeros(current_actual.shape)
        actual = np.where((new_previous_actual == 0) & (new_current_actual_array == 1), 1, actual)
        actual = np.where((new_previous_actual == 1) & (new_current_actual_array == 0), 2, actual)

        actual = actual.flatten()
        prediction = prediction.flatten()
        confusion_matrix = sklearn.metrics.confusion_matrix(actual, prediction)

        return confusion_matrix


DatasetFolder = "./Data/RAW"
PathToData = os.path.join(DatasetFolder, "MonthPNGData")


def provideData(dimension, batch_size, channels, flatten=False, transform_input=None,
                transform_output=None, preTransformation=None):
    getDataSet(DatasetFolder, year=[2017])
    train, test = dataWrapper(PathToData,
                              dimension=dimension,
                              channels=channels,
                              batch_size=batch_size,
                              overwritecsv=True,
                              flatten=flatten,
                              onlyUseYears=[2017],
                              transform_input=transform_input,
                              transform_output=transform_output,
                              preTransformation=preTransformation)

    return train, test


PATH_TO_MODEL = '/home/paul/Documents/DeepRain/clean_git/DeepRain2/Networks/Utils/model_data/medium_thin_UNet64_categorical_crossentropy/medium_thin_UNet64_categorical_crossentropy448x448x5-03.hdf5'
PATH_TO_WEIGHTS = ''
DIMESNION = (448, 448)
SCLICES = [100, 548, 200, 648]
#DIMESNION = (64, 64)
#SCLICES = [256, 320, 256, 320]
NUMBER_OF_PREDICTIONS = 6
NUMBER_OF_TIMESTEPS = 20
CHANNELS = 5
BATCH_SIZE = 1

cutOutFrame = cutOut(SCLICES)

PRETRAINING_TRANSFORMATIONS = [cutOutFrame]

#model = model((*dimension, channels))
model = load_model(PATH_TO_MODEL)

'''
evaluation = Evaluation('Medium_UNet64_categorical_crossentropy_448_448_all_germany1',
                        model=model,
                        Data=provideData(batch_size=BATCH_SIZE, dimension=DIMESNION,
                                         preTransformation=PRETRAINING_TRANSFORMATIONS, transform_input=[Normalize()],
                                         transform_output=[ToCategorical([-10000, 0, 2, 10, 256])], channels=CHANNELS),
                        batch_size=BATCH_SIZE,
                        channels=CHANNELS,
                        dimension=DIMESNION,
                        number_of_predictions=NUMBER_OF_PREDICTIONS,
                        transform_input=[Normalize()],
                        transform_predictions=[from_sparse_categorical()],
                        time_steps=NUMBER_OF_TIMESTEPS
                        )
                        '''
'''
evaluation = Evaluation('Medium_thin_UNet64_mse_sortOut_448_448_64_64_LSTM',
                        model=model,
                        Data=provideData(batch_size=BATCH_SIZE, dimension=DIMESNION,
                                         preTransformation=PRETRAINING_TRANSFORMATIONS, transform_input=[Normalize()],
                                         transform_output=None, channels=CHANNELS),
                        batch_size=BATCH_SIZE,
                        channels=CHANNELS,
                        dimension=DIMESNION,
                        number_of_predictions=NUMBER_OF_PREDICTIONS,
                        transform_input=[Normalize()],
                        transform_predictions=None,
                        time_steps=NUMBER_OF_TIMESTEPS
                        )
                        '''

# Baseline:
'''
evaluation = Evaluation('Baseline',
                        model=False,
                        Data=provideData(batch_size=BATCH_SIZE, dimension=DIMESNION,
                                         preTransformation=PRETRAINING_TRANSFORMATIONS, transform_input=[Normalize()],
                                         transform_output=[], channels=CHANNELS),
                        batch_size=BATCH_SIZE,
                        channels=CHANNELS,
                        dimension=DIMESNION,
                        number_of_predictions=NUMBER_OF_PREDICTIONS,
                        transform_input=[Normalize()],
                        transform_predictions=None,
                        time_steps=NUMBER_OF_TIMESTEPS
                        )
'''

start = time.time()
print("Start to measure Time")
evaluation.make_predictions()
evaluation.create_grayscale_gif(10)
evaluation.evaluation(rain_no_rain=True, four_categories=True, weather_changes_two_categories=True)
evaluation.create_report()
end = time.time()
print("Stop to measure Time")
print('End Time:', end - start)
