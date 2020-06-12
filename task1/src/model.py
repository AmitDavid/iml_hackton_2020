import pickle
import pandas as pd
from Classification import *
from weather_preprocess import *
from preprocess_fllght_data import *

###########################################
#        For running trained model        #
###########################################
PATH_TO_REGRESSION_MODEL = "../pickle/reg_model"
PATH_TO_CLASSIFICATION_MODEL = "../pickle/class_model"


class FlightPredictor:
    def __init__(self, path_to_weather=''):
        """
        Initialize an object from this class.
        @param path_to_weather: The path to a csv file containing weather data.
        """
        self.__weather_df = pd.read_csv(path_to_weather, low_memory=False)

        with open(PATH_TO_REGRESSION_MODEL, 'rb') as reg_file:
            self.__reg_model = pickle.load(reg_file)
            reg_file.close()

        with open(PATH_TO_CLASSIFICATION_MODEL, 'rb') as class_file:
            self.__class_model = pickle.load(class_file)
            class_file.close()

    def predict(self, design_matrix):
        """
        Receives a pandas DataFrame of shape (m, 15) with m flight features, and predicts their
        delay at arrival and the main factor for the delay.
        @param design_matrix: A pandas DataFrame with shape (m, 15)
        @return: A pandas DataFrame with shape (m, 2) with your prediction
        """
        # Preprocess data
        df = preprocess_weather_data(design_matrix, self.__weather_df)
        design_matrix = preprocess_flight_data(df, False)
        y_delay_hat = self.__reg_model.predict(design_matrix)
        y_type_hat = self.__class_model.predict(design_matrix)

        x = {1: "CarrierDelay", 2: "LateAircraftDelay", 3: "NASDelay", 4: "WeatherDelay"}
        y_type_hat = [x[col_type] for col_type in y_type_hat]

        y = pd.DataFrame({'ArrDelay': y_delay_hat, 'DelayFactor': y_type_hat})

        return y
