import numpy as np
import pandas as pd
import sklearn as sk

PATH_TO_TRAIN_DATA = "../data/train_data.csv"
PATH_TO_TEST_DATA = "../data/train_data.csv"
PATH_TO_WEATHER_DATA = "../data/all_weather_data.csv"


class FlightPredictor:
    def __init__(self, path_to_data='', path_to_weather=''):
        """
        Initialize an object from this class.
        @param path_to_weather: The path to a csv file containing weather data.
        """
        raise NotImplementedError

    def predict(self, x):
        """
        Recieves a pandas DataFrame of shape (m, 15) with m flight features, and predicts their
        delay at arrival and the main factor for the delay.
        @param x: A pandas DataFrame with shape (m, 15)
        @return: A pandas DataFrame with shape (m, 2) with your prediction
        """
        raise NotImplementedError


if __name__ == '__main__':
    ml = FlightPredictor(PATH_TO_TRAIN_DATA, PATH_TO_WEATHER_DATA)
    # y_hat = ml.predict(PATH_TO_TEST_DATA)
