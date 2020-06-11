import pandas as pa
import numpy as np
import datetime
import time
import os
import sys

NUM_OF_ARGS = 3

SNOW_THRESHOLD = 10

MAX_TEMP_THRESHOLD = 165
MY_DIR = os.path.dirname(__file__)
WEATHER_FILE_PATH = os.path.join(MY_DIR, '..', 'data', 'all_weather_data.csv')
WEATHER_JFK_PATH = os.path.join(MY_DIR, '..', 'small_data', 'weather_jfk.csv')
TRAIN_DATA_FILE_PATH = os.path.join(MY_DIR, '..', 'data', 'train_data.csv')
SMALL_TRAIN_DATA_PATH = os.path.join(MY_DIR, '..', 'small_data', 'data1k.csv')
MEDIUM_TRAIN_DATA_PATH = os.path.join(MY_DIR, '..', 'small_data', 'data10k.csv')

WEATHER_FILE = {
    'all_weather': WEATHER_FILE_PATH,
    'jfk': WEATHER_JFK_PATH
}

TRAIN_DATA = {
    '1k': SMALL_TRAIN_DATA_PATH,
    '10k': MEDIUM_TRAIN_DATA_PATH,
    'all_data': TRAIN_DATA_FILE_PATH
}
MATCH_COLS = ['day', 'station']

NUMERIC_COLS = ['max_temp_f', 'min_temp_f', 'precip_in', 'avg_wind_speed_kts', 'avg_rh', 'max_dewpoint_f',
                'min_dewpoint_f', 'avg_wind_drct', 'min_rh', 'avg_rh', 'max_rh', 'snow_in', 'snowd_in',
                'max_wind_speed_kts', 'max_wind_gust_kts']
NEW_COLS = ['avg_temp_f']
TABLE_COLS = NUMERIC_COLS + MATCH_COLS


def replace_na(df: pa.DataFrame):
    """
    Replace all na values with the mean of the column
    :param df: data set
    """
    for col in NUMERIC_COLS:
        df[col].fillna(df[col].mean(), inplace=True)


def fix_snow_cols(df: pa.DataFrame):
    """
    Fix snow columns by masking non relevant data,  only data that satisfy "0<data<SNOW_THRESHOLD" is valid.
    :param df: data set
    """
    snow_cols = ['snow_in', 'snowd_in']
    for snow_col in snow_cols:
        df[snow_col].mask(df[snow_col] > SNOW_THRESHOLD, inplace=True)
        df[snow_col].mask(df[snow_col] < 0, inplace=True)


def get_weather_df() -> pa.DataFrame:
    """
    Preprocess of the weather dataset
    """
    df = pa.read_csv(WEATHER_FILE[sys.argv[2]], low_memory=False, usecols=TABLE_COLS)
    df[NUMERIC_COLS] = df[NUMERIC_COLS].apply(pa.to_numeric, errors='coerce')
    df.dropna(subset=['max_temp_f'], inplace=True)
    df['max_temp_f'].mask(df['max_temp_f'] > MAX_TEMP_THRESHOLD)
    fix_snow_cols(df)
    replace_na(df)
    df['avg_temp_f'] = (df['max_temp_f'] + df['min_temp_f']) / 2
    df.rename(columns={'station': 'Origin'}, inplace=True)
    return df


def convert_to_datetime(flight_data_df, weather_df):
    flight_data_df['day'] = pa.to_datetime(arg=flight_data_df['FlightDate'])
    weather_df['day'] = pa.to_datetime(arg=weather_df['day'])
    dep = flight_data_df['CRSDepTime']
    arr = flight_data_df['CRSArrTime']
    flight_data_df['day_arr'] = flight_data_df['CRSDepTime'].where(dep < arr,
                                                                   flight_data_df['day'] + datetime.timedelta(days=1))
    flight_data_df['day_arr'].where(dep > arr, flight_data_df['day'], inplace=True)
    flight_data_df['day_arr'] = pa.to_datetime(arg=flight_data_df['day_arr'])


def get_dataset(data_path) -> pa.DataFrame:
    """
    Main driver to get DataFrame with the flight data combined with weather
    :return: Merged DataFrame of data and weather
    """
    start = time.time()
    weather_df = get_weather_df()
    flight_data_df = pa.read_csv(data_path, low_memory=False)
    convert_to_datetime(flight_data_df, weather_df)
    # changed date format to "datetime64 dtype", as the two are not fitting at the moment
    merged = flight_data_df.merge(weather_df, on=['Origin', 'day'], validate="m:1", how='left')
    # merged = merged.merge(weather_df, right_on=['Origin', 'day'], left_on=['Dest', 'day_arr'], validate="m:1",
    #                       how='left', suffixes=('_dep', '_arr'))
    end = time.time()
    merged.info()
    print(merged.describe().to_string())
    print("Execution time in sec: {}".format(end - start))
    return merged


def is_valid_usage():
    return len(sys.argv) == NUM_OF_ARGS and (sys.argv[1] != 'small_data' or sys.argv[1] != 'all_data') and \
           (sys.argv[2] != 'jfk' or sys.argv[2] != 'all_weather')


if __name__ == '__main__':
    if is_valid_usage():
        get_dataset(TRAIN_DATA[sys.argv[1]])
    else:
        print("Usage: python weather_preprocess.py X \n"
              "'X = 1k' for 1k flight data\n"
              "'X = 10k' for 1k flight data\n"
              "'X = all_data' for all data")
    pass
