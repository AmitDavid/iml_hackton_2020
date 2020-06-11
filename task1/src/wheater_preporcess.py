import pandas as pa
import time
import os

SNOW_THRESHOLD = 10

MAX_TEMP_THRESHOLD = 165
MY_DIR = os.path.dirname(__file__)
WEATHER_FILE_PATH = os.path.join(MY_DIR, '..', 'data', 'all_weather_data.csv')
TRAIN_DATA_FILE_PATH = os.path.join(MY_DIR, '..', 'data', 'train_data.csv')
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


def preprocess():
    """
    Preprocess of the weather dataset
    """
    df = pa.read_csv(WEATHER_FILE_PATH, low_memory=False, usecols=TABLE_COLS)
    df[NUMERIC_COLS] = df[NUMERIC_COLS].apply(pa.to_numeric, errors='coerce')
    df.dropna(subset=['max_temp_f'], inplace=True)
    df['max_temp_f'].mask(df['max_temp_f'] > MAX_TEMP_THRESHOLD)
    fix_snow_cols(df)
    replace_na(df)
    df['avg_temp_f'] = (df['max_temp_f'] + df['min_temp_f']) / 2
    return df


def main():
    """
    Main driver to get DataFrame with the flight data combined with weather
    :return: Merged DataFrame of data and weather
    """
    start = time.time()
    weather_df = preprocess()
    flight_data_df = pa.read_csv(TRAIN_DATA_FILE_PATH, low_memory=False)
    weather_df.rename(columns={'day': 'FlightDate', 'station': 'Origin'}, inplace=True)
    date_bck = flight_data_df['FlightDate'].copy()
    flight_data_df['FlightDate'] = pa.to_datetime(arg=flight_data_df['FlightDate'])
    weather_df['FlightDate'] = pa.to_datetime(arg=weather_df['FlightDate'])
    # changed date format to "datetime64 dtype", as the two are not fitting at the moment
    merged = flight_data_df.merge(weather_df, on=['Origin', 'FlightDate'], validate="m:1")
    merged.info()
    print(merged.describe().to_string())
    end = time.time()
    print("Execution time in sec: {}".format(end - start))


if __name__ == '__main__':
    main()
    pass
