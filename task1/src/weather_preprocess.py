import datetime
import time

import pandas as pa

SNOW_THRESHOLD = 10

MAX_TEMP_THRESHOLD = 165

MATCH_COLS = ['day', 'station']

NUMERIC_COLS = ['max_temp_f', 'min_temp_f', 'precip_in', 'avg_wind_speed_kts', 'avg_rh',
                'max_dewpoint_f',
                'min_dewpoint_f', 'avg_wind_drct', 'min_rh', 'avg_rh', 'max_rh', 'snow_in',
                'snowd_in',
                'max_wind_speed_kts', 'max_wind_gust_kts']
NEW_COLS = ['avg_temp_f']
RELEVANT_COLS = NUMERIC_COLS + MATCH_COLS


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


def drop_unused_cols(df: pa.DataFrame):
    for col in df:
        if col not in RELEVANT_COLS:
            df.drop(columns=col, inplace=True)


def preprocess_weather_df(df: pa.DataFrame) -> pa.DataFrame:
    """
    Preprocess of the weather dataset
    """
    df[NUMERIC_COLS] = df[NUMERIC_COLS].apply(pa.to_numeric, errors='coerce')

    # drops rows with no info (as max_temp_f indicates it)
    df.dropna(subset=['max_temp_f'], inplace=True)
    df['max_temp_f'].mask(df['max_temp_f'] > MAX_TEMP_THRESHOLD)

    fix_snow_cols(df)
    replace_na(df)
    drop_unused_cols(df)

    df['avg_temp_f'] = (df['max_temp_f'] + df['min_temp_f']) / 2
    df.rename(columns={'station': 'Origin'}, inplace=True)
    return df


def handle_dates(flight_data_df: pa.DataFrame, weather_df: pa.DataFrame):
    """
    Converts date format columns to datetime, adding arrival date
    :param flight_data_df: The flight train data
    :param weather_df: The weather train data
    """
    flight_data_df['day'] = pa.to_datetime(arg=flight_data_df['FlightDate'])
    weather_df['day'] = pa.to_datetime(arg=weather_df['day'])
    dep = flight_data_df['CRSDepTime']
    arr = flight_data_df['CRSArrTime']
    flight_data_df['day_arr'] = flight_data_df['CRSDepTime'].where(dep < arr,
                                                                   flight_data_df[
                                                                       'day'] + datetime.timedelta(
                                                                       days=1))
    flight_data_df['day_arr'].where(dep > arr, flight_data_df['day'], inplace=True)
    flight_data_df['day_arr'] = pa.to_datetime(arg=flight_data_df['day_arr'])


def preprocess_weather_data(flight_data_df: pa.DataFrame, weather_df: pa.DataFrame) -> pa.DataFrame:
    """
    Main driver to get DataFrame with the flight data combined with weather
    :return: Merged DataFrame of data and weather, added day_arr column representing the arrival date of the flight
    """
    preprocess_weather_df(weather_df)
    # changed date format to "datetime64 dtype", as the two are not fitting at the moment
    handle_dates(flight_data_df, weather_df)

    # merge by departure date and origin
    merged = flight_data_df.merge(weather_df, on=['Origin', 'day'], validate="m:1", how='left')

    # merge by arrival date and destination
    weather_df.rename(columns={'Origin': 'Dest', 'day': 'day_arr'}, inplace=True)
    merged = merged.merge(weather_df, on=['Dest', 'day_arr'], validate="m:1", how='left',
                          suffixes=('_dep', '_arr'))

    # convert day_arr back to the same format as FlightDate, dropping the col that were required for merging
    merged['day_arr'] = merged['day_arr'].dt.strftime("%Y-%m-%d")
    merged.drop(columns='day', inplace=True)

    return merged
