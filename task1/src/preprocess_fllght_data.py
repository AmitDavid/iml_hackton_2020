import pandas as pd

def preprocess_filght_data(df: pd.DataFrame):
    '''
    :param df: Pandas DataFrame contain the following:
    DayOfWeek:	                     Day of Week
    FlightDate:	                     Flight Date (yyyymmdd)
    Reporting_Airline:               Unique Carrier Code. (When the same code has been used by
                                                           multiple carriers, a numeric suffix is
                                                           used for earlier users, for example, PA,
                                                           PA(1), PA(2). Use this field for analysis
                                                           across a range of years)
    Tail_Number:                     Tail Number
    Flight_Number_Reporting_Airline: Flight Number
    Origin:	                         Origin Airport
    OriginCityName:                  Origin Airport, City Name
    OriginState:                     Origin Airport, State Code
    Dest:                            Destination Airport
    DestCityName:                    Destination Airport, City Name
    DestState:                       Destination Airport, State Code
    CRSDepTime:                      Expected Departure Time (local time: hhmm)
    CRSArrTime:                      Expected Arrival Time (local time: hhmm)
    CRSElapsedTime:                  Expected Elapsed Time of Flight, in Minutes
    Distance:                        Distance between airports (miles)

    ArrDelay:                        Time difference (in minutes) from expected arrival time
    DelayFactor:                     Type fo delay
    :return: Processed data frame.
    '''
    # TODO: might make them dummies as well, check if make prediction better
    # Remove Tail_Number, OriginCityName, OriginState, DestCityName, DestState,
    del df['Tail_Number', 'OriginCityName', 'OriginState', 'DestCityName', 'DestState']

    # Get categorical features (dummies) for dayOfTheWeek, Reporting_Airline,
    # Flight_Number_Reporting_Airline, Origin, Dest
    df = pd.get_dummies(df, columns=['dayOfTheWeek', 'Reporting_Airline',
                                     'Flight_Number_Reporting_Airline'
                                     'Origin', 'Dest'], prefix='cat')

    # Get hour of CRSDepTime and CRSArrTime
    df['CRSDepTime'] = df['CRSDepTime'][:2]
    df['CRSArrTime'] = df['CRSArrTime'][:2]
    df = pd.get_dummies(df, columns=['CRSDepTime', 'CRSArrTime'], prefix='cat')

    del df['CRSDepTime', 'CRSArrTime']

    # Split dayInDate, monthInDate, yearInDate and make than dummies
    df['yearInDate'] = df['FlightDate'][:4]
    df['monthInDate'] = df['FlightDate'][4:6]
    df['dayInDate'] = df['FlightDate'][6:]
    df = pd.get_dummies(df, columns=['yearInDate', 'monthInDate', 'dayInDate'], prefix='cat')

    del df['FlightDate']

    #split ArrDelay and DelayFactor to results DataFrame
    y = df['ArrDelay', 'DelayFactor'].copy()  # TODO: check if copy() need if deleting afterward
    del df['ArrDelay', 'DelayFactor']

    # CRSElapsedTime and Distance left unchanged
    return df, y


# if __name__ == '__main__':
#     df = pd.DataFrame()
#     preprocess_time_duration_table()