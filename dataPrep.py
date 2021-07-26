import pandas as pd
import numpy as np
from math import sin, cos, sqrt, atan2, radians
from geographiclib.geodesic import Geodesic

# RMSE metric to evaluate predictions
def rmse_vectorized(actual, forecast):
    forecast = np.array(forecast)
    return float(np.mean((forecast - actual) ** 2) ** 0.5)

# MAPE metric to evaluate predictions
def mape_vectorized_v2(a, b):
    print(type(a), type(b))
    if type(a) == list:
        a = np.array(a)
    if type(b) == list:
        b = np.array(b)

    mask = np.array(a) != 0.0

    a = a[mask]
    b = b[mask]

    return float((abs(b - a)/a).mean())

# MAE metric to evaluate predictions
def mae_vectorized(a, b):
    b = np.array(b)
    # mask = a != 0
    return float(abs(a - b).mean())

# how to detect consecutive values on raw data so we disregard those target cameras with many consecutive values-we use a threshold e.g. two days
def consecutive_values(df, val, thresh, xcpt):
    columns_above_thresh = []
    for column in df.columns:
        if column in xcpt:
            continue
        max_count = 0
        counter = 0
        for row in df[column]:
            if row == val:
                counter += 1
            else:
                max_count = max(counter, max_count)
                counter = 0
        # print(column, max_count)
        if max_count >= thresh:
            columns_above_thresh.append(column)
    return columns_above_thresh

# data preparation from multi cameras for prediction
def MulticamDataPrepForPredict(inputData, lag, missing, target, agg_period, agg_method, columns, lead=1):
    columns = columns + ['datetime']
    inputData = inputData[inputData['camera'].isin(columns)]

    inputData['datetime'] = pd.to_datetime(inputData['datetime'])
    inputPivot = inputData.pivot_table(index='datetime', columns='camera', values='count')

    inputPivot['day'] = pd.DatetimeIndex(inputPivot.index).dayofweek
    inputPivot['weekday'] = (inputPivot['day'] // 5 == 1).astype(float)
    if agg_method == 'mean':
        inputPivot = inputPivot.resample(rule=agg_period).mean()
    else:
        inputPivot = inputPivot.resample(rule=agg_period).max()

    inputPivot.fillna(missing, inplace=True)

    inputPivot['datetime'] = inputPivot.index
    inputPivot['datetime'] = inputPivot.datetime.dt.strftime('%Y-%m-%d %H:%M')
    inputPivot['serial'] = inputPivot.index.hour * 4 + inputPivot.index.minute.astype('int')/15
    inputPivot['15MinMidnight'] = inputPivot['serial'].apply(lambda x: sqrt((0 - x)**2))
    inputPivot['15MinMidday'] = inputPivot['serial'].apply(lambda x: sqrt((48 - x)**2))

    inputPivot = inputPivot.drop(['serial'], axis=1)
    trainingData = []
    datetimes = []

    inputPivotLen = len(inputPivot)

    for clm in columns:
        if not clm in list(inputPivot.columns):
            inputPivot[clm] = -1

    inputPivot = inputPivot[columns]  # re-arrange columns

    if target not in inputPivot.columns:
        return False, 0, 0
    # print(inputPivot.iloc[0])
    for i in range(0, inputPivotLen - (lag+lead-1)):
        thisRow = []
        for j in range(0, lag+1):
            index = i+j
            for column in inputPivot.columns:
                # print(i, j, index, lag, lead)
                # print(inputPivot.iloc[index + lead - 1])
                if j == lag:
                    if column == 'datetime':
                        datetimes.append(inputPivot.iloc[index+lead-1][column])

                    if column == target:
                        targetValue = inputPivot.iloc[index+lead-1][column]
                    else:
                        continue
                else:
                    if column == 'datetime':
                        continue
                    #print(i, j, index, lag, lead)
                    thisRow.append(inputPivot.iloc[index][column])
                    #print(thisRow)

        thisRow.append(targetValue)
        trainingData.append(thisRow)

    # trainingData = [trainingRow for trainingRow in trainingData if trainingRow[-1] != -1]
    # print(len(trainingData))

    return True, trainingData, datetimes

# we firstly clean the inout data and then run the training
def cleanDataFrame(inputData, startDate, endDate, clean_thresh, missing, agg_period, agg_method):
    inputData['datetime'] = pd.to_datetime(inputData['datetime'])
    inputPivot = inputData.pivot_table(index='datetime', columns='camera', values='count')

    mask = (inputPivot.index > startDate) & (inputPivot.index <= endDate)
    inputPivot = inputPivot.loc[mask]

    inputPivot['weekday'] = ((pd.DatetimeIndex(inputPivot.index).dayofweek) // 5 == 1).astype(float)
    inputPivot['day'] = pd.DatetimeIndex(inputPivot.index).dayofweek
    if agg_method == 'mean':
        inputPivot = inputPivot.resample(rule=agg_period).mean()
    else:
        inputPivot = inputPivot.resample(rule=agg_period).max()

    inputPivot.fillna(missing, inplace=True)

    inputPivot['datetime'] = inputPivot.index
    inputPivot['datetime'] = inputPivot.datetime.dt.strftime('%Y-%m-%d %H:%M')
    inputPivot['serial'] = inputPivot.index.hour * 4 + inputPivot.index.minute.astype('int')/15
    # inputPivot['15MinSerial'] = inputPivot.index.hour * 4 + int(inputPivot['dt'] / 15)
    inputPivot['15MinMidnight'] = inputPivot['serial'].apply(lambda x: sqrt((0 - x)**2))
    inputPivot['15MinMidday'] = inputPivot['serial'].apply(lambda x: sqrt((48 - x)**2))

    inputPivot = inputPivot.drop(['serial'], axis=1)

    columns_to_remove1 = consecutive_values(inputPivot, 0, clean_thresh, xcpt=['weekday', 'day'])
    columns_to_remove2 = consecutive_values(inputPivot, -1, clean_thresh, xcpt=['weekday', 'day'])
    columns_to_remove = columns_to_remove1 + columns_to_remove2
    columns_to_remove = list(dict.fromkeys(columns_to_remove))
    return columns_to_remove

# preparing the cleaned data for training
def MulticamDataPrep(inputData, startDate, endDate, lag, missing, target, agg_period, agg_method, clean_thresh, lead=1):
    inputData['datetime'] = pd.to_datetime(inputData['datetime'])

    inputPivot = inputData.pivot_table(index='datetime', columns='camera', values='count')

    mask = (inputPivot.index > startDate) & (inputPivot.index <= endDate)
    inputPivot = inputPivot.loc[mask]

    inputPivot['weekday'] = ((pd.DatetimeIndex(inputPivot.index).dayofweek) // 5 == 1).astype(float)
    inputPivot['day'] = pd.DatetimeIndex(inputPivot.index).dayofweek
    if agg_method == 'mean':
        inputPivot = inputPivot.resample(rule=agg_period).mean()
    else:
        inputPivot = inputPivot.resample(rule=agg_period).max()


    inputPivot.fillna(missing, inplace=True)

    inputPivot['datetime'] = inputPivot.index
    inputPivot['datetime'] = inputPivot.datetime.dt.strftime('%Y-%m-%d %H:%M')
    inputPivot['serial'] = inputPivot.index.hour * 4 + inputPivot.index.minute.astype('int')/15
    # inputPivot['15MinSerial'] = inputPivot.index.hour * 4 + int(inputPivot['dt'] / 15)
    inputPivot['15MinMidnight'] = inputPivot['serial'].apply(lambda x: sqrt((0 - x)**2))
    inputPivot['15MinMidday'] = inputPivot['serial'].apply(lambda x: sqrt((48 - x)**2))

    inputPivot = inputPivot.drop(['serial'], axis=1)

    # print(inputPivot)

    trainingData = []
    datetimes = []

    columns_to_remove1 = consecutive_values(inputPivot, 0, clean_thresh, xcpt=['weekday', 'day'])
    columns_to_remove2 = consecutive_values(inputPivot, -1, clean_thresh, xcpt=['weekday', 'day'])
    columns_to_remove = columns_to_remove1 + columns_to_remove2
    columns_to_remove = list(dict.fromkeys(columns_to_remove))

    inputPivot = inputPivot.drop(columns_to_remove, axis=1)
    outputColumns = [column for column in inputPivot.columns if column != 'datetime']

    inputPivotLen = len(inputPivot)

    if target not in inputPivot.columns:
        return False, 0, 0, 0

    for i in range(0, inputPivotLen - (lag+lead-1)):
        thisRow = []
        for j in range(0, lag+1):
            index = i+j
            for column in inputPivot.columns:
                if j == lag:
                    if column == 'datetime':
                        datetimes.append(inputPivot.iloc[index+lead-1][column])
                    if column == target:
                        targetValue = inputPivot.iloc[index+lead-1][column]
                    else:
                        continue
                else:
                    if column == 'datetime':
                        continue
                    thisRow.append(inputPivot.iloc[index][column])

        thisRow.append(targetValue)
        trainingData.append(thisRow)

    trainingData = [trainingRow for trainingRow in trainingData if trainingRow[-1] != -1]
    # print(trainingData)
    return True, outputColumns, trainingData, datetimes

# calculating bearings between two sensors - OD matrix is used
def bearing_between_sensors(csv_path):
    df = pd.read_csv(csv_path)
    distance_matrix_dict = {}
    distance_matrix_list = []
    for index1, row1 in df.iterrows():
        s1 = row1['Sensor Name']
        p1lat = row1['Sensor Centroid Latitude']
        p1lon = row1['Sensor Centroid Longitude']
        for index2, row2 in df.iterrows():
            s2 = row2['Sensor Name']
            if s1 == s2:
                continue
            p2lat = row2['Sensor Centroid Latitude']
            p2lon = row2['Sensor Centroid Longitude']
            dist = get_bearing(p1lat, p2lat, p1lon, p2lon)
            if s1 in distance_matrix_dict:
                distance_matrix_dict[s1][s2] = dist
            else:
                distance_matrix_dict[s1] = {s2: dist}
            distance_matrix_list.append([s1, s2, dist])
    return distance_matrix_dict, distance_matrix_list

def get_bearing(lat1, lat2, long1, long2):
    brng = Geodesic.WGS84.Inverse(lat1, long1, lat2, long2)['azi1']
    if brng < 0:
        brng += 360
    return brng

# calculating distances between sensors across the network based on OD matrix
def network_distance_between_sensors(path_to_csv):
    df = pd.read_csv(path_to_csv, sep=',')
    df = df[['origin_id', 'destination_id', 'total_cost']]
    df['total_cost'] = df['total_cost'].fillna(999999)
    df['total_cost'] = df['total_cost'] / 1000
    df['total_cost'] = df['total_cost'].astype(float)
    distance_matrix_list = df.values.tolist()

    return {}, distance_matrix_list

# calculating distances between two sensors
def distances_between_sensors(csv_path):
    df = pd.read_csv(csv_path)
    distance_matrix_dict = {}
    distance_matrix_list = []
    for index1, row1 in df.iterrows():
        s1 = row1['Sensor Name']
        p1lat = row1['Sensor Centroid Latitude']
        p1lon = row1['Sensor Centroid Longitude']
        for index2, row2 in df.iterrows():
            s2 = row2['Sensor Name']
            if s1 == s2:
                continue
            p2lat = row2['Sensor Centroid Latitude']
            p2lon = row2['Sensor Centroid Longitude']
            dist = lat_lon_dist(p1lat, p1lon, p2lat, p2lon)
            if s1 in distance_matrix_dict:
                distance_matrix_dict[s1][s2] = dist
            else:
                distance_matrix_dict[s1] = {s2: dist}
            distance_matrix_list.append([s1, s2, dist])
    return distance_matrix_dict, distance_matrix_list
