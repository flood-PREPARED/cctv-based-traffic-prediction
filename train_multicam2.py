from dataPrep import MulticamDataPrep as mdp, distances_between_sensors as dbs, mape_vectorized_v2 as mape_vec, rmse_vectorized as rmse_vec, network_distance_between_sensors as ndbs, bearing_between_sensors as bbs, cleanDataFrame as cdf
from sklearn.ensemble import RandomForestRegressor
from math import sqrt
import pandas as pd
import numpy as np
import logging
import datetime
import time
import pickle
import copy
import sys
import os

import warnings
warnings.filterwarnings('ignore')

path1 = os.path.abspath(os.path.join(os.getcwd()))
print(path1)
logname = path1+'/log/tpt.log'
#CarCountsDataPath = path1+'/input4training/dataJan_Aug19.csv'
CarCountsDataPathPickle = path1+'/input4training/dataJan_Aug19.pickle'
SensorDataPath = path1+'/input4training/sensors2.csv'
DistanceMatrixCsvPath = path1+'/input4training/OD_matrix_updated.csv'
output_path = path1+'/trained_model4prediction'


logging.basicConfig(filename=logname, filemode='a', level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')

regression_models = dict(
    RandomForestRegressor=RandomForestRegressor(n_estimators=96, max_features='auto', max_depth=25,
                                                min_samples_split=10, min_samples_leaf=4, bootstrap=True))

# when selecting many nearby cameras for the selection to be right, set: power_dist = 4, power_bearing = 1, nearbyLimits = [4], target_dists = [0.2]
# when selecting ZERO nearby cameras for the selection to be right, set: power_dist = 4, power_bearing = 1, nearbyLimits = [0.01], target_dists = [1]
lags = [49]
leads = [1]

power_dist = 4
power_bearing = 1

prepend = 'PER_PEOPLE-FTP_'
trim_len = len(prepend)

###### Train a single camera
#target_cameras = ['PER_PEOPLE-FTP_NC_A1058C1']

##### Train many cameras at once
target_cameras = ['PER_PEOPLE-FTP_NC_A167E1', 'PER_PEOPLE-FTP_NC_B1307B1', 'PER_PEOPLE-FTP_NC_A695E1', 'PER_PEOPLE-FTP_NC_A167B1', 'PER_PEOPLE-FTP_NC_A1058C1', 'PER_PEOPLE-FTP_GH_A184D1']

nearbyLimits = [4] # number of selected nearby cameras
traintest = 0
target_dists = [0.2] # the distance closest to the nearby cameras we want to select
missing = -1
agg_period = '30Min'
agg_method = 'mean'
startDate = "2019/07/01 00:00:00"
endDate = "2019/08/02 00:00:00"

#clean_thresh = 192 # for two days to clean nan data when agg_period = '15Min'
clean_thresh = 96 # for two days to clean nan data when agg_period = '30Min'
#clean_thresh = 48 # for two days to clean nan data when agg_period = '60Min'

####################################################################################################################
# Use these lines if input is the csv
# print('[{}] Reading csv'.format(datetime.datetime.now()))
# inputDf = pd.read_csv(CarCountsDataPath)
# inputDf.columns = ['camera', 'variable', 'units', 'datetime', 'count', 'flag']
####################################################################################################################
# Use this line if input is pickle and comment out the previous section
print('[{}] Reading pickle'.format(datetime.datetime.now()))
inputDf = pd.read_pickle(CarCountsDataPathPickle)

print('org columns', inputDf.columns)
ctr = cdf(inputDf, startDate, endDate, clean_thresh, missing, agg_period, agg_method) # ColumnsToRemove

# columns to remove using the threshold
print('ctr', ctr)
inputDf = inputDf[~inputDf['camera'].isin(ctr)]

# print('[{}] Data loaded'.format(datetime.datetime.now()))
cameras = inputDf['camera'].unique()
# print(cameras)
models = {camera: {nearbyLimit: {target_dist: {lag: {lead: {regression_model_name: {} for regression_model_name, _ in regression_models.items()} for lead in leads} for lag in lags} for target_dist in target_dists} for nearbyLimit in nearbyLimits} for camera in cameras}
models["agg_period"] = agg_period
models["agg_method"] = agg_method

found = 0
for enum, target_cam in enumerate(target_cameras):
    #if found > 0:
    #    break
    #if target_cam != target_cameras:
    #    continue
    #found = 1
    #if enum == 1:
    #     break
    print('----------------------')
    print(target_cam)
    for nearbyLimit in nearbyLimits:
        _, distance_matrix_list = ndbs(DistanceMatrixCsvPath)
        _, bearing_between_sensors = bbs(SensorDataPath)

        # to fix the preoblem, with the name we have to ensure that both distance and bearing matrices do not include the cameras that have been filtered out
        distance_matrix_list = [[d[0], d[1], d[2]] for d in distance_matrix_list if d[0] not in ctr and d[1] not in ctr]
        distance_matrix_list = [[d[0], d[1], d[2]] for d in distance_matrix_list if prepend+d[0] not in ctr and prepend+d[1] not in ctr]
        bearing_between_sensors = [[d[0], d[1], d[2]] for d in bearing_between_sensors if d[0] not in ctr and d[1] not in ctr]
        bearing_between_sensors = [[d[0], d[1], d[2]] for d in bearing_between_sensors if prepend+d[0] not in ctr and prepend+d[1] not in ctr]

        for target_dist in target_dists:

            distance_matrix_list1 = [[d[0], d[1], sqrt((d[2] - target_dist) ** 2), d[2]] for d in distance_matrix_list if prepend + d[1] != target_cam]

            distance_matrix_list = sorted(distance_matrix_list1, key=lambda x: x[2])
            distance_matrix_list = [[d[0], d[1], d[2]] for d in distance_matrix_list if prepend + d[0] == target_cam]

            # we pick the camera thats closest to desired distance
            firstNearbyCamera = ['PER_PEOPLE-FTP_'+d[1] for d in distance_matrix_list][0]  #fnc
            nearbyCameras = [firstNearbyCamera]

            # we find the picked cameras bearing
            for origin, destination, bearing in bearing_between_sensors:
                if origin == target_cam and destination == firstNearbyCamera:
                    fncBearing = bearing
            # move distance and bearing ranges from their original to common 0-1
            minDist, maxDist = min([d[2] for d in distance_matrix_list1]), max([d[2] for d in distance_matrix_list1])
            normalizedDistance_matrix_list1 = [[prepend + d[0], prepend + d[1], (d[2]-minDist)/maxDist, d[3]] for d in distance_matrix_list1]

            # we define bearing step to use when searching for cameras in different directions
            bearingStep = int(360 / nearbyLimit)
            # loop through bearing steps, add them to the bearing of first picked camera
            # and find camera that's closest to desired distance and best aligned with desired bearing

            for enum2, thisBearingDiff in enumerate(range(bearingStep, 360, bearingStep)):

                thisBearing = fncBearing + thisBearingDiff
                if thisBearing >= 360:
                    thisBearing -= 360  # if it goes over 360 subtract 360
                print('thisBearing {}, fncBearing {}, thisBearingDiff {}'.format(thisBearing, fncBearing, thisBearingDiff))

                # if enum2 == 0:
                #     continue
                bearing_to_our_camera = [[b[0], b[1], b[2]] for b in bearing_between_sensors if b[0] == target_cam]
                sensors_closeness_to_thisBearing = [[b[0], b[1], sqrt((b[2] - thisBearing) ** 2), b[2]] for b in bearing_to_our_camera]
                sensors_closeness_to_thisBearing = sorted(sensors_closeness_to_thisBearing, key=lambda x: x[2])
                normalizedBearings_between_sensors = [[b[0], b[1], b[2] / 360, b[2], b[3]] for b in sensors_closeness_to_thisBearing]

                # join two lists
                normalizedDistanceAndBearing_between_sensors = []
                for org1, dest1, dist, orgDist in normalizedDistance_matrix_list1:
                    if dest1 in nearbyCameras:
                        continue
                    for org2, dest2, bear, bearDiff, orgBear in normalizedBearings_between_sensors:
                        if org1 == org2 and dest1 == dest2:
                            normalizedDistanceAndBearing_between_sensors.append([org1, dest1, dist, bear, orgDist, bearDiff, orgBear])

                # multiplying normalized values (distance n[2], bearing n[3]) by 100 to assure that multiplication of two values or taking to power of N results in higher numbers
                normalizedDistanceAndBearing_between_sensors = [[n[0], n[1], n[2], n[3], (n[2])**power_dist * (n[3])**power_bearing, n[4], n[5], n[6]] for n in normalizedDistanceAndBearing_between_sensors] ########### HERE: the higher the Weight for distance the higher the chance to pick up a closest camera along the specified direction
                sorted_normalizedDistanceAndBearing_between_sensors = sorted(normalizedDistanceAndBearing_between_sensors, key=lambda x: x[4])
                # print(sorted_normalizedDistanceAndBearing_between_sensors)
                chosenCamera = sorted_normalizedDistanceAndBearing_between_sensors[0][1]
                print('cam, dest, dist, bear, score, real_dist, bear_diff, real_bear', sorted_normalizedDistanceAndBearing_between_sensors[0])
                nearbyCameras.append(chosenCamera)
                # print('chosen camera: {}'.format(chosenCamera))

            if 1 > nearbyLimit >= 0:
                nearbyCameras = [target_cam]
                print('nearbyCameras when nearbyLimits is between 0-1', nearbyCameras)
            else:
                nearbyCameras = nearbyCameras + [target_cam]
                print('nearbyCameras when nearbyLimits is > 1', nearbyCameras)

            ThisInputDf = inputDf[inputDf['camera'].isin(nearbyCameras)]
            print('input df cameras', ThisInputDf['camera'].unique())

            for lag in lags:
                for regression_model_name, regression_model in regression_models.items():
                    for lead in leads:
                        model = copy.deepcopy(regression_model)
                        try:
                        # if True:
                            t1 = time.time()
                            # if True:
                            success, columns, samples, datetimes = mdp(inputData=ThisInputDf, startDate=startDate, endDate=endDate, lag=lag, lead=lead, missing=-1,
                                                                       target=target_cam, agg_period=agg_period, agg_method=agg_method, clean_thresh=clean_thresh)

                            if not success:
                                continue

                            # print(samples[0])

                            array = np.array([np.array(xi) for xi in samples])
                            train_split = int(traintest * array.shape[0])

                            # train = array[:train_split, :]
                            train = array[:, :]
                            test = array[train_split:, :]
                            # trainDt = datetimes[:train_split]
                            trainDt = datetimes[:]
                            testDt = datetimes[train_split:]

                            trainX = train[:, :-1]
                            trainY = train[:, -1]
                            testX = test[:, :-1]
                            testY = test[:, -1]

                            model.fit(trainX, trainY)

                            mape = 0
                            rmse = 0

                            arrayX = array[:, :-1]
                            arrayY = array[:, -1]
                            model.fit(arrayX, arrayY)
                            models[target_cam][nearbyLimit][target_dist]['columns'] = columns
                            print(columns)
                            models[target_cam][nearbyLimit][target_dist][lag][lead][regression_model_name] = {'model': model, 'mape': mape, "rmse": rmse}
                            delta_time = time.time() - t1
                            sql_selection_string = '"Name" IN (' + "'" + ("', '").join([target_cam[trim_len:] + ' - ' + i[trim_len:] for i in columns if prepend in i and target_cam != i]) + "')"
                            print(sql_selection_string)
                            msg = 'Success camera {}, nearby {}, target dist {}, lag {}, lead {}, model {}; mape: {}, rmse: {}, time: {}'.format(target_cam, nearbyLimit, target_dist, lag, lead, regression_model_name, mape, rmse, delta_time)
                            print(msg)
                            logging.info(msg)
                        except Exception as e:
                            msg = '[ERROR] {} for camera {}, nearby {}, target dist {}, lag {}, lead {}: {}'.format(sys.exc_info()[0], target_cam, nearbyLimit, target_dist, lag, lead, e)
                            print(msg)
                            logging.info(msg)
                            pass

    save_path = os.path.join(output_path, "traffic_predictor-" + str(target_cam) + '_lag49' + '_' + str(datetime.datetime.now()).replace(':','').replace(' ','_').replace('-','')+'_tests2021.pickle')

    pickle.dump(models, open(save_path, "wb"))
