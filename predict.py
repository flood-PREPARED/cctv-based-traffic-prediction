from dataPrep import MulticamDataPrepForPredict as mdp, mape_vectorized_v2 as mape_vec, rmse_vectorized as rmse_vec, mae_vectorized as mae_vec
# from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import pickle
import time
import os
# models[target_cam][nearbyLimit][lag][lead] = {'model': model, 'mape': mape}

# get file path
path1 = os.path.abspath(os.path.join(os.getcwd()))

# set list of camera's to run predictions for
camera_names = ['NC_B1307B1',]

# set static paths
input_data_path = os.path.join(path1, 'input_data_evaluation/data.csv')
scv_save_folder = os.path.join(path1, 'outputs')

# set static settings
target_dists = [0.20]
nearbyLimits = [4]
#lags = [4, 8, 12]
lags = [49]
agg_period ='30Min'
agg_method = 'mean'
model_names = ['RandomForestRegressor']
leads =[1]

max_lead = max(leads)

# run the predictions
for camera_name in camera_names:

    target_cam = 'PER_PEOPLE-FTP_' + camera_name
    model_path = os.path.join(path1, 'models', 'traffic_predictor-PER_PEOPLE-FTP_NC_B1307B1_lag49_20210725_150045.901155_tests2021.pickle')
    models = pickle.load(open(model_path, "rb"))

    for model_name in model_names:
        for nearbyLimit in nearbyLimits:
            for target_dist in target_dists:
                for lag in lags:
                    t1 = time.time()
                    columns = models[target_cam][nearbyLimit][target_dist]['columns']
                    #print(columns)
                    inputDf = pd.read_csv(input_data_path)
                    inputDf.columns = ['camera', 'variable', 'units', 'datetime', 'count', 'flag']

                    success, testing_data, datetimes = mdp(inputData=inputDf, lag=lag, missing=-1, target=target_cam, agg_period=agg_period, agg_method=agg_method, columns=columns, lead=max_lead )
                    testing_data = np.array([np.array(xi) for xi in testing_data])
                    #print(testing_data)
                    tests = list(testing_data[:, -1])

                    predictions = []
                    breaklines = []
                    j = 0
                    for m in range(0, len(testing_data), max_lead):
                        breaklines.append(m+max_lead)
                        testX = [testing_data[m, :-1]]
                        #print(m)

                        for lead in leads:
                            # print(models[target_cam])
                            model = models[target_cam][nearbyLimit][target_dist][lag][lead][model_name]['model']

                            # mape = models[target_cam][nearbyLimit][target_dist][lag][lead][model_name]['mape']
                            # rmse = models[target_cam][nearbyLimit][target_dist][lag][lead][model_name]['rmse']
                            prediction = model.predict(testX)
                            #print(testing_data[m], prediction)
                            predictions.extend(prediction)
                            #print(predictions)

                            j += 1

                    x = list(range(len(predictions)))

                    print(len(x), len(tests), len(predictions))

                    # replace the -1 in the list of true values
                    for n, i in enumerate(tests):
                        if i == -1:
                            tests[n] = 0
                        else:
                            continue

                    mape = mape_vec(tests, predictions)
                    rmse = rmse_vec(tests, predictions)
                    mae = mae_vec(tests, predictions)

                    print('Our metrics:   ', 'mape {}, rmse {}, mae {}'.format(mape, rmse, mae))
                    mae2 = mean_absolute_error(tests, predictions)
                    mse2 = mean_squared_error(tests, predictions)
                    print('sklearn metrics:   ', 'mae2 {}, mse2 {}'.format(mae2, mse2))

                    pred_over_tests = [1 for x, y in zip(tests, predictions) if x < y]

                    print('pred_over_tests: {} in tests: {}'.format(sum(pred_over_tests), len(tests)))

                    print('mean tests: {}, mean pred: {}'.format(np.mean(tests), np.mean(predictions)))
                    # adding the true values and the predicitons in a dataframe
                    dict_results = {'datetimes': datetimes, 'tests': tests, 'predictions': predictions}
                    df_results = pd.DataFrame(dict_results, index=datetimes)


                    #print(datetimes)
                    time_x = pd.date_range(datetimes[0], datetimes[-1], freq='30min').strftime('%H:%M')

                    #print(time_x)

                    ax = plt.gca()
                    df_results.plot(kind='line', x='datetimes', y='tests', color='blue', ax=ax)
                    df_results.plot(kind='line', x='datetimes', y='predictions', color='red', ax=ax)

                    df_results.to_csv(scv_save_folder + str(target_cam)+'predictions_4wholeweek.csv', index=False)
                    #plt.show()
