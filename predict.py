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


def main(camera_name):
    """
    Run the prediction code
    :return:
    """
    # get file path
    path1 = os.path.abspath(os.path.join(os.getcwd()))

    # check if model exists for named camera and if so get path to model
    # this is a dynamic function so will get the first model it finds with the name of the camera in
    # therefore if the model is updated with a different name, ensure the old version is deleted
    model_path = check_model_exists(camera_name)
    if model_path is False:
        return False

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
    target_cam = 'PER_PEOPLE-FTP_' + camera_name
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


def list_available_models():
    """
    List the available models
    :return:
    """
    # set path to models
    model_loc = os.path.abspath(os.path.join(os.getcwd(), 'models'))

    return [f for f in os.listdir(model_loc) if os.path.isfile(os.path.join(model_loc, f))]


def ident_camera_name(model_name):
    """
    Parse model name to get camera name
    :param
    :return:
    """

    model_name = model_name.split('_')
    return model_name[3]+'_'+model_name[4]


def check_model_exists(camera_name):
    """
    Check the model exists for camera
    :return:
    """
    model_loc = os.path.abspath(os.path.join(os.getcwd(), 'models'))
    models = list_available_models()

    for model in models:
        print('Looking for model for %s' %camera_name)
        if camera_name in model:
            print('Found model - %s' %model)
            return os.path.join(model_loc, model)

    return False


#def __main__():
#    """
#
#    :return:
#    """

# fetching env setting for 'camera'
camera_name = os.getenv('camera')

# if no env passed, set to run for all cameras
if camera_name is None:
    print('Camera name not passed so setting to run for all')
    camera_name = 'all'

if camera_name == 'all':
    models = list_available_models()
    for model in models:
        main(camera_name=ident_camera_name(model))
else:
    main(camera_name=camera_name)
