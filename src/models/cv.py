import pandas as pd 
import numpy as np

from src.metrics.compute_metrics import cal_mape, cal_smape

from sklearn.metrics import r2_score,root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import warnings
import time
import gc
from joblib import Parallel
from src.features.feature_processing import scale_features
from src.models.SARIMAX_run import SARIMAX_train,  SARIMAX_forecast



def rolling_cv_splits(data, initial_train_size, horizon, step=1, n=122):
    
    n = n
    start = initial_train_size
    test_end =  start + horizon 
    while start <= n:
        train = data.iloc[:start] if hasattr(data, 'iloc') else data[:start] 
        if test_end > n:
            test_end = n 
        test = data.iloc[start:test_end ] if hasattr(data, 'iloc') else data[start:test_end ]
        if len(test) <4:
            break
        yield train, test
        start += step
        test_end =  start + horizon  



def model_run(df_store, parm_list, featrs_to_scale, binary_featrs, initial_train_size = 104, horizon = 4, step = 4  ):

    warnings.filterwarnings("ignore")
    splits = list(rolling_cv_splits(df_store, initial_train_size, horizon, step ))
     
    smape_list =[]
    rmse_list =[]
    r2_list = []

    # Iterate over each set of parameters in the grid
    for train,test in splits:
         
         Srmx_model, forecast, x_scld_train,preprocessor, x_scld_test, y_scld_train, y_scld_test  = endog_exog_split(train, test,featrs_to_scale, binary_featrs, parm_list )

         y_predicted = forecast.predicted_mean

         smape_list.append(cal_smape(y_scld_test, y_predicted))
         rmse_list.append( root_mean_squared_error(y_scld_test, y_predicted))
         r2_list.append(r2_score(y_scld_test, y_predicted))

         p_list = [parm_list['D'],parm_list['P'],parm_list['Q'],parm_list['d'],parm_list['p'],parm_list['q']]
         #print(Srmx_res.summary())

         return {
            "params":p_list,
            "rmse": np.mean(rmse_list),
            "smape":np.mean( smape_list),
            "r2":np.mean( r2_list),
           
            
         }


def endog_exog_split(train, test, featrs_to_scale, binary_featrs, parm_list ):
         
         x_scld_train = pd.DataFrame()
         x_scld_test = pd.DataFrame()
             
         if featrs_to_scale is not None:
           train_scld,test_scld, preprocessor =  scale_features(train, test, featrs_to_scale)

         if "y" in featrs_to_scale:
            featrs_to_scale.remove("y")
            y_scld_train = train_scld.loc[:, "y"]
            y_scld_test = test_scld.loc[:, "y"]
         else:
            featrs_to_scale.remove("y_diff_1")
            y_scld_train = train_scld.loc[:, "y_diff_1"]
            y_scld_test = test_scld.loc[:, "y_diff_1"]

         if(len(featrs_to_scale) >1 ):
             x_scld_train = train_scld.loc[:,  featrs_to_scale]
             x_scld_test = test_scld.loc[:,  featrs_to_scale]
         
         if(binary_featrs is not None):
            x_scld_train[binary_featrs ] = train.loc[:,binary_featrs]
            x_scld_test[binary_featrs] = test.loc[:, binary_featrs] 

         Srmx_model = SARIMAX_train(y = y_scld_train , x = x_scld_train, parm_list = parm_list)

         forecast = SARIMAX_forecast(Srmx_model, y_scld_test, x_scld_test )

         return  Srmx_model, forecast, preprocessor, x_scld_train, x_scld_test, y_scld_train, y_scld_test
