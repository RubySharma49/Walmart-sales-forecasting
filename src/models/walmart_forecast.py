### import all necessary files

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import month_plot, quarter_plot, plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from sklearn.metrics import root_mean_squared_error,r2_score, mean_squared_error,mean_absolute_error, mean_absolute_percentage_error
from pmdarima import auto_arima, ARIMA, model_selection
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import skew,kurtosis,shapiro
import statsmodels.api as sm
import scipy.stats as st
from scipy.stats import kruskal
from sklearn.model_selection import ParameterGrid
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pmdarima.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from scipy.stats import f_oneway
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller

## all necessary functions 

## All necessary function defitions

def plot_scatter(x, y, xlabel, ylabel, title, col, figsize):
    plt.figure(figsize = figsize)
    sns.boxplot(data=df, x=x, y=y, color=col)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=90) 
    plt.show()
    
    
    
## Function to introduce holiday categories

def holiday_df(df):
    
    gen_hol = pd.DataFrame({ "holiday":"gen_hol",
                          'ds': df[df["Holiday_Flag"]==1].index,
                          'lower_window':-2,
                          'upper_window':2})

    thxgiv = pd.DataFrame({'holiday': 'thanksgiving',
                     'ds': pd.to_datetime(['2010-11-25','2011-11-24', '2012-11-22']),
                             'lower_window': -7,
                             'upper_window': 7})

    xmas = pd.DataFrame({'holiday': 'christmas',
                     'ds': pd.to_datetime(['2010-12-25','2011-12-25', '2012-12-25']),
                             'lower_window': -7,
                             'upper_window': 7})
     # New Year's eve
    nye = pd.DataFrame({'holiday': 'new_years',
                     'ds': pd.to_datetime(['2010-12-31', '2011-12-31', '2012-12-31']),
                             'lower_window': -3,
                             'upper_window': 3})
     # Easter
    easter = pd.DataFrame({'holiday': 'easter',
                     'ds': pd.to_datetime(['2010-04-04', '2011-04-24', '2012-04-08']),
                     'lower_window': -3,
                             'upper_window': 3})
    # Concatenate holiday                      
    holiday = pd.concat([gen_hol, easter, thxgiv, xmas, nye] )
    #holiday.reset_index(inplace=True)
    return(holiday)

def cal_mape(actual, pred):
    return np.mean(np.abs((actual - pred) / actual)) * 100

def cal_smape(actual, pred):
    return np.mean(2 * np.abs(pred - actual) / (np.abs(actual) + np.abs(pred))) * 100

## function for forcasting the exogenous variables

def forcast_exogenous(df, var_list):
    
    exo_forecast = pd.DataFrame()
    for v in var_list:

        target_data = df.loc[:,["ds",v]]
        target_data = target_data.rename(columns={v:"y"})
        
        param_grid = {
            'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
            'seasonality_prior_scale': [0.1, 1.0, 10.0, 20.0],
            'holidays_prior_scale':[0.1, 1.0, 10.0, 20.0],
            'seasonality_mode':['additive', 'multiplicative']
        }
        params = list(ParameterGrid(param_grid))
        rmse_results=[]
        mape_results=[]
        
        for pr in params:
            p = Prophet(yearly_seasonality=True,
                        weekly_seasonality=False, 
                        daily_seasonality=False,
                        holidays = holiday,
                        **pr
                       )
            p.fit(target_data)
            df_cv = cross_validation(model=p,
                 initial= '60 W',
                 period = '4 W' ,
                 horizon= '6 W')
            rmse = performance_metrics(df_cv)['rmse'].mean()
            mape = performance_metrics(df_cv)['mape'].mean()
            rmse_results.append(rmse)
            mape_results.append(mape)
        result = pd.DataFrame(params)
        result["rmse"] = rmse_results
        result["mape"] = mape_results
    
        best_score = result.loc[result["rmse"] == result["rmse"].min(),:]
        exo_best_pr = best_score.drop(columns = ["rmse", "mape"]) 
        exo_best_pr = exo_best_pr.iloc[0].to_dict()
    
        p = Prophet(
             yearly_seasonality=True,
             weekly_seasonality=False, 
             daily_seasonality=False,
             holidays = holiday,
             **exo_best_pr
        )
        p.fit(target_data)

        future = p.make_future_dataframe(periods=365)
        forcast = p.predict(future)
        exo_forecast[v] = forcast["yhat"]
        
    exo_forecast["ds"] = forcast["ds"]  
    return exo_forecast



## Function to scale the features using Min_Max scaler 
def Min_max_scling(data, num_featrs):
    
    from sklearn.preprocessing import MinMaxScaler
    
    preprocessor =  MinMaxScaler(feature_range=(0, 1))
    
    df_scld = preprocessor.fit_transform(data[num_featrs]) 
    
    df_scld = pd.DataFrame(
          df_scld,
          columns = num_featrs,
          index = data.index
    )

    
    #df_scld.plot.box(subplots = True, figsize = (12,3))
    #plt.tight_layout()
    #plt.show()
    
    return(df_scld)

## Function to plot prediction
## Function to plot prediction
def plot_pred(train,test, forecast, conf_int, rmse, smape, param, plotpdf ):
    
    
    
    plt.figure(figsize = (8,4))
    plt.plot(train, color = "gold", label = "Train")
    plt.plot(test, color = "blue", label = "Test")
    if forecast is not None:
        plt.plot(forecast, color = "red", label = "Forecast")
        
    if conf_int is not None:
        conf_values = conf_int.index.to_pydatetime()
        lower = conf_int.iloc[:, 0].values.astype(float)
        upper = conf_int.iloc[:, 1].values.astype(float)
        plt.fill_between(conf_values, lower, upper, alpha=0.3)
        
    if rmse is not None:
        mid_point = test.index[int(len(test)/2)]
        plt.text(mid_point, max(test.max(), forecast.max()), f'$RMSE$ = {rmse:.2f}\nSMAPE ={smape:.1f}%',fontsize=8)
    
    plt.title(f"SARIMAX Forecast with 95%CI using {param}")
    plt.ylabel("Weekly Sales")
    plt.xlabel("Date")
    plt.xticks(rotation = 90)
    plt.legend()  
    plt.tight_layout()
    if plotpdf:
        pdf.savefig()    # save   # saves the current figure
        plt.clf()        # clear the figure for next plot
    else:
        plt.show()

# Function to create the data segments for train and 
# test based on rolling cv

"""
    Generate rolling CV train/test splits.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame or array-like
        Time-ordered data.
    initial_train_size : int
        Number of observations to use in the first training split.
    horizon : int
        Number of observations in each test split.
    step : int, default=1
        How many observations to advance the window at each split.

    Yields
    ------
    train, test : tuple
        Slices of `data` for training and testing.
    """

def rolling_cv_splits(data, initial_train_size, horizon, step=1):
    
    n = 122#len(data)
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
       

def month_start_flag(dates):
    """
    Given a list of dates (strings or datetime), return a list where:
    - 1 = first occurrence of each month in the sequence
    - 0 = all other dates
    """
    # Convert to pandas datetime
    dates = pd.to_datetime(dates)
    
    # Create a DataFrame
    df = pd.DataFrame({'date': dates})
    
    # Sort by date (if not already sorted)
    df = df.sort_values('date').reset_index(drop=True)
    
    # Group by year and month, mark first row in each group as 1
    df['month_flag'] = df.groupby([df['date'].dt.year, df['date'].dt.month]).cumcount().eq(0).astype(int)
    
    return df['month_flag'].tolist()




print(" ----------------- Reading File in Python ---------------")
file_name = "data_ingest/walmart-sales-dataset-of-45stores.csv" 
df = pd.read_csv(file_name)
df['y'] = df['Weekly_Sales']
df = df.drop(columns=["Weekly_Sales"])
df['Date'] = pd.to_datetime(df['Date'], format = "%d-%m-%Y")

## Convert Date to datetime index, essential for time-series data
df['Date'] = pd.DatetimeIndex(df['Date'])
df.set_index('Date', inplace = True)


####  Missing data check
print("---------------- Checking is any null values in the data ------------")
print(df.isnull().sum())

#### Extract year and month from date
year = df.index.year
month = df.index.month
unique_years = year.unique()


print("Stastistical confirmation of sales are different acorss stores" )
groups = [grp['y'].values for _, grp in df.groupby('Store')]
f_stat, p_val = f_oneway(*groups)
print(f"F‐statistic: {f_stat:.2f}, p‐value: {p_val:.3f}")


'''
print("-------- Model building for store has started -----------")
print("---------------------------------------------------------")
from sklearn.metrics import r2_score, mean_squared_error,root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import warnings
import time

# Globally ignore all warnings
warnings.filterwarnings("ignore")

start = time.perf_counter()
stores = [1]#df["Store"].unique().astype("int")



for s in stores:

    stores_tunning_rest = pd.DataFrame(columns=range(8) )
    df_store = df.loc[df["Store"]==s,]
    print("\n-------Store: ",s, "----------")

    num_featrs = ['Temperature', 'Fuel_Price', 'CPI',
              'Unemployment', 'y']
   
    
    preprocessor =  MinMaxScaler(feature_range=(0, 1))    

    ### Selcting p,q,P,Q parameters based on ACF and PACF significant lag values 
    #series = df_store['y'].diff(1).dropna()

    ## acf values for non-seasonal parameters
    #acf_vals, acf_confint = acf(series, nlags=20, alpha=0.05)

    ## pacf values for non-seasonal parameters
    #pacf_vals, pacf_confint = pacf(series, nlags=20, alpha=0.05, method='ywm')


    #N = len(series)
    #bound = 1.96 / np.sqrt(N)

    #signif_nons_acf = np.where(np.abs(acf_vals[1:]) > bound)[0] + 1
    #signif_nons_pacf = np.where(np.abs(pacf_vals[1:]) > bound)[0] + 1

    #sea = list(signif_nons_acf[:2])                
    #q_vals = [0] + sea 

    #sea = list(signif_nons_pacf[:2])                 
    #p_vals = [0] + sea 

    #ses = 52
 

    #y_sdiff = df_store['y'].diff(ses).dropna()
    #plt.plot(y_sdiff)


    # Compute up to 3 seasonal cycles
    #nlags = 20

    ## acf and pacf values for non-seasonal parameters
    #acf_vals = acf(y_sdiff, nlags=nlags, fft=True)
    #pacf_vals = pacf(y_sdiff, nlags=nlags, method='ywm')

    #bound = 1.96 / np.sqrt(len(y_sdiff))

    # Look at seasonal lags
    #seasonal_lags = 20

    #sign_sea_acf = np.where(np.abs(acf_vals[1:]) > bound)[0] + 1
    #sign_sea_pacf = np.where(np.abs(pacf_vals[1:]) > bound)[0] + 1


    #sea = list(sign_sea_pacf[:2])     
    #P_vals = [0] + sea  

    #sea = list(sign_sea_acf[:2] )               
    #Q_vals = [0] + sea  

   
    param = {'p': [0,1],
         'd': [0,1],
         'q': [0,1],
         'P': [0,1,2],
         'D': [0,1],
         'Q': [0,1,2]}
    
    grid = ParameterGrid(param)
    tuning_results = pd.DataFrame(grid)
    r2 = []
    rmse = []

    splits = list(rolling_cv_splits(df_store, 90, 20, step = 20))

    # Iterate over each set of parameters in the grid
    with PdfPages('Results/evalutaion.pdf') as pdf:
         for parm_list in grid:
             r2_list = []
             rmse_list = []
             mse_list = []
        
           
             for train,test in splits:
            
               preprocessor.fit(train.iloc[:, 2:7])
    
               train_scld = preprocessor.transform(train.iloc[:, 2:7])
               test_scld = preprocessor.transform(test.iloc[:, 2:7])
               
               train_scld = pd.DataFrame(
                          train_scld,
                          columns = num_featrs,
                          index = train.index
               )
               test_scld = pd.DataFrame(
                         test_scld,
                         columns = num_featrs,
                         index = test.index
               )
               x_scld_train = train_scld.iloc[:, 0:4]
               y_scld_train = train_scld.iloc[:, 4]

               x_scld_test = test_scld.iloc[:, 0:4]
               y_scld_test = test_scld.iloc[:, 4]
    
               x_scld_train["Holiday_Flag"] = train.iloc[:, 1]
               x_scld_test["Holiday_Flag"] = test.iloc[:, 1]
    
               Srmx_model = SARIMAX(endog = y_scld_train , exog = x_scld_train ,
                          order = (parm_list['p'],parm_list['d'],parm_list['q']),
                          seasonal_order =(parm_list['P'],parm_list['D'],parm_list['Q'],52) )
 
               Srmx_res = Srmx_model.fit(disp = False)

               forecast = Srmx_res.get_prediction(start = y_scld_test.index[0],
                               end = y_scld_test.index[-1],
                               exog = x_scld_test  )

               y_predicted = forecast.predicted_mean

               param_str = ", ".join(f"{k}={v}" for k, v in parm_list.items())
               

               r2_scr = r2_score(y_scld_test, y_predicted)
               rmse_scr = root_mean_squared_error(y_scld_test, y_predicted)
               mse_scr = mean_squared_error(y_scld_test,y_predicted)
               
              
               r2_list.append(r2_scr) 
               rmse_list.append(rmse_scr)
               mse_list.append(mse_scr)
        
               conf_int = forecast.conf_int()
            
            
               
             
               #plot_pred(train = y_scld_train, test = y_scld_test, forecast=y_predicted,conf_int=conf_int, 
               #   r2=r2_scr, param = param_str, plotpdf=True)
        
             r2.append(np.average(r2_list))
             rmse.append( np.average(rmse_list))
       
        
# Add the RMSE values calculated during parameter tuning to the DataFrame
    tuning_results['rmse'] = rmse
    tuning_results['r2'] = r2  
    arima_best_params = tuning_results[tuning_results['rmse'] == tuning_results['rmse'].min()]
    #arima_best_params = pd.DataFrame(arima_best_params)
   
    stores_tunning_rest.loc[s-1] = arima_best_params.iloc[0].values
    stores_tunning_rest.columns = [ "D", "P", "Q", "d", "p", "q","rmse", "r2"]
    print(stores_tunning_rest)
    stores_tunning_rest["Stores"] = stores

    stores_tunning_rest.to_csv('Results/stores_best_params.txt',mode='a', index=False)

    end = time.perf_counter()
    print(f"Elapsed: {end - start:.6f} seconds")
'''

















print("-------- Model building for store has started -----------")
print("---------------------------------------------------------")
from sklearn.metrics import r2_score, mean_squared_error,root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import warnings
import time
import gc
from joblib import Parallel, delayed


def model_run(parm_list, features):
    warnings.filterwarnings("ignore")
    splits = list(rolling_cv_splits(df_store, 104, 4, step = 4))

 

    smape_list =[]
    rmse_list =[]
    r2_list = []
    coef=[]
    # Iterate over each set of parameters in the grid
    for train,test in splits:
         
     
         #scalest = time.perf_counter()
         preprocessor.fit(train.loc[:, features])
         train_scld = preprocessor.transform(train.loc[:, features])
         test_scld = preprocessor.transform(test.loc[:, features])
               
         train_scld = pd.DataFrame(
                          train_scld,
                          columns = features,
                          index = train.index
               )
         test_scld = pd.DataFrame(
                         test_scld,
                         columns = features,
                         index = test.index
               )
         train_scld = train_scld.dropna()
         test_scld = test_scld.dropna()

         if "y" in features:
            features.remove("y")
            y_scld_train = train_scld.loc[:, "y"]
            y_scld_test = test_scld.loc[:, "y"]
         else:
            features.remove("y_diff_1")
            y_scld_train = train_scld.loc[:, "y_diff_1"]
            y_scld_test = test_scld.loc[:, "y_diff_1"]

         x_scld_train = train_scld.loc[:, features]
         x_scld_test = test_scld.loc[:, features]
         
    
         x_scld_train[["Holiday_Flag", 'Month_Start_Flag',  'cal_thanksgiving_w0', 
          'cal_christmas_w-1', 'cal_christmas_w0'] ] = train.loc[:,[ "Holiday_Flag", 'Month_Start_Flag',
                                                               'cal_thanksgiving_w0', 
                                                              'cal_christmas_w-1', 'cal_christmas_w0']]

         x_scld_test[["Holiday_Flag", 'Month_Start_Flag', 'cal_thanksgiving_w0',
          'cal_christmas_w-1', 'cal_christmas_w0']] = test.loc[:, [ "Holiday_Flag", 'Month_Start_Flag', 
                                                              'cal_thanksgiving_w0', 
                                                              'cal_christmas_w-1', 'cal_christmas_w0']]
         #scaleend = time.perf_counter() 
         #print(f"scaled Elapsed: {scaleend - scalest:.6f} seconds")

   

             #modelst = time.perf_counter()
         Srmx_model = SARIMAX(endog = y_scld_train , exog = x_scld_train[["Holiday_Flag", 'Month_Start_Flag', 
                                                                             'cal_thanksgiving_w0', 
                                                                             'cal_christmas_w-1', 'cal_christmas_w0']],
                          order = (parm_list['p'],parm_list['d'],parm_list['q']),
                          seasonal_order =(parm_list['P'],parm_list['D'],parm_list['Q'],52) )
 
         Srmx_res = Srmx_model.fit(disp = False)

         forecast = Srmx_res.get_prediction(start = y_scld_test.index[0],
                               end = y_scld_test.index[-1],
                               exog = x_scld_test[["Holiday_Flag", 'Month_Start_Flag',
                                                                             'cal_thanksgiving_w0', 
                                                                             'cal_christmas_w-1', 'cal_christmas_w0'] ] )

         y_predicted = forecast.predicted_mean
             #modelend = time.perf_counter()
             #print(f"Model Elapsed: {modelend - modelst:.6f} seconds")

             #param_str = ", ".join(f"{k}={v}" for k, v in parm_list.items())
               
             #scorest = time.perf_counter()
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

# Globally ignore all warnings
warnings.filterwarnings("ignore")

start = time.perf_counter()
stores = range(1,46)#df["Store"].unique().astype("int") #14, 30,36, 38, 42, 43, 44
print(stores)

for s in stores:

    stores_tunning_rest = pd.DataFrame(columns=range(9) )
    stores_tunning_rest_r2 = pd.DataFrame(columns=range(9))
    df_store = df.loc[df["Store"]==s,]
    print("\n-------Store: ",s, "----------")

    num_featrs = ['Temperature', 'Fuel_Price', 'CPI','Unemployment', 'y']

    ### 1) Handling collinearity of exogenous variables 
    rm_feature = collinearity_test( df_store)
    #print("rm feature is ")
    #print(rm_feature)
    #print(num_featrs)

    if rm_feature!="":
        num_featrs.remove(rm_feature )
        df_store.drop(rm_feature, axis = 1, inplace = True)

 
    ### 2) Check for stationary and keep final differenced variables
    for col in num_featrs:
      
        pval = check_stationary(df_store[col], col)
        diff_no = 1
        name = col
        while pval > 0.05 and diff_no<=2:
            df_store[f'{col}_diff_{diff_no}'] = df_store[name].diff()
            pval = check_stationary(df_store[f'{col}_diff_{diff_no}'], name)
            df_store.drop(name, axis = 1, inplace=True)
            name = f'{col}_diff_{diff_no}'
            diff_no+=1

    ### 3) Add good friday flag
    df_store.loc[df_store.index.month.isin([11, 12]), "Holiday_Flag"] = 0


    df_store = add_thanksgiving_christmas_flags(
               df_store,
               week_anchor="FRI",
               windows={"thanksgiving":[0], "christmas":[-1,0]},
              col_prefix="cal_"
    )

    Month_Start_Flag = month_start_flag(df_store.index)
    df_store['Month_Start_Flag'] = Month_Start_Flag


    all_cols = df_store.columns.values.tolist()
    for e in ["Store", "Holiday_Flag", 'Month_Start_Flag',
          'cal_thanksgiving_w0',
          'cal_christmas_w-1', 'cal_christmas_w0']:
          all_cols.remove(e)
     
    preprocessor =  MinMaxScaler(feature_range=(0, 1))    


    param = {'p': [0,1,2],
         'd': [0],
         'q': [0,1,2],
         'P': [0,1,2],
         'D': [1],
         'Q': [0,1,2]}
    
    grid = ParameterGrid(param)
    
    results = Parallel(n_jobs=-2, verbose=10)(
         delayed(model_run)(params,all_cols ) for params in grid
    )      

    best = min(results, key=lambda x: x["rmse"])

    r2_best = max(results, key=lambda x: x["r2"])
   

    best_df = pd.DataFrame([best['params'] + [best['rmse'], best['smape'], best['r2']]], columns=[ "D", "P", "Q", "d", "p", "q","rmse", "smape", "r2"])

    best_df_r2 = pd.DataFrame([r2_best['params'] + [r2_best['rmse'], r2_best['smape'], r2_best['r2']]], columns=[ "D", "P", "Q", "d", "p", "q","rmse", "smape", "r2"])

    #stores_tunning = pd.DataFrame( best, columns=[ "D", "P", "Q", "d", "p", "q","rmse", "r2"])
    
    stores_tunning_rest.loc[s-1] = best_df.iloc[0].values
    stores_tunning_rest.columns = [ "D", "P", "Q", "d", "p", "q","rmse", "smape", "r2"]
    stores_tunning_rest["Stores"] = s
    stores_tunning_rest.to_csv('Results/stores_best_params_rmse.txt',mode='a', index=False, header=False)

    stores_tunning_rest_r2.loc[s-1] =  best_df_r2.iloc[0].values
    stores_tunning_rest_r2.columns = [ "D", "P", "Q", "d", "p", "q","rmse", "smape", "r2"]
    stores_tunning_rest_r2["Stores"] = s
    stores_tunning_rest_r2.to_csv('Results/stores_best_params_r2.txt',mode='a', index=False, header=False)

    end = time.perf_counter()
    print(f"total Elapsed: {end - start:.6f} seconds")



'''
def exo_model_run( parm_list,featre, use_sarimax=False, use_holts=False):

   
    warnings.filterwarnings("ignore")
    splits = list(rolling_cv_splits(df_store, 104, 15, step = 15))

    r2_list =[]
    rmse_list =[]
    # Iterate over each set of parameters in the grid
    for train,test in splits:
         
         train = train.iloc[:,featre]
         test = test.iloc[:,featre]
        
         if use_sarimax is True:
             
        
             Srm_model = SARIMAX(endog = train , trend = "t",
                          order = (parm_list['p'],parm_list['d'],parm_list['q']),
                          seasonal_order =(parm_list['P'],parm_list['D'],parm_list['Q'],52) )
             Srm_res = Srm_model.fit(disp = False )

             forecast = Srm_res.get_prediction(start = test.index[0],
                               end = test.index[-1] )

             y_predicted = forecast.predicted_mean
      
         if use_holts is True:
             Exp_season_model = ExponentialSmoothing(endog=train,
                                        trend="add",
                                        seasonal=None,
                                        seasonal_periods=52,
                                        initialization_method="estimated"
                                        ).fit()
             y_predicted = Exp_season_model.forecast(len(test))
            
            
            
         r2_list.append(r2_score(test, y_predicted))
         rmse_list.append( root_mean_squared_error(test, y_predicted))


    if use_sarimax == True:
        p_list = [parm_list['D'],parm_list['P'],parm_list['Q'],parm_list['d'],parm_list['p'],parm_list['q']]
    if use_holts == True:
        p_list = None

    return {
            "params":p_list,
            "rmse": np.mean(rmse_list),
            "r2":np.mean(r2_list),
            "med_r2":np.median(r2_list)
    }

# Globally ignore all warnings
warnings.filterwarnings("ignore")

gc.collect()

stores = range(2, 46) #df["Store"].unique().astype("int") # temp :7 , 43not working


for s in stores:
    
    start = time.perf_counter()
    results = []
    feature_exog =  pd.DataFrame()
    stores_tunning_rest = pd.DataFrame(columns=range(8) )
   
    print("\n-------Store: ",s, "----------")
    df_store = df.loc[df["Store"]==s,]
    #num_featrs = ['Temperature', 'Fuel_Price', 'CPI',
    #          'Unemployment']
    
    fetr_col = df_store.columns.get_loc("Fuel_Price")
   

    param = {'p': [1,2],
         'd': [0,1],
         'q': [0,1],
         'P': [1,2],
         'D': [0, 1],
         'Q': [0,1]}
    
    #param = {'p': [0],
    #     'd': [1],
    #     'q': [2],
    #     'P': [1],
    #     'D': [1],
    #     'Q': [1]}
    
    grid = ParameterGrid(param)

    results = Parallel(n_jobs=-3, verbose=10)(
         delayed(exo_model_run)( parm_list = params, featre = fetr_col, use_sarimax=True, use_holts=False ) for params in grid
    )      
    
    best = min(results, key=lambda x: x["rmse"])

    print(best)

    best_df = pd.DataFrame([best['params'] + [best['rmse'], best['r2']]], columns=[ "D", "P", "Q", "d", "p", "q","rmse", "r2"])
  
    tst_period = 30

    train_sg = df_store.iloc[:-tst_period, fetr_col]
    test_sg = df_store.iloc[-tst_period:, fetr_col]     


    Srm_model = SARIMAX(endog = train_sg,trend = "t",
                          order = ( best_df.loc[0,'p'], best_df.loc[0,'d'], best_df.loc[0,'q']),
                          seasonal_order =( best_df.loc[0,'P'], best_df.loc[0,'D'], best_df.loc[0,'Q'],52) )
   
 
    Srm_res = Srm_model.fit( disp = False )

    forecast = Srm_res.get_prediction(start = test_sg.index[0],
                               end = test_sg.index[-1] )
    
    future_exog = Srm_res.get_prediction(start = test_sg.index[-1],
                               end = '2013-12-31' )
    future_exog = future_exog.predicted_mean

    # updated scores of final evalutation using best parameters
    best_df["r2"] = r2_score(test_sg, forecast.predicted_mean )  
    best_df["rmse"] = root_mean_squared_error(test_sg, forecast.predicted_mean)
    #stores_tunning = pd.DataFrame( best, columns=[ "D", "P", "Q", "d", "p", "q","rmse", "r2"])
    
    stores_tunning_rest.loc[s-1] = best_df.iloc[0].values
    stores_tunning_rest.columns = [ "D", "P", "Q", "d", "p", "q","rmse", "r2"]
    stores_tunning_rest["Stores"] = s
    
    future_exog =  pd.DataFrame(future_exog)
    future_exog ["Stores"] = s
    feature_exog = pd.concat([feature_exog, future_exog ])
    end = time.perf_counter()
    print(f"total Elapsed: {end - start:.6f} seconds")

    stores_tunning_rest.to_csv('Results/Fuel_Price_params.csv',mode='a', index=False, header=False)
    feature_exog.to_csv('Results/Fuel_Price_exog.csv',mode='a', index=False, header=False)


'''






'''

warnings.filterwarnings("ignore")


stores =range(1,46) #df["Store"].unique().astype("int") # temp :7 , 43not working

print(stores)
for s in stores:
    
    start = time.perf_counter()
    results = []
    feature_exog =  pd.DataFrame()
    stores_tunning_rest = pd.DataFrame(columns=range(2) )
   
    print("\n-------Store: ",s, "----------")
    df_store = df.loc[df["Store"]==s,]
    #num_featrs = ['Temperature', 'Fuel_Price', 'CPI',
    #          'Unemployment']
    
    fetr_col = df.columns.get_loc("Fuel_Price")
   
    results = exo_model_run(parm_list = None, featre = fetr_col, use_sarimax = False, use_holts = True ) 
          
    print(results)
    best = results

    best_df = pd.DataFrame([[best['rmse'], best['r2']]], columns=[ "rmse", "r2"])
    
    print(best_df)
    tst_period = 30

    train_sg = df_store.iloc[:-tst_period, fetr_col]
    test_sg = df_store.iloc[-tst_period:, fetr_col]     


    Exp_season_model = ExponentialSmoothing(endog=train_sg,
                                        trend="add",
                                        seasonal=None,
                                        seasonal_periods=52,
                                        initialization_method="estimated"
                                        ).fit()
    y_predicted = Exp_season_model.forecast(len(test_sg)+52)
             

    future_exog = y_predicted[len(test_sg)+1:]

    # updated scores of final evalutation using best parameters
    best_df["r2"] = r2_score(test_sg, y_predicted[:len(test_sg)] )  
    best_df["rmse"] = root_mean_squared_error(test_sg, y_predicted[:len(test_sg)])
    #stores_tunning = pd.DataFrame( best, columns=[ "D", "P", "Q", "d", "p", "q","rmse", "r2"])
    
    stores_tunning_rest.loc[s-1] = best_df.iloc[0].values
    stores_tunning_rest.columns = [ "rmse", "r2"]
    stores_tunning_rest["Stores"] = s
    
    future_exog =  pd.DataFrame(future_exog)
    future_exog ["Stores"] = s
    feature_exog = pd.concat([feature_exog, future_exog ])
    end = time.perf_counter()
    print(f"total Elapsed: {end - start:.6f} seconds")

    stores_tunning_rest.to_csv('Results/Fuel_Price_params.csv',mode='a', index=False, header=False)
    feature_exog.to_csv('Results/Fuel_Price_exog.csv',mode='a', index=False, header=False)


'''