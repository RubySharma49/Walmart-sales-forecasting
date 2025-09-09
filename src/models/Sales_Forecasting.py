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
from sklearn.preprocessing import MinMaxScaler
import warnings
import time
import gc
from joblib import Parallel, delayed
from Save_model_info import save_store_artifacts
import fastparquet 
from metrics import cal_mape, cal_smape
from features import Min_max_scling, month_start_flag, add_thanksgiving_christmas_flags
from features import collinearity_test, check_stationary



####################################################################################################

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



Future_exog = pd.read_csv("Results/Future_exog.csv", delimiter="\t")
Future_exog["Date"] =  pd.to_datetime(Future_exog["Date"])
#Future_exog.set_index("Date", inplace = True)
Future_exog.index = pd.date_range(start=Future_exog["Date"][0], periods=len(Future_exog), freq='W-FRI')
Future_exog.drop(columns= ["Date"],inplace=True)


Future_exog = add_thanksgiving_christmas_flags(
    Future_exog,
    week_anchor="FRI",
    windows={"thanksgiving":[0], "christmas":[-1,0]},
    col_prefix="cal_"
)

stores_best_params = pd.read_csv("Results/stores_best_params_rmse.txt",
                                header=None)

stores_best_params.columns = [ "D", "P", "Q", "d", "p", "q","rmse", "smape","r2","store"]

Key_regressors = ["Holiday_Flag", "Month_Start_Flag", 'cal_thanksgiving_w0',  'cal_christmas_w-1', 'cal_christmas_w0']

Store_model_dict = {}

Metric_store = {}

with PdfPages("Results/Final_evaluation.pdf") as pdf:

    
    stores = range(1, 46)
    for s in stores:
        
        print(s)
        fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharey=False, sharex=False,  constrained_layout=True, squeeze=False)
        axs = axes.flat
        df_store = df.loc[df["Store"]==s,]

        parm_list = stores_best_params.loc[stores_best_params["store"]==s,["D", "P", "Q", "d", "p", "q" ]]
        parm_list = parm_list.iloc[0,:].to_dict()

        periods = 21

        num_featrs = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment','y']

        rm_feature = collinearity_test( df_store)
        print(rm_feature)

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

        
        features = all_cols

        train = df_store.iloc[:-periods,]
        test = df_store.iloc[-periods:,]

        train = train.loc[:,features]
        test = test.loc[:,features]


        preprocessor =  MinMaxScaler(feature_range=(0, 1))  


        train_scld = preprocessor.fit_transform(train)
        test_scld = preprocessor.transform(test)


                
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
         
        
        x_scld_train[Key_regressors] = df_store.loc[:, Key_regressors]
        x_scld_test[Key_regressors] = df_store.loc[:, Key_regressors]





        Srmx_model = SARIMAX(endog = y_scld_train , exog = x_scld_train[Key_regressors] ,
                            order = (parm_list['p'],parm_list['d'],parm_list['q']) ,
                            seasonal_order = (parm_list['P'],parm_list['D'],parm_list['Q'],52))
    
        Srmx_res = Srmx_model.fit(disp = False)

        print(Srmx_res.summary())




        forecast = Srmx_res.get_prediction(start = y_scld_test.index[0],
                                end = y_scld_test.index[-1],
                                exog = x_scld_test[Key_regressors] )
    
        y_predicted = forecast.predicted_mean
    
        y_pred_ci = forecast.conf_int().astype(float)  

        if "y" in train.columns:
            loc_y = train.columns.get_loc("y")
        else:
            loc_y = train.columns.get_loc("y_diff_1")

        
        data_range = preprocessor.data_range_[loc_y]
        
        
        org_coeff = pd.concat([Srmx_res.params[Key_regressors], Srmx_res.pvalues[Key_regressors],
                        Srmx_res.conf_int().loc[Key_regressors][0], 
                        Srmx_res.conf_int().loc[Key_regressors][1]],axis=1)
        
        org_coeff.columns = ["Coef", "Pvalue", "CI_lower", "CI_upper"]  

        print("\n org_coeff")
        print(org_coeff)

        
        org_coeff_scl_back = org_coeff.copy()
        org_coeff_scl_back[["Coef","CI_lower", "CI_upper"]] =  org_coeff_scl_back[["Coef","CI_lower", "CI_upper"]]*data_range 

        print("\n scaled back coeff")
        print(org_coeff_scl_back)
       


        meaningful_changes = {
        "Temperature": 10,                # 10 degrees
        "Temperature_diff_1": 10,
        "Fuel_Price": 0.50,  
        "Fuel_Price_diff_1": 0.50,
        "Fuel_Price_diff_2": 0.50,         # $0.50 change
        "Unemployment_diff_2": 0.5,        # 0.5% points
        "CPI": 1,
        "CPI_diff_1": 1,
        "CPI_diff_2": 1,
        "Holiday_Flag": 1,                 # binary change
        "Month_Start_Flag": 1 ,        # binary change,
        'cal_thanksgiving_w0':1,
        'cal_christmas_w-1':1,
        'cal_christmas_w0':1
        }

        # Add business impact calculations
        org_coeff_scl_back["Meaningful_Change"] = org_coeff_scl_back.index.map(meaningful_changes)
        org_coeff_scl_back["dollar_impact"] = org_coeff_scl_back["Coef"] * org_coeff_scl_back["Meaningful_Change"]


        threshold = 50  
        org_coeff_scl_back["Business_Worthy"] = ((org_coeff_scl_back["dollar_impact"].abs() >= threshold) | (org_coeff_scl_back["Pvalue"] < 0.05))


        param_str = ", ".join(f"{k}={v}" for k, v in parm_list.items())

        rmse = root_mean_squared_error(y_scld_test, y_predicted)
        mape = cal_mape(y_scld_test, y_predicted)
        smape = cal_smape(y_scld_test, y_predicted)

       

        r2 = r2_score(y_scld_test, y_predicted)
        rmse_org_scale =  rmse * data_range 

        metric_store = {"rmse": rmse, "mape":mape, "smape": smape,"r2":r2, "rmse_org_scale": rmse_org_scale}

        scaling_info = {
             "scaled_y": True,
             "scaled_x": True,
             "y_scaler": "MinMaxScaler(0,1)",
             "x_scaler": "MinMaxScaler(0,1)",
        }

        print("\nrmse")
        print(rmse)
        print("\nmape")
        print(mape)
        print("\nsmape")
        print(smape)
        print("\nr2")
        print(r2)
        print("reg_coeff")
        print(org_coeff_scl_back)

        #3. Final Forecast of next 11 week, untill the end of year 2012
        fc = Srmx_res.get_forecast(steps =11,
                                exog = Future_exog, )

        fc_mean = fc.predicted_mean
        fc_ci = fc.conf_int().astype(float)
        fc_mean.index = Future_exog.index
        fc_ci.index = Future_exog.index

        y_pred = pd.concat([y_predicted, fc_mean])

        ci_scaled = pd.concat([y_pred_ci,fc_ci ])
        ci_unscaled = pd.concat([y_pred_ci*data_range,fc_ci *data_range])
        ci_unscaled = ci_unscaled/1000000
        print("ci_scaled")
        print(ci_scaled)

      
        ## unscale
        ## unscale

        y_predicted_scl_back = inverse_minmax(y_pred, preprocessor.data_min_[loc_y],preprocessor.data_max_[loc_y] )
        y_train_scl_back = inverse_minmax(y_scld_train, preprocessor.data_min_[loc_y],preprocessor.data_max_[loc_y] )
        y_test_scl_back = inverse_minmax(y_scld_test, preprocessor.data_min_[loc_y],preprocessor.data_max_[loc_y] )

        Store_model_dict[f"S_{s}"]= {
             "coeff_scaled":   org_coeff,
             "coeff_unscaled": org_coeff_scl_back,
             "metrics":        metric_store,
             "sarimax_res":    Srmx_res,
             "scaling":        scaling_info,

        }

      
        ## Plotting
        plot_train_test_forecast(y_scld_train, y_scld_test, y_pred, conf_int=ci_scaled, ax=axs[0],
                                    title="Sales Forecast Scaled", ylabel="Weekly Sales ($)", r2 =r2, rmse = rmse, smape=smape )   
    

            
        
        plot_exog_coefficients(axs[1], org_coeff, title="Exog Coefficients scaled")
         
        print("ci_scaled")
        print(ci_unscaled.info())

        plot_train_test_forecast(y_train_scl_back , y_test_scl_back , y_predicted_scl_back, conf_int=ci_unscaled, ax=axs[2],
                                    title="Sales Forecast original scale", ylabel="Weekly Sales ($)", r2 = r2, rmse = rmse_org_scale, smape = smape )   
        
        
        plot_exog_coefficients( axs[3], org_coeff_scl_back, title="Exog Coefficients original scale")
    



    
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


paths = save_store_artifacts(
                Store_model_dict,
                out_dir="Results/exp_2025_08_17a",  # folder will be created if missing
                run_id="exp_2025_08_17a"
        )