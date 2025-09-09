import sys, os
sys.path.append(os.path.abspath("."))
from sklearn.metrics import r2_score, mean_squared_error,root_mean_squared_error
from src.metrics.compute_metrics import cal_mape, cal_smape
from src.models.cv import model_run, endog_exog_split
from src.utils.config import load_config
from src.data.load import load_weekly_sales, load_exog_variables, load_best_params
import argparse
import pandas as pd
from src.features.feature_validation import collinearity_test, check_stationary
from src.features.feature_processing import add_thanksgiving_christmas_flags, month_start_flag,inverse_minmax 
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from src.viz.visualize_data import plot_train_test_forecast, plot_exog_coefficients
import pickle
from src.models.Save_model_info import save_store_artifacts
from difflib import get_close_matches






def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--config", default="configs/Params_set.yml")

    args = ap.parse_args()

    cfg = load_config(args.config)

    df = load_weekly_sales(cfg["data"]["path_file"],
                      cfg["data"]["index_col"],
                      cfg["data"]["target_var"])

   
    future_exog = load_exog_variables(cfg["final_evaluation"]["future_exog_file"])

    periods = cfg["final_evaluation"]["no_weeks_test"]

    Key_regressors = cfg["final_evaluation"]["binary_features"]

    Store_model_dict = {}

    with PdfPages(cfg["final_evaluation"]["path_forecast_plot"]) as pdf:

        
        stores =  range(1,25)#df["Store"].unique().astype("int") 
        for s in stores:
            
            print(s)
            fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharey=False, sharex=False,  constrained_layout=True, squeeze=False)
            axs = axes.flat

            df_store = df.loc[df["Store"]==s,]

            num_featrs = list(cfg["final_evaluation"]["numeric_featres"])

            binary_featrs = list(cfg["final_evaluation"]["binary_features"])

            # Load the cv best params of store_s
            stores_best_params = load_best_params(cfg["final_evaluation"]["cv_best_param_file"])
            parm_list = stores_best_params.loc[stores_best_params["store"]==s,["D", "P", "Q", "d", "p", "q" ]]
            parm_list = parm_list.iloc[0,:].to_dict()
           

            # Handling collinearity of exogenous variables 
            if len(num_featrs) > 1:
                rm_feature = collinearity_test( df_store)
                if rm_feature!="":
                    num_featrs.remove(rm_feature )
                    df_store.drop(rm_feature, axis = 1, inplace = True)

            # Check for stationary and keep final differenced variables
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

                # if any of the num features are modified, add the modified names in num_features
            modified_var = [var for var in num_featrs if var not in df_store.columns]

            for var in modified_var:
                match = get_close_matches(var, df_store.columns, n=1, cutoff=0.1)
                if match:
                    num_featrs.append(match[0])
                    num_featrs.remove(var)
    
            

            df_store = df_store.dropna()

            df_store.loc[df_store.index.month.isin([11, 12]), "Holiday_Flag"] = 0

            # Adding thanksgiving and christmas flags
            df_store = add_thanksgiving_christmas_flags(
                df_store,
                week_anchor="FRI",
                windows={"thanksgiving":[0], "christmas":[-1,0]},
                col_prefix="cal_"
            )

            Month_Start_Flag = month_start_flag(df_store.index)
            df_store['Month_Start_Flag'] = Month_Start_Flag

            features = num_featrs + binary_featrs
            
            # Spilitting data into train and test
            train = df_store.iloc[:-periods,]
            test = df_store.iloc[-periods:,]

            train = train.loc[:,features]
            test = test.loc[:,features]

            # Split data into endog and exog dataframes
            Srmx_model, forecast,  preprocessor, x_scld_train, x_scld_test, y_scld_train, y_scld_test  = endog_exog_split(train, test, num_featrs, binary_featrs, parm_list)
            y_predicted = forecast.predicted_mean
            y_pred_ci = forecast.conf_int().astype(float)  


            # calculate the model evaluation metrics
            rmse = root_mean_squared_error(y_scld_test, y_predicted)
            mape = cal_mape(y_scld_test, y_predicted)
            smape = cal_smape(y_scld_test, y_predicted)
            r2 = r2_score(y_scld_test, y_predicted)

            # get index of column of y
            if "y" in train.columns:
                loc_y = train.columns.get_loc("y")
            else:
                loc_y = train.columns.get_loc("y_diff_1")

            
            data_range = preprocessor.data_range_[loc_y]
            
            # get regressor coeffs based on scaled target variable
            org_coeff = pd.concat([Srmx_model.params[Key_regressors], Srmx_model.pvalues[Key_regressors],
                            Srmx_model.conf_int().loc[Key_regressors][0], 
                            Srmx_model.conf_int().loc[Key_regressors][1]],axis=1)
            org_coeff.columns = ["Coef", "Pvalue", "CI_lower", "CI_upper"]  

            # get regressor coeffs based on original scaled target variable
            org_coeff_scl_back = org_coeff.copy()
            org_coeff_scl_back[["Coef","CI_lower", "CI_upper"]] =  org_coeff_scl_back[["Coef","CI_lower", "CI_upper"]]*data_range 
            meaningful_changes = cfg["final_evaluation"]["meaningful_changes"]
            # Add business impact calculations
            org_coeff_scl_back["Meaningful_Change"] = org_coeff_scl_back.index.map(meaningful_changes)
            org_coeff_scl_back["dollar_impact"] = org_coeff_scl_back["Coef"] * org_coeff_scl_back["Meaningful_Change"]

            
            # rmse scaled back to orginal target variable
            rmse_org_scale =  rmse * data_range 
            metric_store = {"rmse": rmse, "mape":mape, "smape": smape,"r2":r2, "rmse_org_scale": rmse_org_scale}

            scaling_info = {
                "scaled_y": True,
                "scaled_x": False,
                "y_scaler": "MinMaxScaler(0,1)",
                "x_scaler": "MinMaxScaler(0,1)",
            }

            # Final Forecast of next 11 week, untill the end of year 2012
            fc = Srmx_model.get_forecast(steps = cfg["final_evaluation"]["no_weeks_forecast"],
                                         exog = future_exog )

            fc_mean = fc.predicted_mean
            fc_ci = fc.conf_int().astype(float)
            fc_mean.index = future_exog.index
            fc_ci.index = future_exog.index

            y_pred = pd.concat([y_predicted, fc_mean])

            ci_scaled = pd.concat([y_pred_ci,fc_ci ])
            ci_unscaled = pd.concat([y_pred_ci*data_range,fc_ci *data_range])
            ci_unscaled = ci_unscaled/1000000
            ci_unscaled = pd.DataFrame(ci_unscaled)

            y_predicted_scl_back = inverse_minmax(y_pred, preprocessor.data_min_[loc_y],preprocessor.data_max_[loc_y] )
            y_train_scl_back = inverse_minmax(y_scld_train, preprocessor.data_min_[loc_y],preprocessor.data_max_[loc_y] )
            y_test_scl_back = inverse_minmax(y_scld_test, preprocessor.data_min_[loc_y],preprocessor.data_max_[loc_y] )

            Store_model_dict[f"S_{s}"]= {
                "coeff_scaled":   org_coeff,
                "coeff_unscaled": org_coeff_scl_back,
                "metrics":        metric_store,
                "sarimax_res":    Srmx_model,
                "scaling":        scaling_info,

            }
           
       
        
            ## Plot the train test forecast
            plot_train_test_forecast(y_scld_train, y_scld_test, y_pred, conf_int=ci_scaled, ax=axs[0],
                                        title="Sales Forecast Scaled", ylabel="Weekly Sales ($)", r2 =r2, rmse = rmse, smape=smape )   
        
            plot_exog_coefficients(axs[1], org_coeff, title="Exog Coefficients scaled")


            plot_train_test_forecast(y_train_scl_back , y_test_scl_back , y_predicted_scl_back, conf_int=ci_unscaled, ax=axs[2],
                                        title="Sales Forecast original scale", ylabel="Weekly Sales ($)", r2 = r2, rmse = rmse_org_scale, smape = smape )   
            
            plot_exog_coefficients( axs[3], org_coeff_scl_back, title="Exog Coefficients original scale")
        
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)   


    paths = save_store_artifacts(
                    Store_model_dict,
                    out_dir=cfg["final_evaluation"]["path_model_save"],  # folder will be created if missing
                    run_id=cfg["final_evaluation"]["path_model_runid"]
            )
if __name__ == "__main__":
  main()