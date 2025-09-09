import sys, os
sys.path.append(os.path.abspath("."))
from src.models.cv import model_run
from src.utils.config import load_config
from src.data.load import load_weekly_sales
import argparse
import pandas as pd
from src.features.feature_validation import collinearity_test, check_stationary
from src.features.feature_processing import add_thanksgiving_christmas_flags, month_start_flag 
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
from difflib import get_close_matches



def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--config", default="configs/Params_set.yml")

    args = ap.parse_args()

    cfg = load_config(args.config)

    df = load_weekly_sales(cfg["data"]["path_file"],
                      cfg["data"]["index_col"],
                      cfg["data"]["target_var"])

    stores = range(15, 45)#df["Store"].unique().astype("int") 
 

    for s in stores:

        stores_tunning_rest = pd.DataFrame(columns=range(9) )

        # Get specific store
        df_store = df.loc[df["Store"]==s,]

        num_featrs = list(cfg["cross_validation"]["numeric_featres"])

        binary_featrs = list(cfg["cross_validation"]["binary_features"])

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

        # Add thanksgiving and christmas flags
        df_store.loc[df_store.index.month.isin([11, 12]), "Holiday_Flag"] = 0

   
        df_store = add_thanksgiving_christmas_flags(
                df_store,
                week_anchor="FRI",
                windows={"thanksgiving":[0], "christmas":[-1,0]},
                col_prefix="cal_"
        )
        
        # Add month start sale flags
        Month_Start_Flag = month_start_flag(df_store.index)
        df_store['Month_Start_Flag'] = Month_Start_Flag

        param = cfg["cross_validation"]["model_param"]
        
        grid = ParameterGrid(param)

        results = Parallel(n_jobs=-2, verbose=10)(
            delayed(model_run)(df_store, params, num_featrs,  binary_featrs ) for params in grid
        )      
        
        best = min(results, key=lambda x: x["rmse"])

        best_df = pd.DataFrame([best['params'] + [best['rmse'], best['smape'], best['r2']]], columns=[ "D", "P", "Q", "d", "p", "q","rmse", "smape", "r2"])

        stores_tunning_rest.loc[s-1] = best_df.iloc[0].values
        stores_tunning_rest.columns = [ "D", "P", "Q", "d", "p", "q","rmse", "smape", "r2"]
        stores_tunning_rest["Stores"] = s
        stores_tunning_rest.to_csv(cfg["cross_validation"]["cv_best_param_file"],mode='a', index=False, header=False)


if __name__ == "__main__":
    main()