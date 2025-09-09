import pandas as pd
from src.features.feature_processing import add_thanksgiving_christmas_flags


def load_weekly_sales(path_file:str, index_col:str = "Date", target_var:str = "Weekly_Sales"):
    df = pd.read_csv(path_file)
    df['y'] = df[target_var]
    df.drop(columns =[target_var], inplace = True)
    df[index_col] = pd.to_datetime(df[index_col], format = "%d-%m-%Y")
    df[index_col] = pd.DatetimeIndex(df[index_col])
    df.set_index(index_col, inplace = True)
    return df


def load_exog_variables(path_exog_file:str, index_col:str = "Date"):
    Future_exog = pd.read_csv(path_exog_file, delimiter="\t")
    Future_exog["Date"] =  pd.to_datetime(Future_exog["Date"])
    Future_exog.index = pd.date_range(start=Future_exog["Date"][0], periods=len(Future_exog), freq='W-FRI')
    Future_exog.drop(columns= ["Date"],inplace=True)


    Future_exog = add_thanksgiving_christmas_flags(
        Future_exog,
        week_anchor="FRI",
        windows={"thanksgiving":[0], "christmas":[-1,0]},
        col_prefix="cal_"
    )
    return Future_exog



def load_best_params(cv_best_params: str):

    stores_best_params = pd.read_csv(cv_best_params,header=None)
    stores_best_params.columns = [ "D", "P", "Q", "d", "p", "q","rmse", "smape","r2","store"]
     
    return stores_best_params


def load_models(path_model: str):
    
    ENGINE = "fastparquet"   # or "pyarrow" if you installed it
    print(path_model+"coefficients_scaled.parquet")
    coeff_s = pd.read_parquet(path_model+"coefficients_scaled.parquet",   engine=ENGINE)
    coeff_u = pd.read_parquet(path_model+"coefficients_unscaled.parquet", engine=ENGINE)
    models  = pd.read_parquet(path_model+"models.parquet",                engine=ENGINE)
    metrics = pd.read_parquet(path_model+"metrics.parquet",               engine=ENGINE)

    return coeff_s, coeff_u, models, metrics