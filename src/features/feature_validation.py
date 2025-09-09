import pandas as pd 
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller


def collinearity_test(df):

    X = df[['Temperature', 'Fuel_Price','CPI', 'Unemployment']]
 
    # add constant for VIF calculation
    overX = X.copy()
    overX['const'] = 1

    vif_data = pd.DataFrame({
            'feature': X.columns,
            'VIF': [variance_inflation_factor(overX.values, i)
                  for i in range(len(X.columns))]
    })
 
    high_vif_f = vif_data.loc[vif_data["VIF"]>5,:]
    
   
    rm_feature = high_vif_f.loc[ high_vif_f["VIF"] == high_vif_f["VIF"].max(),"feature"]

    if rm_feature.empty:
        return ""
    else:
        return(rm_feature.item())


def check_stationary(series, name, alpha=0.05):
    stat, pval, *_ = adfuller(series.dropna())
    return pval

