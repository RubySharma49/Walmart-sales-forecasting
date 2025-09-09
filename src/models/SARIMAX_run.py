from statsmodels.tsa.statespace.sarimax import SARIMAX


def SARIMAX_train(y, x, parm_list):
 
    Srmx_model = SARIMAX(endog = y , exog = x,
                          order = (parm_list['p'],parm_list['d'],parm_list['q']),
                          seasonal_order =(parm_list['P'],parm_list['D'],parm_list['Q'],52) )

    return Srmx_model.fit(disp = False)


def SARIMAX_forecast(Srmx_model, y_test, x_test):
     
     forecast = Srmx_model.get_prediction(start = y_test.index[0],
                               end = y_test.index[-1],
                               exog = x_test )
     

     return forecast