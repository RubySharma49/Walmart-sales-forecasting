import pandas as pd
import numpy as np

def cal_mape(actual, pred):
    return np.mean(np.abs((actual - pred) / actual)) * 100

def cal_smape(actual, pred):
    return np.mean(2 * np.abs(pred - actual) / (np.abs(actual) + np.abs(pred))) * 100
