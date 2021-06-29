import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

def dropnan_both(pred, obs):
    idx = np.logical_not(np.logical_or(np.isnan(pred), np.isnan(obs)))     
    idx = np.array(idx)
    pred = pred[idx]
    obs = obs[idx]
    return pred, obs

def mse(pred, obs):
    pred, obs = dropnan_both(pred, obs)
    return mean_squared_error(pred, obs, squared=True)

def rmse(pred, obs):
    pred, obs = dropnan_both(pred, obs)
    return mean_squared_error(pred, obs, squared=False)

def bias(pred, obs):
    return np.nanmean(pred)-np.nanmean(obs)

def mae(pred, obs):
    pred, obs = dropnan_both(pred, obs)
    return mean_absolute_error(pred, obs)

def mape(pred, obs):
    pred, obs = dropnan_both(pred, obs)
    return mean_absolute_percentage_error(pred, obs)

def corr_coef(pred, obs):
    pred, obs = dropnan_both(pred, obs)
    return pearsonr(pred, obs)[0]

def std_ratio(pred, obs):
    return np.nanstd(pred)/np.nanstd(obs)

def rmsd(pred, obs):
    pred, obs = dropnan_both(pred, obs)
    return np.sqrt(np.sum(np.power((pred-np.mean(pred))-(obs-np.mean(obs)), 2))/len(obs))
    
def ss4(pred, obs):
    pred, obs = dropnan_both(pred, obs)
    return np.power(corr_coef(pred, obs), 4)/(4*np.power(std_ratio(pred, obs)+std_ratio(obs, pred), 2))

def all_metrics_dict(pred, obs):
    return {'mse':mse(pred, obs), 'rmse':rmse(pred, obs), 'bias':bias(pred, obs), 
            'mae':mae(pred, obs), 'mape':mape(pred, obs), 'corr_coef':corr_coef(pred, obs), 
            'std_ratio':std_ratio(pred, obs), 'rmsd':rmsd(pred, obs), 'ss4':ss4(pred, obs)}

def all_metrics_list(pred, obs):
    return [mse(pred, obs), rmse(pred, obs), bias(pred, obs), mae(pred, obs), mape(pred, obs), corr_coef(pred, obs), 
            std_ratio(pred, obs), rmsd(pred, obs), ss4(pred, obs)]

def all_metrics_names():
    return ['mse','rmse','bias','mae','mape','corr_coef','std_ratio','rmsd','ss4']

def all_metrics_from_dataframes(pred: pd.DataFrame, obs: pd.DataFrame):
    result = {}
    if pred.shape[1] == obs.shape[1]:
        for i, label in enumerate(obs.columns.values):
            result[label] = all_metrics_list(pred.values[:,i], obs.values[:,i])
        return pd.DataFrame(data=result, index=all_metrics_names())


