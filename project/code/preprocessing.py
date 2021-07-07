import pandas as pd

def make_shifts(serie, lags, horizons, dropnan, savename=None):
    shifts = {}

    for l in range(lags-1, -1, -1):
        shifts[f't-{l}'] = serie.shift(l)
  
    for h in range(1, horizons+1):
        shifts[f't+{h}'] = serie.shift(-h)

    shifts = pd.DataFrame(data=shifts)

    if dropnan:
        shifts = shifts.dropna()
    
    shifts_lags = shifts.iloc[:,:lags]
    shifts_horizons = shifts.iloc[:,-horizons:]

    if savename != None:
        shifts_lags.to_csv(f'../data/{savename}_lags.csv')
        shifts_horizons.to_csv(f'../data/{savename}_horizons.csv')

    return shifts_lags, shifts_horizons