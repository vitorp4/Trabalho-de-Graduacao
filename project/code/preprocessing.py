import pandas as pd

def make_shifts(serie, lags, horizons):
    shifts = {}

    for l in range(lags-1, -1, -1):
        shifts[f't-{l}'] = serie.shift(l)
    for h in range(1, horizons+1):
        shifts[f't+{h}'] = serie.shift(-h)

    shifts = pd.DataFrame(data=shifts).dropna()
    shifts.to_csv(f'../data/shifts_{serie.name}.csv')
    return shifts