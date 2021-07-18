import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA as ARIMA_base
from processing import inverse_scaling

class ARIMA():

    def __init__(self, horizons, order):
        self.horizons = horizons
        self.order = order

    def predict(self, train, test, index=None):
        if type(train) == pd.Series:
            train = train.values
        if type(test) == pd.Series:
            test = test.values

        pred_matrix = np.empty((0, self.horizons))
        for i in range(len(test)):
            model = ARIMA_base(train, order=self.order)
            model = model.fit() 
            pred_line = model.forecast(steps=self.horizons)
            pred_matrix = np.concatenate((pred_matrix, pred_line.reshape(1,-1)))
            train = np.append(train, test[i])

        pred = {}
        for h in range(1, self.horizons+1):
            pred_h = pred_matrix[:, h-1]
            pred[f't+{h}'] = inverse_scaling(pred_h, shifts=h)
        pred = pd.DataFrame(data=pred)

        if index is not None:
            pred.index = index

        return pred


# for label, serie in df.items():
#     plt.figure(figsize=(15,4))
#     plt.title(label)
#     plt.plot(serie)
#     plt.savefig(f'../images/{label}.png')

# for label, serie in df.items():
#     plot_pacf(serie, title=f'PACF {label}', lags=40)
#     plt.savefig(f'../images/PACF_{label}.png')

# for label, serie in df.items():
#     plot_acf(serie, title=f'ACF {label}', lags=100)
#     plt.savefig(f'../images/ACF_{label}.png')

