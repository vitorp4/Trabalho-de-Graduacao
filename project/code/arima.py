import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

class Arima():

    def __init__(self, n_horizons, order):
        self.n_horizons = n_horizons
        self.order = order

    def predict_multiout(self, train, test):
        pred_matrix = np.empty((0, self.n_horizons))
        for i in range(len(test)):
            model = ARIMA(train, order=self.order)
            model = model.fit() 
            pred_line = model.forecast(steps=self.n_horizons)
            pred_matrix = np.concatenate((pred_matrix, pred_line.reshape(1,-1)))
            train = np.append(train, test[i])

        pred_df = pd.DataFrame(data={f't+{h}':pred_matrix[:,h-1] for h in range(1, self.n_horizons+1)}) 
        return pred_df

    def predict_recursive(self, train, test):
        pred_matrix = np.empty((0, self.n_horizons))

        for i in range(len(test)):
            train_h = train
            pred_line = np.array([])

            for _ in range(self.n_horizons):
                model = ARIMA(train_h, order=self.order)
                model = model.fit() 
                pred = model.forecast()
                train_h = np.append(train_h, pred)
                pred_line = np.append(pred_line, pred)
                
            pred_matrix = np.concatenate((pred_matrix, pred_line.reshape(1,-1)))
            train = np.append(train, test[i]) 
        
        pred_df = pd.DataFrame(data={f't+{h}':pred_matrix[:,h-1] for h in range(1, self.n_horizons+1)}) 
        return pred_df