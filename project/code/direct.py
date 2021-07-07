from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd

class Direct:

    models = {}

    def __init__(self, type, n_horizons, n_inits):
        if type not in ['MLP','LSTM']:
            raise('Tipo de rede neural inválido.')
        else: 
            self.type = type
            self.n_horizons = n_horizons
            self.n_inits = n_inits

    def build(self, hidden_layers, input_size, activation, optimizer, loss):
        for h in range(1, self.n_horizons+1):
            model = Sequential()

            if self.type == 'MLP':
                for i, neurons in enumerate(hidden_layers):
                    if i == 0:
                        model.add(Dense(neurons, input_shape=(input_size,), activation=activation))
                    else:
                        model.add(Dense(neurons, activation=activation))

            elif self.type == 'LSTM':
                if len(hidden_layers) == 1:
                    neurons = hidden_layers[0]
                    model.add(LSTM(neurons, activation=activation, input_shape=(input_size, 1)))
                else:
                    for i, neurons in enumerate(hidden_layers):
                        if i == 0:
                            model.add(LSTM(neurons, activation=activation, return_sequences=True, input_shape=(input_size, 1)))
                        else:
                            model.add(LSTM(neurons, activation=activation))

            model.add(Dense(1))
            model.compile(optimizer=optimizer, loss=loss)
            self.models[f't+{h}'] = model

    def train(self, X, y, validation_split, epochs):
        stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        if self.type == 'LSTM':
            X = X.reshape((X.shape[0], X.shape[1], 1))

        for h in range(1, self.n_horizons+1):
            best_val_loss = np.inf
            best_model = None
    
            for _ in range(self.n_inits):
                model = self.models[f't+{h}']
                history = model.fit(X, y[:,h-1], epochs=epochs, validation_split=validation_split, callbacks=[stop])
                val_loss = history.history['val_loss'][-1]

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model
     
            self.models[f't+{h}'] = best_model
    
    def predict(self, X, scaler=None):
        pred = np.empty((X.shape[0], 0))
        if self.type == 'LSTM':
            X = X.reshape((X.shape[0], X.shape[1], 1))

        for h in range(1, self.n_horizons+1):
            model = self.models[f't+{h}']
            pred = np.append(pred, model.predict(X), 1)
            
        if scaler != None:
            pred = scaler.inverse_transform(pred)

        pred = pd.DataFrame(data={f't+{h}':pred[:,h-1] for h in range(1, self.n_horizons+1)})
        return pred
