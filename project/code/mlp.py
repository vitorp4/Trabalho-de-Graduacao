from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from processing import inverse_scaling
import numpy as np
import pandas as pd

class MLP:

    models = {}
    model = None

    def __init__(self, strategy, input_size=4, horizons=12, inits=1):
        self.strategy = strategy
        self.input_size = input_size
        self.horizons = horizons
        self.inits = inits

    def build(self, hidden_layers=[10], activation='sigmoid', optimizer='sgd', loss='mse'):
        if self.strategy == 'direct':
            for h in range(1, self.horizons+1):
                model = Sequential()
            
                for i, neurons in enumerate(hidden_layers):
                    if i == 0:
                        model.add(Dense(neurons, input_shape=(self.input_size,), activation=activation))
                    else:
                        model.add(Dense(neurons, activation=activation))
                model.add(Dense(1))
                model.compile(optimizer=optimizer, loss=loss)
                self.models[f't+{h}'] = model

        elif self.strategy == 'recursive':
            model = Sequential()
            for i, neurons in enumerate(hidden_layers):
                if i == 0:
                    model.add(Dense(neurons, input_shape=(self.input_size,), activation=activation))
                else:
                    model.add(Dense(neurons, activation=activation))
            model.add(Dense(1))
            model.compile(optimizer=optimizer, loss=loss)
            self.model = model

        elif self.strategy == 'multioutput':
            model = Sequential()
            for i, neurons in enumerate(hidden_layers):
                if i == 0:
                    model.add(Dense(neurons, input_shape=(self.input_size,), activation=activation))
                else:
                    model.add(Dense(neurons, activation=activation))
            model.add(Dense(self.horizons))
            model.compile(optimizer=optimizer, loss=loss)
            self.model = model

    def train(self, X, Y, validation_split=0.2, epochs=200):
        if type(X) == pd.DataFrame:
            X = X.values
        if type(Y) == pd.DataFrame:
            Y = Y.values

        stop = EarlyStopping(monitor='val_loss', patience=5)
        
        if self.strategy == 'direct':
            for h in range(1, self.horizons+1):
                best_val_loss = np.inf
                best_model = None
        
                for _ in range(self.inits):
                    model = self.models[f't+{h}']
                    history = model.fit(X, Y[:,h-1], epochs=epochs, validation_split=validation_split, callbacks=[stop])
                    val_loss = history.history['val_loss'][-1]

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model = model
        
                self.models[f't+{h}'] = best_model
        
        elif self.strategy == 'recursive':
            best_val_loss = np.inf
            best_model = None
        
            for _ in range(self.inits):
                model = self.model
                history = model.fit(X, Y[:,0], epochs=epochs, validation_split=validation_split, callbacks=[stop])
                val_loss = history.history['val_loss'][-1]

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model
        
            self.model = best_model 

        elif self.strategy == 'multioutput':
            best_val_loss = np.inf
            best_model = None
        
            for _ in range(self.inits):
                model = self.model
                history = model.fit(X, Y, epochs=epochs, validation_split=validation_split, callbacks=[stop])
                val_loss = history.history['val_loss'][-1]
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model
        
            self.model = best_model
    
    def predict(self, X, scaler=None, index=None):
        if type(X) == pd.DataFrame:
            X = X.values

        if self.strategy == 'direct':
            pred = {}
            for h in range(1, self.horizons+1):
                model = self.models[f't+{h}']
                pred_h = model.predict(X)
                pred[f't+{h}'] = inverse_scaling(pred_h, scaler=scaler, shifts=h)
            pred = pd.DataFrame(data=pred)

        elif self.strategy == 'recursive':
            pred = {}
            model = self.model

            for h in range(1, self.horizons+1):
                pred_h = model.predict(X)
                X = np.append(X, pred_h, 1)
                X = np.delete(X, 0, 1)
                pred[f't+{h}'] = inverse_scaling(pred_h, scaler=scaler, shifts=h)
            pred = pd.DataFrame(data=pred)

        elif self.strategy == 'multioutput':
            model = self.model
            pred_matrix = model.predict(X)

            pred = {}
            for h in range(1, self.horizons+1):
                pred_h = pred_matrix[:, h-1]
                pred[f't+{h}'] = inverse_scaling(pred_h, scaler=scaler, shifts=h)
            pred = pd.DataFrame(data=pred)

        if index is not None:
            pred.index= index

        return pred

