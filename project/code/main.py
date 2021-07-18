#%%
import pandas as pd
import mlflow

from processing import supervised_patterns, scaling, timeseries_split
from metrics import metrics_dict

from mlp import MLP
from lstm import LSTM
from arima import ARIMA
from persistence import Persistence


def mlflow_save(experiment_id, params, metrics, tags):
  with mlflow.start_run(experiment_id=experiment_id) as run:
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.set_tags(tags)

def network_result(strategy, model_name, params, data):
    model = strategy(model_name, input_size=params['input_size'], horizons=params['horizons'], inits=params['inits'])
    model.build(hidden_layers=params['hidden_layers'], activation=params['activation'], optimizer=params['optimizer'], loss=params['loss'])
    model.train(data['X_train'], data['Y_train'], validation_split=params['validation_split'], epochs=params['epochs'])
    pred = model.predict(data['X_test'], data['scaler'])
    return pred

if __name__ == '__main__':

    HORIZONS = 12

    df = pd.read_csv('../data/australia_wind_power.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').rolling(6).mean().iloc[6::6]
    df.index.name = None
    df = df.interpolate(method='polynomial', order=5, axis=0).clip(lower=0)

    for central in ['BOCORWF1']: # ['BOCORWF1','MACARTH1','STARHLWF']
        serie = df[central]
        experiment_id = mlflow.create_experiment(central)

        walkforward_validation, holdout_folder = timeseries_split(serie, groupby='month', train_expand=False, test_size=1, holdout_size=1)

        for folder, (train, test) in enumerate(walkforward_validation, start=1):

            params = {
                'horizons': HORIZONS
            }
            pred = Persistence(horizons=params['horizons']).predict(test)
            for h in range(1, HORIZONS+1):
                tags = {
                    'model': 'persistence', 
                    'horizon': h, 
                    'folder': folder
                }
                metrics = metrics_dict(pred.values[:,h-1], test)
                mlflow_save(experiment_id, params, metrics, tags)

            params = {
                'horizons': HORIZONS,
                'order': (4,1,0)
            }
            pred = ARIMA(horizons=params['horizons'], order=params['order']).predict(train, test)
            for h in range(1, HORIZONS+1):
                tags = {
                    'model': 'ARIMA', 
                    'horizon': h, 
                    'folder': folder
                }
                metrics = metrics_dict(pred.values[:,h-1], test)
                mlflow_save(experiment_id, params, metrics, tags)

            train, test, scaler = scaling(train, test, feature_range=(0,1))

            for input_size in [4,12]:   # [4,12]
                for hidden_layers in [[10], [20,10]]:  # [[10], [20,10]]

                    params = {
                        'input_size': input_size,
                        'hidden_layers': hidden_layers,
                        'timelag': 1,
                        'horizons': HORIZONS,
                        'inits': 1,
                        'activation': 'sigmoid',
                        'optimizer': 'sgd',
                        'loss': 'mse',
                        'validation_split': 0.1,
                        'epochs': 200
                    }

                    X_train, Y_train = supervised_patterns(train, params['input_size'], params['timelag'], params['horizons'], dropnan=True)
                    X_test = supervised_patterns(test, params['input_size'], params['timelag'], dropnan=False)

                    data = {
                        'X_train': X_train,
                        'Y_train': Y_train,
                        'X_test': X_test,
                        'scaler': scaler
                    }
                   
                    for network in ['MLP','LSTM']:
                        for strategy in ['direct','recursive','multioutput']:

                            if network == 'MLP':
                                pred = network_result(MLP, strategy, params, data)
                            elif network == 'LSTM':
                                pred = network_result(LSTM, strategy, params, data)

                            for h in range(1, params['horizons']+1):
                                tags = {
                                    'model': network, 
                                    'strategy': strategy, 
                                    'horizon': h, 
                                    'folder': folder
                                }
                                metrics = metrics_dict(pred.values[:,h-1], test)
                                mlflow_save(experiment_id, params, metrics, tags)


