import pandas as pd
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import optuna
import ta


def y_generator(file):
    file['price_in_10_days'] = file['Close'].shift(-10)
    file.dropna(inplace=True)
    y_target = []
    for price_10, clos in zip(file['price_in_10_days'], file['Close']):
        if price_10 > clos * 1.005:
            y_target.append(2)
        elif price_10 / clos - 1 < -0.005:
            y_target.append(0)
        else:
            y_target.append(1)

    file['target'] = y_target
    return file


data = pd.read_csv('../files/aapl_5m_train.csv')
df_results = pd.DataFrame({'gain': [], 'optimal_sl': [], 'optimal_tp': []})

data = y_generator(data)

data['rend'] = data['Close'].pct_change()

short_sma = ta.trend.SMAIndicator(data.Close, window=5)
long_sma = ta.trend.SMAIndicator(data.Close, window=15)
data['short_sma'] = short_sma.sma_indicator()
data['long_sma'] = long_sma.sma_indicator()
data['rsi'] = ta.momentum.RSIIndicator(data.Close).rsi()

data.drop(['Timestamp', 'Gmtoffset', 'Datetime'], inplace=True, axis=1)
data.dropna(inplace=True)

X_train = data.drop(['target', 'price_in_10_days'], axis=1)
y_train = data['target']
##############################################################################

def objective_mlp(trial):
    model = keras.Sequential()

    # Define the search space for number of layers and neurons
    num_layers = trial.suggest_categorical('num_layers', [2, 3, 4]) #agregar lista en lugar de min max
    units = trial.suggest_categorical('units', [[32, 16, 8, 4], [64, 32, 16, 8], [128, 64, 32, 16], [256, 128, 32, 8], [256, 64, 16, 4]])#agregar lista en lugar de min max
    # loss = trial.suggest_categorical('loss', ['CategoricalCrossentropy', 'SparseCategoricalCrossentropy'])
    activation = trial.suggest_categorical('activation', ['sigmoid', 'softmax', 'tanh'])
    # optimizer = trial.suggest_categorical('optimizer', [keras.optimizers.SGD(), 'rmsprop', 'adam'])

    for i in range(num_layers):
        model.add(Dense(units[i], activation='relu'))

    model.add(Dense(3, activation=activation))

    model.compile(optimizer='adam',
                  loss='SparseCategoricalCrossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train, validation_data=(X_train, y_train), epochs=10, batch_size=32, verbose=0)

    # Optimize for validation accuracy
    return history.history['val_accuracy'][-1]


# Create and run the study
study = optuna.create_study(direction='maximize')
study.optimize(objective_mlp, n_trials=3)

# Print the best parameters and accuracy
print("Best parameters found: ", study.best_params)
print("Best validation accuracy found: ", study.best_value)

##################  RNN  #######################

def objective_rnn(trial):
    model = Sequential()

    num_layers = trial.suggest_categorical('num_layers', [2, 3]) #agregar lista en lugar de min max
    units = trial.suggest_categorical('units', [[32, 16, 8, 4], [64, 32, 16, 8], [128, 64, 32, 16], [256, 128, 32, 8], [256, 64, 16, 4]])#agregar lista en lugar de min max
    # loss = trial.suggest_categorical('loss', ['CategoricalCrossentropy', 'SparseCategoricalCrossentropy'])
    activation = trial.suggest_categorical('activation', ['sigmoid', 'softmax', 'tanh'])
    dropouts = trial.suggest_categorical('dropouts', [0, 1, 2])
    # optimizer = trial.suggest_categorical('optimizer', [keras.optimizers.SGD(), 'rmsprop', 'adam'])

    for i in range(num_layers):
        if dropouts == 2 and i < 2:
            model.add(LSTM(units=units[i], activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)))
            model.add(Dropout(0.2))
        elif dropouts == 1 and i < 1:
            model.add(LSTM(units=units[i], activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)))
            model.add(Dropout(0.2))
        else:
            model.add(LSTM(units=units[i], activation='relu', return_sequences=False, input_shape=(X_train.shape[1], 1)))

    model.add(Dense(10))
    model.add(Dense(3, activation=activation))

    model.compile(optimizer='adam',
                  loss='SparseCategoricalCrossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train, validation_data=(X_train, y_train), epochs=10, batch_size=32, verbose=0)

    # Optimize for validation accuracy
    return history.history['val_accuracy'][-1]


# Create and run the study
study = optuna.create_study(direction='maximize')
study.optimize(objective_rnn, n_trials=3)

# Print the best parameters and accuracy
print("Best parameters found: ", study.best_params)
print("Best validation accuracy found: ", study.best_value)




# param_grid_mlp = {'?????': [.0001, .001, .01, 0.1, 1, 10, 100?????]}
# input_dim = 10
#
# mlp = keras.Sequential([
#     keras.layers.Dense(units=128, activation='relu', input_shape=(X_train.shape[1], 1)),
#     keras.layers.Dense(units=64, activation='relu'),  # Hidden layer
#     keras.layers.Dense(units=3, activation='softmax')  # Output layer for three classes
# ])
#
# mlp.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
# grid_search_mlp = GridSearchCV(mlp, param_grid_mlp, scoring='f1_weighted', cv=5)
#
# grid_search_mlp.fit(X_train, y_train)
# best_?_lr = grid_search_mlp.best_params_['?']
# best_f1_lr = grid_search_mlp.best_score_
# print(best_?_lr, best_f1_lr)


# mlp.fit(X_train, y_train, epochs=10, batch_size=32)

#
# # Create an RNN model
# rnn = Sequential()
# rnn.add(LSTM(units=40, activation='relu', return_sequences=True, input_shape=(data.shape[1], 1)))
# # model.add(Dropout(0.2))
# rnn.add(LSTM(units=40, activation='relu', return_sequences=True))
# #model.add(Dropout(0.2))
# rnn.add(LSTM(units=40, activation='relu'))
# rnn.add(Dense(10))
# rnn.add(Dense(3, activation='softmax'))
#
# rnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# rnn.fit(X_train, y_train, epochs=10, batch_size=32)
#
#
# param_grid_lr_svc = {'C': [.0001, .001, .01, 0.1, 1, 10, 100]}
#
# grid_search_lr = GridSearchCV(rl, param_grid_lr_svc, scoring='f1_weighted', cv=5)
#
# grid_search_lr.fit(data.drop(['target', 'price_in_10_days'], axis=1), data['target'])
# best_C_lr = grid_search_lr.best_params_['C']
# best_f1_lr = grid_search_lr.best_score_
# print(best_C_lr, best_f1_lr)
#
# # SVC
# svc = SVC()
# grid_search_svc = GridSearchCV(svc, param_grid_lr_svc, scoring='f1_weighted', cv=5)
#
# grid_search_svc.fit(data.drop(['target', 'price_in_10_days'], axis=1), data['target'])
# best_C_svc = grid_search_svc.best_params_['C']
# best_f1_svc = grid_search_svc.best_score_
#
# print(best_C_svc, best_f1_svc)
#
# # XGBoost
# xgboost = GradientBoostingClassifier()
#
# param_grid_gb_1 = {
#     'n_estimators': [1, 7, 10, 30, 50, 100, 200, 300, 500, 1000],
#     'subsample': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
# }
#
# grid_search_gb = GridSearchCV(xgboost, param_grid_gb_1, scoring='f1_weighted')
# grid_search_gb.fit(data.drop(['target', 'price_in_10_days'], axis=1), data['target'])
#
# best_params_gb = grid_search_gb.best_params_
# best_f1_gb = grid_search_gb.best_score_
#
# print(best_params_gb, best_f1_gb)
#
# param_grid_gb_2 = {
#     'learning_rate': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
#     'loss': ['log_loss', 'exponential']
# }
#
#
# xgboost = GradientBoostingClassifier()
#
# grid_search_gb2 = GridSearchCV(xgboost, param_grid_gb_2, scoring='f1_weighted')
# grid_search_gb2.fit(data.drop(['target', 'price_in_10_days'], axis=1), data['target'])
#
# best_params_gb2 = grid_search_gb2.best_params_
# best_f1_gb2 = grid_search_gb2.best_score_
#
# # {'learning_rate': 1, 'loss': 'log_loss'} 0.5274808861533322
# # {'n_estimators': 30, 'subsample': 0.01} 0.5477666784470923
#
#
# LR_C = best_C_lr
# SVC_C = best_C_svc
# XGBOOST_N_EST = best_params_gb['n_estimators']
# XGBOOST_SUBSAMPLE = best_params_gb['subsample']
# XGBOOST_LOSS = best_params_gb['loss']
# XGBOOST_LEARNING_RATE = best_params_gb['learning_rate']
