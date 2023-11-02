from itertools import combinations
import numpy as np
import ta
# from ..utils.utils import IndicatorNotFoundError, Order
import pandas as pd
from scipy.optimize import minimize
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score


def y_generator(file):
    file['price_in_10_days'] = file['Close'].shift(-10)
    file.dropna(inplace=True)
    y_target = []
    for price_10, clos in zip(file['price_in_10_days'], file['Close']):
        if price_10 > clos * 1.005:
            y_target.append(1)
        elif price_10 / clos - 1 < -0.005:
            y_target.append(-1)
        else:
            y_target.append(0)

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


xgboost = GradientBoostingClassifier()

param_grid_gb_1 = {
    'n_estimators': [1, 7, 10, 30, 50, 100, 200, 300, 500, 1000],
    'subsample': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
}

param_grid_gb_2 = {
    'learning_rate': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
    'loss': [‘log_loss’, ‘exponential’]
}

grid_search_gb = GridSearchCV(xgboost, param_grid_gb_1, scoring='f1_weighted', cv=5)
grid_search_gb.fit(data.drop(['target', 'price_in_10_days'], axis=1), data['target'])

best_params_gb = grid_search_gb.best_params_
best_f1_gb = grid_search_gb.best_score_

print(best_params_gb, best_f1_gb)


xgboost = GradientBoostingClassifier()

grid_search_gb = GridSearchCV(xgboost, param_grid_gb_2, scoring='f1_weighted', cv=5)
grid_search_gb.fit(data.drop(['target', 'price_in_10_days'], axis=1), data['target'])

best_params_gb = grid_search_gb.best_params_
best_f1_gb = grid_search_gb.best_score_

print(best_params_gb, best_f1_gb)