import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import ta


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


rl = LogisticRegression()

param_grid_lr_svc = {'C': [.0001, .001, .01, 0.1, 1, 10, 100]}

grid_search_lr = GridSearchCV(rl, param_grid_lr_svc, scoring='f1_weighted', cv=5)

grid_search_lr.fit(data.drop(['target', 'price_in_10_days'], axis=1), data['target'])
best_C_lr = grid_search_lr.best_params_['C']
best_f1_lr = grid_search_lr.best_score_
print(best_C_lr, best_f1_lr)

# SVC
svc = SVC()
grid_search_svc = GridSearchCV(svc, param_grid_lr_svc, scoring='f1_weighted', cv=5)

grid_search_svc.fit(data.drop(['target', 'price_in_10_days'], axis=1), data['target'])
best_C_svc = grid_search_svc.best_params_['C']
best_f1_svc = grid_search_svc.best_score_

print(best_C_svc, best_f1_svc)

# XGBoost
xgboost = GradientBoostingClassifier()

param_grid_gb_1 = {
    'n_estimators': [1, 7, 10, 30, 50, 100, 200, 300, 500, 1000],
    'subsample': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
}

grid_search_gb = GridSearchCV(xgboost, param_grid_gb_1, scoring='f1_weighted')
grid_search_gb.fit(data.drop(['target', 'price_in_10_days'], axis=1), data['target'])

best_params_gb = grid_search_gb.best_params_
best_f1_gb = grid_search_gb.best_score_

print(best_params_gb, best_f1_gb)

param_grid_gb_2 = {
    'learning_rate': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
    'loss': ['log_loss', 'exponential']
}


xgboost = GradientBoostingClassifier()

grid_search_gb2 = GridSearchCV(xgboost, param_grid_gb_2, scoring='f1_weighted')
grid_search_gb2.fit(data.drop(['target', 'price_in_10_days'], axis=1), data['target'])

best_params_gb2 = grid_search_gb2.best_params_
best_f1_gb2 = grid_search_gb2.best_score_

# {'learning_rate': 1, 'loss': 'log_loss'} 0.5274808861533322
# {'n_estimators': 30, 'subsample': 0.01} 0.5477666784470923


LR_C = best_C_lr
SVC_C = best_C_svc
XGBOOST_N_EST = best_params_gb['n_estimators']
XGBOOST_SUBSAMPLE = best_params_gb['subsample']
XGBOOST_LOSS = best_params_gb['loss']
XGBOOST_LEARNING_RATE = best_params_gb['learning_rate']
