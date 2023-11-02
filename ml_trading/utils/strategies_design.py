from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from ml_trading.hyperparam_fine_tune import LR_C, SVC_C, XGBOOST_N_EST, XGBOOST_SUBSAMPLE, XGBOOST_LOSS, XGBOOST_LEARNING_RATE


def strategies_design(strate, train_data, validate_data, df_buy, df_sell):
    if 'svc':
        svc = SVC(C=SVC_C)
        svc.fit(train_data.drop(['target', 'price_in_10_days', 'Timestamp', 'Gmtoffset', 'Datetime'], axis=1),
                train_data['target'])
        svc_predictions = svc.predict(
            validate_data.drop(['target', 'price_in_10_days', 'Timestamp', 'Gmtoffset', 'Datetime'], axis=1))
        df_buy['svc_buy_trade_signal'] = [True if cat == 1 else False for cat in svc_predictions]
        df_sell['svc_sell_trade_signal'] = [True if cat == -1 else False for cat in svc_predictions]

    if 'lr' in strate:
        rl = LogisticRegression(C=LR_C)
        rl.fit(train_data.drop(['target', 'price_in_10_days', 'Timestamp', 'Gmtoffset', 'Datetime'], axis=1),
               train_data['target'])
        rl_predictions = rl.predict(
            validate_data.drop(['target', 'price_in_10_days', 'Timestamp', 'Gmtoffset', 'Datetime'], axis=1))
        df_buy['rl_buy_trade_signal'] = [True if cat == 1 else False for cat in rl_predictions]
        df_sell['rl_sell_trade_signal'] = [True if cat == -1 else False for cat in rl_predictions]

    if 'xgboost' in strate:
        xgb = GradientBoostingClassifier(learning_rate=XGBOOST_LEARNING_RATE, n_estimators=XGBOOST_N_EST,
                                         subsample=XGBOOST_SUBSAMPLE, loss=XGBOOST_LOSS)
        xgb.fit(train_data.drop(['target', 'price_in_10_days', 'Timestamp', 'Gmtoffset', 'Datetime'], axis=1),
                train_data['target'])
        xgb_predictions = xgb.predict(
            validate_data.drop(['target', 'price_in_10_days', 'Timestamp', 'Gmtoffset', 'Datetime'], axis=1))
        df_buy['xgb_buy_trade_signal'] = [True if cat == 1 else False for cat in xgb_predictions]
        df_sell['xgb_sell_trade_signal'] = [True if cat == -1 else False for cat in xgb_predictions]