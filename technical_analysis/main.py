from itertools import combinations

import numpy as np
import ta
# from ..utils.utils import IndicatorNotFoundError, Order
import pandas as pd
from scipy.optimize import minimize


class Order:
    def __init__(self, timestamp, bought_at, stop_loss, take_profit, order_type, sold_at=None, is_active=True):
        self.timestamp = timestamp
        self.bought_at = bought_at
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.order_type = order_type
        self.sold_at = sold_at
        self.is_active = is_active

    def __repr__(self):
        return f'{self.order_type} position - id {self.timestamp}'


data = pd.read_csv('../files/aapl_5m_train.csv')
df_results = pd.DataFrame({'gain': [], 'optimal_sl': [], 'optimal_tp': []})


# df = pd.DataFrame()
# df_sell = pd.DataFrame()
# df_buy = pd.DataFrame()
# COMISSION = 0.0025
# positions = []
# operations = []

def strategies_design(strat, data, df, df_buy, df_sell, *args):  # window1, window2, window_long, window_short, window,

    if 'MACD':
        short_macd = ta.trend.MACD(data.Close)
        long_macd = ta.trend.MACD(data.Close)
        df['short_macd'] = long_macd.macd_signal()
        df['long_macd'] = short_macd.macd()
        df_buy['MACD_buy_trade_signal'] = df.short_macd < df.long_macd
        df_sell['MACD_sell_trade_signal'] = df.short_macd > df.long_macd  # buy trade signal

    if 'SMA' in strat:
        # print(args[3], args[2])
        short_sma = ta.trend.SMAIndicator(data.Close, window=int(args[3]))
        long_sma = ta.trend.SMAIndicator(data.Close, window=int(args[2]))
        df['short_sma'] = short_sma.sma_indicator()
        df['long_sma'] = long_sma.sma_indicator()
        df_buy['SMA_buy_trade_signal'] = df.short_sma < df.long_sma
        df_sell['SMA_sell_trade_signal'] = df.short_sma > df.long_sma

    if 'RSII' in strat:
        rsii = ta.momentum.RSIIndicator(data.Close)
        df_buy['RSII_buy_trade_signal'] = rsii.rsi() > 70
        df_sell['RSII_sell_trade_signal'] = rsii.rsi() < 30

    if 'ichimoku' in strat:
        ichimoku = ta.trend.IchimokuIndicator(data.High, data.Low)
        df['tenkan_sen'] = ichimoku.ichimoku_a()
        df['kijun_sen'] = ichimoku.ichimoku_b()
        df_buy['ichimoku_buy_trade_signal'] = data.Close > df.tenkan_sen
        df_sell['ichimoku_sell_trade_signal'] = data.Close < df.kijun_sen

    if 'ROC' in strat:
        roc = ta.momentum.ROCIndicator(data.Close, window=int(args[4]))
        df_buy['ROC_buy_trade_signal'] = roc.roc() > 0
        df_sell['ROC_sell_trade_signal'] = roc.roc() < 0

    if 'awesome_oscillator' in strat:
        awesome_oscillatorr = ta.momentum.AwesomeOscillatorIndicator(data.High, data.Low, window1=int(args[0]),
                                                                     window2=int(args[1]))
        # print(dir(awesome_oscillator))
        df['awesome_oscillator'] = awesome_oscillatorr.awesome_oscillator()
        df_buy['awesome_oscillator_buy_trade_signal'] = df.awesome_oscillator > 0
        df_sell['awesome_oscillator_sell_trade_signal'] = df.awesome_oscillator < 0


def backtest(x):
    cash = 1_000_000
    df = pd.DataFrame()
    df_sell = pd.DataFrame()
    df_buy = pd.DataFrame()
    COMISSION = 0.0025
    positions = []
    closed_positions = []

    if 'SMA' in strat and 'ROC' in strat and 'awesome_oscillator' in strat:
        window1, window2, window_long, window_short, window, STOP_LOSS, TAKE_PROFIT = x
        strategies_design(strat, data, df, df_buy, df_sell, window1, window2, window_long, window_short, window)
    elif 'SMA' in strat and 'ROC' in strat:
        window_long, window_short, window, STOP_LOSS, TAKE_PROFIT = x
        strategies_design(strat, data, df, df_buy, df_sell, None, None, window_long, window_short, window)
    elif 'SMA' in strat and 'awesome_oscillator' in strat:
        window1, window2, window_long, window_short, STOP_LOSS, TAKE_PROFIT = x
        strategies_design(strat, data, df, df_buy, df_sell, window1, window2, window_long, window_short, None)
    elif 'ROC' in strat and 'awesome_oscillator' in strat:
        window1, window2, window, STOP_LOSS, TAKE_PROFIT = x
        strategies_design(strat, data, df, df_buy, df_sell, window1, window2, None, None, window)
    elif 'awesome_oscillator' in strat:
        window1, window2, STOP_LOSS, TAKE_PROFIT = x
        strategies_design(strat, data, df, df_buy, df_sell, window1, window2, None, None, None)
    elif 'ROC' in strat:
        window, STOP_LOSS, TAKE_PROFIT = x
        strategies_design(strat, data, df, df_buy, df_sell, None, None, None, None, window)
    elif 'SMA' in strat:
        window_long, window_short, STOP_LOSS, TAKE_PROFIT = x
        strategies_design(strat, data, df, df_buy, df_sell, None, None, window_long, window_short, None)
    else:
        STOP_LOSS, TAKE_PROFIT = x
        strategies_design(strat, data, df, df_buy, df_sell, None, None, None, None, None)

    for row, row_buy, row_sell in zip(data.iterrows(), df_buy.iterrows(), df_sell.iterrows()):
        # Close positions
        # print(row[1].Close)
        price = row[1].Close
        for position in positions:
            j = positions.index(position)
            if position.is_active:
                if (price <= position.stop_loss) & (position.order_type == 'LONG'):
                    # Close position, loss
                    cash += price * (1 - COMISSION)
                    position.is_active = False
                    position.sold_at = price
                    closed_positions.append(positions.pop(j))
                    # print(f'Closing losing long position bought at {position.bought_at}, sold at {position.sold_at}')

                elif (price >= position.take_profit) & (position.order_type == 'LONG'):
                    # Close position, profit
                    cash += price * (1 - COMISSION)
                    position.is_active = False
                    position.sold_at = price
                    closed_positions.append(positions.pop(j))
                    # print(f'Closing winning long position bought at {position.bought_at}, sold at {position.sold_at}')

                if (cash < 2 * (price - position.bought_at) + price) & (position.order_type == 'SHORT'):
                    cash -= price * (1 + COMISSION)
                    position.is_active = False
                    position.sold_at = price
                    closed_positions.append(positions.pop(j))
                    # print(f'Closing losing MARGIN short position bought at {position.bought_at}, sold at {position.sold_at}')

                elif (price >= position.stop_loss) & (position.order_type == 'SHORT'):
                    # Close position, loss
                    cash -= price * (1 + COMISSION)
                    position.is_active = False
                    position.sold_at = price
                    closed_positions.append(positions.pop(j))
                    # print(f'Closing losing short position bought at {position.bought_at}, sold at {position.sold_at}')

                elif (price <= position.take_profit) & (position.order_type == 'SHORT'):
                    # Close position, profit
                    cash -= price * (1 + COMISSION)
                    position.is_active = False
                    position.sold_at = price
                    closed_positions.append(positions.pop(j))
                    # print(f'Closing winning short position bought at {position.bought_at}, sold at {position.sold_at}')
        # buy
        # print(row_buy[1].sum(), len(row_buy[1]))
        # if row_buy[1].MACD_buy_trade_signal and row_buy[1].SMA_buy_trade_signal and row_buy[1].RSII_buy_trade_signal and row_buy[1].ichimoku_buy_trade_signal and row_buy[1].awesome_oscillator_buy_trade_signal and row_buy[1].ROC_buy_trade_signal:
        if row_buy[1].sum() == len(df_buy.columns):
            if cash < row[1].Close * (1 + COMISSION):
                continue
            cash -= row[1].Close * (1 + COMISSION)

            order = Order(timestamp=row[1].Timestamp,
                          bought_at=row[1].Close,
                          stop_loss=row[1].Close * (1 - STOP_LOSS),
                          take_profit=row[1].Close * (1 + TAKE_PROFIT),
                          order_type='LONG'
                          )

            positions.append(order)

        # Sell
        # close positons
        price = row[1].Close

        # do we have money
        # Buy
        # if row_sell[1].MACD_sell_trade_signal and row_sell[1].SMA_sell_trade_signal and row_sell[1].RSII_sell_trade_signal and row_sell[1].ichimoku_sell_trade_signal and row_sell[1].awesome_oscillator_sell_trade_signal and row_sell[1].ROC_sell_trade_signal:
        if row_sell[1].sum() == len(df_sell.columns):
            if cash < row[1].Close * (1 + COMISSION):
                continue
            cash += row[1].Close * (1 - COMISSION)

            order = Order(timestamp=row[1].Timestamp,
                          bought_at=row[1].Close,
                          stop_loss=row[1].Close * (1 + STOP_LOSS),
                          take_profit=row[1].Close * (1 - TAKE_PROFIT),
                          order_type='SHORT'
                          )
            positions.append(order)

    cash_still_open_positions = []

    for position in positions:
        if position.order_type == 'LONG' and position.is_active == True:
            cash_still_open_positions.append(data.Close.iloc[-1])
        if position.order_type == 'SHORT' and position.is_active == True:
            cash_still_open_positions.append(-data.Close.iloc[-1])

    return -sum(cash_still_open_positions) - cash


a = ['ichimoku', 'SMA', 'MACD', 'RSII', 'ROC', 'awesome_oscillator']

coun = 0
for j in range(len(a)):
    strategies = [i for i in combinations(a, j)]
    if strategies != [()]:
        for strat in strategies:
            # print(strat)
            # strategies_design(strat, data)

            if 'SMA' in strat and 'ROC' in strat and 'awesome_oscillator' in strat:
                x0 = np.array([5, 15, 15, 5, 10, 0.15, 0.15])  # Valores iniciales arbitrarios
                # window1, window2, window_long, window_short, STOP_LOSS, TAKE_PROFIT = x
                limites = [(2, 10), (11, 30), (11, 30), (2, 10), (3, 30), (0.0025, 0.30), (0.0025, 0.30)]
                result = minimize(backtest, x0, bounds=limites, method="Nelder-Mead",
                                  options={"maxiter": 100, "maxfev": 100})
                optimal_gain = -result.fun  # Ganancia óptima
                df_trash = pd.DataFrame({'gain': [optimal_gain], 'strategy': [strat], 'optimal_params': [result.x]})
                df_results = pd.concat([df_results, df_trash], ignore_index=True)

            elif 'SMA' in strat and 'ROC' in strat:
                x0 = np.array([15, 5, 10, 0.15, 0.15])  # Valores iniciales arbitrarios`
                # window1, window2, window_long, window_short, STOP_LOSS, TAKE_PROFIT = x
                limites = [(11, 30), (2, 10), (3, 30), (0.0025, 0.30), (0.0025, 0.30)]
                result = minimize(backtest, x0, bounds=limites, method="Nelder-Mead",
                                  options={"maxiter": 100, "maxfev": 100})
                optimal_gain = -result.fun  # Ganancia óptima
                df_trash = pd.DataFrame({'gain': [optimal_gain], 'strategy': [strat], 'optimal_params': [result.x]})
                df_results = pd.concat([df_results, df_trash], ignore_index=True)

            elif 'SMA' in strat and 'awesome_oscillator' in strat:
                x0 = np.array([5, 15, 15, 5, 0.15, 0.15])  # Valores iniciales arbitrarios
                # window1, window2, window_long, window_short, STOP_LOSS, TAKE_PROFIT = x
                limites = [(2, 10), (11, 30), (11, 30), (2, 10), (0.0025, 0.30), (0.0025, 0.30)]
                result = minimize(backtest, x0, bounds=limites, method="Nelder-Mead",
                                  options={"maxiter": 100, "maxfev": 100})
                optimal_gain = -result.fun  # Ganancia óptima
                df_trash = pd.DataFrame({'gain': [optimal_gain], 'strategy': [strat], 'optimal_params': [result.x]})
                df_results = pd.concat([df_results, df_trash], ignore_index=True)
            elif 'ROC' in strat and 'awesome_oscillator' in strat:
                x0 = np.array([5, 15, 10, 0.15, 0.15])  # Valores iniciales arbitrarios
                # window1, window2, window_long, window_short, STOP_LOSS, TAKE_PROFIT = x
                limites = [(2, 10), (11, 30), (2, 30), (0.0025, 0.30), (0.0025, 0.30)]
                result = minimize(backtest, x0, bounds=limites, method="Nelder-Mead",
                                  options={"maxiter": 100, "maxfev": 100})
                optimal_gain = -result.fun  # Ganancia óptima
                df_trash = pd.DataFrame({'gain': [optimal_gain], 'strategy': [strat], 'optimal_params': [result.x]})
                df_results = pd.concat([df_results, df_trash], ignore_index=True)
            elif 'awesome_oscillator' in strat:
                x0 = np.array([5, 15, 0.15, 0.15])  # Valores iniciales arbitrarios
                # window1, window2, window_long, window_short, STOP_LOSS, TAKE_PROFIT = x
                limites = [(2, 10), (11, 30), (0.0025, 0.30), (0.0025, 0.30)]
                result = minimize(backtest, x0, bounds=limites, method="Nelder-Mead",
                                  options={"maxiter": 100, "maxfev": 100})
                optimal_gain = -result.fun  # Ganancia óptima
                df_trash = pd.DataFrame({'gain': [optimal_gain], 'strategy': [strat], 'optimal_params': [result.x]})
                df_results = pd.concat([df_results, df_trash], ignore_index=True)
            elif 'ROC' in strat:
                x0 = np.array([10, 0.15, 0.15])  # Valores iniciales arbitrarios
                # window1, window2, window_long, window_short, STOP_LOSS, TAKE_PROFIT = x
                limites = [(2, 30), (0.0025, 0.30), (0.0025, 0.30)]
                result = minimize(backtest, x0, bounds=limites, method="Nelder-Mead",
                                  options={"maxiter": 100, "maxfev": 100})
                optimal_gain = -result.fun  # Ganancia óptima
                df_trash = pd.DataFrame({'gain': [optimal_gain], 'strategy': [strat], 'optimal_params': [result.x]})
                df_results = pd.concat([df_results, df_trash], ignore_index=True)
            elif 'SMA' in strat:
                x0 = np.array([15, 5, 0.15, 0.15])  # Valores iniciales arbitrarios
                # window1, window2, window_long, window_short, STOP_LOSS, TAKE_PROFIT = x
                limites = [(11, 30), (2, 10), (0.0025, 0.30), (0.0025, 0.30)]
                result = minimize(backtest, x0, bounds=limites, method="Nelder-Mead",
                                  options={"maxiter": 100, "maxfev": 100})
                optimal_gain = -result.fun  # Ganancia óptima
                df_trash = pd.DataFrame({'gain': [optimal_gain], 'strategy': [strat], 'optimal_params': [result.x]})
                df_results = pd.concat([df_results, df_trash], ignore_index=True)
            else:
                x0 = np.array([0.15, 0.15])  # Valores iniciales arbitrarios
                # window1, window2, window_long, window_short, STOP_LOSS, TAKE_PROFIT = x
                limites = [(0.0025, 0.30), (0.0025, 0.30)]
                result = minimize(backtest, x0, bounds=limites, method="Nelder-Mead",
                                  options={"maxiter": 100, "maxfev": 100})
                optimal_gain = -result.fun  # Ganancia óptima
                df_trash = pd.DataFrame({'gain': [optimal_gain], 'strategy': [strat], 'optimal_params': [result.x]})
                df_results = pd.concat([df_results, df_trash], ignore_index=True)
        coun += 1
        print(f"Finished Backtest {coun}/64")

print(df_results)
df_results.to_csv('resultados.csv')
