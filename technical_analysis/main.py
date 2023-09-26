from itertools import combinations

import numpy as np
import ta
from utils.utils import IndicatorNotFoundError, Order
import pandas as pd
from scipy.optimize import minimize

data = pd.read_csv('../files/aapl_5m_train.csv')
df_results = pd.DataFrame({'gain': [], 'optimal_sl': [], 'optimal_tp': []})
# df = pd.DataFrame()
# df_sell = pd.DataFrame()
# df_buy = pd.DataFrame()
# COMISSION = 0.0025
# positions = []
# operations = []

def strategies_design(strat, data, window1, window2, window_long, window_short, df, df_buy, df_sell):

    if 'MACD':
        short_macd = ta.trend.MACD(data.Close)
        long_macd = ta.trend.MACD(data.Close)
        df['short_macd'] = long_macd.macd_signal()
        df['long_macd'] = short_macd.macd()
        df_buy['MACD_buy_trade_signal'] = df.short_macd < df.long_macd
        df_sell['MACD_sell_trade_signal'] = df.short_macd > df.long_macd  # buy trade signal

    if 'SMA' in strat:
        short_sma = ta.trend.SMAIndicator(data.Close, window=window_short)
        long_sma = ta.trend.SMAIndicator(data.Close, window=window_long)
        df['short_sma'] = short_sma.sma_indicator()
        df['long_sma'] = long_sma.sma_indicator()
        df_buy['SMA_buy_trade_signal'] = data.short_sma < data.long_sma
        df_sell['SMA_sell_trade_signal'] = data.short_sma > data.long_sma

    if 'RSII' in strat:
        rsii = ta.momentum.RSIIndicator(data.Close)
        df_buy['RSII_buy_trade_signal'] = rsii.rsi() > 70
        df_sell['RSII_sell_trade_signal'] = rsii.rsi() < 30

    if 'ichimoku' in strat:
        ichimoku = ta.trend.IchimokuIndicator(data.High, data.Low)
        df['tenkan_sen'] = ichimoku.ichimoku_a()
        df['kijun_sen'] = ichimoku.ichimoku_b()
        df_buy['ichimoku_buy_trade_signal'] = data.Close > data.tenkan_sen
        df_sell['ichimoku_sell_trade_signal'] = data.Close < data.kijun_sen


    if 'ROC' in strat:
        roc = ta.momentum.ROCIndicator(data.Close, window=window_short)
        df_buy['ROC_buy_trade_signal'] = roc.roc() > 0
        df_sell['ROC_sell_trade_signal'] = roc.roc() < 0

    if 'awesome_oscillator' in strat:
        awesome_oscillator = ta.momentum.AwesomeOscillatorIndicator(data.High, data.Low, window1=window1, window2=window2)
        df['awesome_oscillator'] = awesome_oscillator.ao()
        df_buy['awesome_oscillator_buy_trade_signal'] = df.awesome_oscillator > 0
        df_sell['awesome_oscillator_sell_trade_signal'] = df.awesome_oscillator < 0

def backtest(x):
    cash = 1_000_000
    df = pd.DataFrame()
    df_sell = pd.DataFrame()
    df_buy = pd.DataFrame()
    COMISSION = 0.0025
    positions = []

    window1, window2, window_long, window_short, STOP_LOSS, TAKE_PROFIT = x

    strategies_design(strat, data, window1, window2, window_long, window_short, df, df_buy, df_sell)

    for i, row, i_buy, row_buy, i_sell, row_sell in zip(data.iterrows(), df_buy.iterrows(), df_sell.iterrows()):
        # Close positions
        price = row.Close
        for position in positions:
            j = positions.index(position)
            if position.is_active:
                if (price <= position.stop_loss) & (position.order_type == 'LONG'):
                    # Close position, loss
                    cash += price * (1 - COMISSION)
                    position.is_active = False
                    position.sold_at = price
                    print(f'Closing losing long position bought at {position.bought_at}, sold at {position.sold_at}')

                elif (price >= position.take_profit) & (position.order_type == 'LONG'):
                    # Close position, profit
                    cash += price * (1 - COMISSION)
                    position.is_active = False
                    position.sold_at = price
                    print(f'Closing winning long position bought at {position.bought_at}, sold at {position.sold_at}')

                if (cash < 2 * (price - position.bought_at) + price) & (position.order_type == 'SHORT'):
                    cash -= price * (1 + COMISSION)
                    position.is_active = False
                    position.sold_at = price
                    print(
                        f'Closing losing MARGIN short position bought at {position.bought_at}, sold at {position.sold_at}')

                elif (price >= position.stop_loss) & (position.order_type == 'SHORT'):
                    # Close position, loss
                    cash -= price * (1 + COMISSION)
                    position.is_active = False
                    position.sold_at = price
                    print(f'Closing losing short position bought at {position.bought_at}, sold at {position.sold_at}')

                elif (price <= position.take_profit) & (position.order_type == 'SHORT'):
                    # Close position, profit
                    cash -= price * (1 + COMISSION)
                    position.is_active = False
                    position.sold_at = price
                    print(f'Closing winning short position bought at {position.bought_at}, sold at {position.sold_at}')
        # buy
        if row_buy.MACD_buy_trade_signal and row_buy.SMA_buy_trade_signal and row_buy.RSII_buy_trade_signal and row_buy.ichimoku_buy_trade_signal and row_buy.awesome_oscillator_buy_trade_signal and row_buy.ROC_buy_trade_signal:
            if cash < row.Close * (1 + COMISSION):
                continue
            cash -= row.Close * (1 + COMISSION)

            order = Order(timestamp=row.Timestamp,
                          bought_at=row.Close,
                          stop_loss=row.Close * (1 - STOP_LOSS),
                          take_profit=row.Close * (1 + TAKE_PROFIT),
                          order_type='LONG'
                          )

            positions.append(order)

        # Sell
        # close positons
        price = row.Close

        # do we have money
        # Buy
        if row_sell.MACD_sell_trade_signal and row_sell.SMA_sell_trade_signal and row_sell.RSII_sell_trade_signal and row_sell.ichimoku_sell_trade_signal and row_sell.awesome_oscillator_sell_trade_signal and row_sell.ROC_sell_trade_signal:
            if cash < row.Close * (1 + COMISSION):
                continue
            cash += row.Close * (1 - COMISSION)

            order = Order(timestamp=row.Timestamp,
                          bought_at=row.Close,
                          stop_loss=row.Close * (1 + STOP_LOSS),
                          take_profit=row.Close * (1 - TAKE_PROFIT),
                          order_type='SHORT'
                          )
            positions.append(order)

    cash_still_open_positions = []

    for position in positions:
        if position.order_type == 'LONG' and position.is_active == True:
            cash_still_open_positions.append(position)
        if position.order_type == 'SHORT' and position.is_active == True:
            cash_still_open_positions.append(-position)

    return -sum(cash_still_open_positions) - cash


a = ['ichimoku_a', 'SMA', 'MACD', 'RSI', 'ROC', 'awesome_oscillator']

for j in range(len(a)):
    strategies = [i for i in combinations(a, j)]
    if strategies != [()]:
        for strat in strategies:
            strategies_design(strat, data)
            x0 = np.array([0.5, 0.6])  # Valores iniciales arbitrarios
            # Definir límites de las variables
            limite_inferior = 0.0025
            limite_superior = 0.30

            # Crear una tupla de límites para cada variable
            limites = [(limite_inferior, limite_superior)] * len(x0)

            # Optimización
            result = minimize(backtest, x0, bounds=limites)

            # Resultados óptimos
            optimal_sl, optimal_tp = result.x
            optimal_gain = -result.fun  # Ganancia óptima
            df_trash = pd.DataFrame({'gain': [optimal_gain], 'optimal_sl': [optimal_sl], 'optimal_tp': [optimal_tp]})

            df_results.concat([df_trash], ignore_index=True)

print(df_results)
