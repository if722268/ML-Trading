import pandas as pd
from ml_trading.utils.utils import Order
from ml_trading.utils.training_data_generation import x_y_generator
from ml_trading.utils import strategies_design


def perform():
    data = pd.read_csv('./files/aapl_5m_train.csv')
    data_val = pd.read_csv('./files/aapl_5m_validation.csv')
    df_results = pd.DataFrame({'gain': [], 'strategy': []})

    data = x_y_generator(data)

    data_validation = x_y_generator(data_val)

    portfolio_values = []
    cash_values = []

    strategies = [['svc', 'xgboost', 'lr'], ['svc', 'xgbost'], ['svc', 'lr'], ['xgboost', 'lr'], ['svc'], ['xgboost'],
                  ['lr']]

    def backtest(strat):
        cash = 1_000_000
        df_sell = pd.DataFrame()
        df_buy = pd.DataFrame()
        COMISSION = 0.0025
        STOP_LOSS = 0.05
        TAKE_PROFIT = 0.05
        positions = []
        closed_positions = []

        strategies_design(strat, data, data_validation, df_buy, df_sell)

        for row, row_buy, row_sell in zip(data_validation.iterrows(), df_buy.iterrows(), df_sell.iterrows()):
            # Close positions
            price = 3333 * row[1].Close
            for position in positions:
                j = positions.index(position)
                if position.is_active:
                    if (price <= position.stop_loss) & (position.order_type == 'LONG'):
                        # Close position, loss
                        cash += price * (1 - COMISSION)
                        position.is_active = False
                        position.sold_at = price
                        closed_positions.append(positions.pop(j))

                    elif (price >= position.take_profit) & (position.order_type == 'LONG'):
                        # Close position, profit
                        cash += price * (1 - COMISSION)
                        position.is_active = False
                        position.sold_at = price
                        closed_positions.append(positions.pop(j))

                    if (cash < 2 * (price - position.bought_at) + price) & (position.order_type == 'SHORT'):
                        cash -= price * (1 + COMISSION)
                        position.is_active = False
                        position.sold_at = price
                        closed_positions.append(positions.pop(j))

                    elif (price >= position.stop_loss) & (position.order_type == 'SHORT'):
                        # Close position, loss
                        cash -= price * (1 + COMISSION)
                        position.is_active = False
                        position.sold_at = price
                        closed_positions.append(positions.pop(j))

                    elif (price <= position.take_profit) & (position.order_type == 'SHORT'):
                        # Close position, profit
                        cash -= price * (1 + COMISSION)
                        position.is_active = False
                        position.sold_at = price
                        closed_positions.append(positions.pop(j))
            # buy
            if row_buy[1].sum() == len(df_buy.columns):
                if cash < 3333 * row[1].Close * (1 + COMISSION):
                    continue
                cash -= 3333 * row[1].Close * (1 + COMISSION)

                order = Order(timestamp=row[1].Timestamp,
                              bought_at=3333 * row[1].Close,
                              stop_loss=3333 * row[1].Close * (1 - STOP_LOSS),
                              take_profit=3333 * row[1].Close * (1 + TAKE_PROFIT),
                              order_type='LONG'
                              )

                positions.append(order)

            # Sell
            # close positons
            price = 3333 * row[1].Close

            # do we have money
            # Buy
            if row_sell[1].sum() == len(df_sell.columns):
                if cash < price * (1 + COMISSION):
                    continue
                cash += price * (1 - COMISSION)

                order = Order(timestamp=row[1].Timestamp,
                              bought_at=price,
                              stop_loss=price * (1 + STOP_LOSS),
                              take_profit=price * (1 - TAKE_PROFIT),
                              order_type='SHORT'
                              )
                positions.append(order)

            cash_still_open_positions = []

            for position in positions:
                if position.order_type == 'LONG' and position.is_active:
                    cash_still_open_positions.append(3333 * data_validation.Close.iloc[-1])
                if position.order_type == 'SHORT' and position.is_active:
                    cash_still_open_positions.append(-data_validation.Close.iloc[-1] * 1000)

            cash_values.append(cash)
            portfolio_values.append(sum(cash_still_open_positions) + cash)

        cash_still_open_positions2 = []
        for position in positions:
            if position.order_type == 'LONG' and position.is_active:
                cash_still_open_positions2.append(3333 * data_validation.Close.iloc[-1])
            if position.order_type == 'SHORT' and position.is_active:
                cash_still_open_positions2.append(-data_validation.Close.iloc[-1] * 1000)

        return sum(cash_still_open_positions2) + cash

    coun = 1
    for strat in strategies:
        portfolio = backtest(strat)
        df_trash = pd.DataFrame({'gain': [portfolio], 'strategy': [strat]})
        df_results = pd.concat([df_results, df_trash], ignore_index=True)
        print(f"Finished Backtest {coun}/7")
        coun += 1

    df_results.to_csv('resultados.csv')
    df_results.sort_values('gain', ascending=False, inplace=True)
    print(df_results.iloc[0])

    df_plots = pd.DataFrame({'Portfolio': portfolio_values, 'Cash': cash_values})
    df_plots.to_csv('port_cash_values.csv')
