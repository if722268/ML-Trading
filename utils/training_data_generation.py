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


def x_y_generator(file):
    data = y_generator(file)

    data['rend'] = data['Close'].pct_change()

    short_sma = ta.trend.SMAIndicator(data.Close, window=5)
    long_sma = ta.trend.SMAIndicator(data.Close, window=15)
    data['short_sma'] = short_sma.sma_indicator()
    data['long_sma'] = long_sma.sma_indicator()
    data['rsi'] = ta.momentum.RSIIndicator(data.Close).rsi()

    data.dropna(inplace=True)
    return data
