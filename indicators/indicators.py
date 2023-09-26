import pandas as pd
import ta



class TrendIndicator:
    def __init__(self, indicator, window_short=None, window_long=None):
        self.indicator = indicator
        self.window_short = window_short
        self.window_long = window_long
        self.df = pd.DataFrame()

    def generate_signals(self, data):
        if self.indicator == 'MACD':
            short_macd = ta.trend.MACD(data.Close)
            long_macd = ta.trend.MACD(data.Close)
            self.df['short_macd'] = long_macd.macd_signal()
            self.df['long_macd'] = short_macd.macd()
            return self.df.short_macd < self.df.long_macd, self.df.short_macd > self.df.long_macd #buy trade signal

        elif self.indicator == 'SMA':
            short_sma = ta.trend.SMAIndicator(data.Close, window=self.window_short)
            long_sma = ta.trend.SMAIndicator(data.Close, window=self.window_long)
            self.df['short_sma'] = short_sma.sma_indicator()
            self.df['long_sma'] = long_sma.sma_indicator()
            return data.short_sma < data.long_sma, data.short_sma > data.long_sma

        elif self.indicator == 'RSII':
            rsii = ta.momentum.RSIIndicator(data.Close)
            return rsii.rsi() > 70, rsii.rsi() < 30

        elif self.indicator == 'ichimoku':
            ...
        elif self.indicator == 'ROC':
            ...
        elif self.indicator == 'awesome_oscillator':
            ...
        else:
            raise IndicatorNotFoundError(f"Indicator '{self.indicator}' not found")


