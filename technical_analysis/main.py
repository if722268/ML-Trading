from itertools import combinations
import ta

a = [ta.trend.ichimoku_a, ta.trend.SMAIndicator, ta.trend.MACD, ta.momentum.RSIIndicator, ta.momentum.ROCIndicator, ta.momentum.awesome_oscillator]

for j in range(len(a)):
    print([i for i in combinations(a, j)])
