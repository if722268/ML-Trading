import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('port_cash_values.csv').loc[67545:81054]
data.reset_index(inplace=True)
data['Timestamp'] = data.index

plt.plot(data.Timestamp, data.Cash, label='Cash')
plt.plot(data.Timestamp, data.Portfolio, label='Portfolio value')
plt.legend()
plt.title('USD over time')
plt.show()
