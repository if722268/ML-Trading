


class IndicatorNotFoundError(Exception):
    pass


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


def y_generator(file):
    file['price_in_10_days'] = file['Close'].shift(-10)
    file.dropna(inplace=True)
    y_target = []
    for price_10, clos in zip(file['price_in_10_days'], file['Close']):
        if price_10 > clos * 1.02:
            y_target.append(1)
        elif price_10/clos-1 < -0.02:
            y_target.append(-1)
        else:
            y_target.append(0)

    file['target'] = y_target
    return file