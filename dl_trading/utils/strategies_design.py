from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler


def strategies_design(strate, train_data, validate_data, df_buy, df_sell):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train_data.drop(['target', 'price_in_10_days'], axis=1))
    X_test = scaler.fit_transform(validate_data.drop(['target', 'price_in_10_days'], axis=1))
    y_test = validate_data['target']
    y_train = train_data['target']


    if 'mlp':
        mlp = Sequential()

        mlp.add(Dense(32, activation='relu'))
        # mlp.add(Dropout(0.2))
        mlp.add(Dense(16, activation='relu'))
        # mlp.add(Dropout(0.2))
        mlp.add(Dense(8, activation='relu'))
        mlp.add(Dense(4, activation='relu'))
        mlp.add(Dense(3, activation='softmax'))

        mlp.compile(optimizer='adam',
                    loss='SparseCategoricalCrossentropy',
                    metrics=['accuracy'])

        mlp.fit(X_train, y_train, validation_data=(X_train, y_train), epochs=10, batch_size=32, verbose=0)
        mlp_predictions = mlp.predict(X_test)
        mlp_predictions = mlp_predictions.argmax(axis=1)
        print(confusion_matrix(y_test, mlp_predictions))
        df_buy['mlp_buy_trade_signal'] = [True if cat == 2 else False for cat in mlp_predictions]
        df_sell['mlp_sell_trade_signal'] = [True if cat == 0 else False for cat in mlp_predictions]

    if 'rnn' in strate:
        rnn = Sequential()
        rnn.add(LSTM(units=256, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)))
        rnn.add(Dropout(0.2))
        rnn.add(LSTM(units=64, activation='relu', return_sequences=False, input_shape=(X_train.shape[1], 1)))
        rnn.add(Dense(10))
        rnn.add(Dense(3, activation='sigmoid'))

        rnn.compile(optimizer='adam',
                    loss='SparseCategoricalCrossentropy',
                    metrics=['accuracy'])

        rnn.fit(X_train, y_train, validation_data=(X_train, y_train), epochs=10, batch_size=32, verbose=0)
        rnn_predictions = rnn.predict(validate_data.drop(X_test)).argmax(axis=1)
        df_buy['rl_buy_trade_signal'] = [True if cat == 2 else False for cat in rnn_predictions]
        df_sell['rl_sell_trade_signal'] = [True if cat == 0 else False for cat in rnn_predictions]
