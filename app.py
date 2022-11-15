import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date,datetime,timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout , LSTM
from keras.models import Sequential
import time
from datetime import datetime,date,timedelta
from flask import Flask, render_template, request, url_for
import pickle
import numpy as np

app = Flask(__name__)
IMG_FOLDER = os.path.join('static', 'IMG')

app.config['UPLOAD_FOLDER'] = IMG_FOLDER

@app.route('/')
def index():
    Flask_Logo = os.path.join(app.config['UPLOAD_FOLDER'], 'nse.png')
    # Flask_Logo2 = os.path.join(app.config['UPLOAD_FOLDER'], 'bg.jpg')
    return render_template('index.html', user_image=Flask_Logo)

@app.route('/about')
def about():
   pass

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    company = str(request.form.get("company_name"))
    start_date = str(request.form.get('start_date'))
    end_date = str(request.form.get('end_date'))
    total_stocks = int(request.form.get("no_of_stock"))
    sell_date_price=0
    buy_date_price=0
    sell_year = int(end_date.split('-')[0])
    sell_month = int(end_date.split('-')[1])
    sell_day = int(end_date.split('-')[2])
    our_end_date = date(2022, 3, 31)
    selling_date = date(sell_year, sell_month, sell_day)  
    print(selling_date)

    buy_year = int(start_date.split('-')[0])   
    buy_month = int(start_date.split('-')[1])
    buy_day = int(start_date.split('-')[2])
    Buying_date = date(buy_year, buy_month, buy_day)
    print(Buying_date)

    company = company + '.NS'
    quantity = int(total_stocks)

    df = yf.download(company, '2018-01-01', '2022-03-31')
    data = df.sort_index(ascending=True, axis=0)
    dat = data.reset_index()
    new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])
    for i in range(0, len(data)):
        new_data['Date'][i] = dat['Date'][i]
        new_data['Close'][i] = dat['Close'][i]

    # setting index
    new_data.index = new_data.Date
    new_data.drop('Date', axis=1, inplace=True)

    # creating train and test sets
    dataset = new_data.values
    print(df.index[-1].date())
    train = dataset[0:int(0.7 * len(df)), :]
    valid = dataset[int(0.7 * len(df)):, :]

    # converting dataset into x_train and y_train
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    x_train, y_train = [], []
    for i in range(100, len(train)):
        x_train.append(scaled_data[i - 100:i, 0])
        y_train.append(scaled_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    valid = scaled_data[int(0.7 * len(df)):, :]

    model = Sequential()
    model.add(LSTM(units=50, activation='relu', return_sequences=True,
                   input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units=60, activation='relu', return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(units=120, activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=50)

    for i in df.index:
        if selling_date == i.date():
            sell_date_price = df.loc[i]['Close']
            print(sell_date_price)
            break
    if sell_date_price == 0:
        while (selling_date - our_end_date).days != 0:
            sample = scaled_data[scaled_data.shape[0] - 100:scaled_data.shape[0]]
            sample = sample.reshape(1, 100, 1)
            predicted = model.predict(sample)
            scaled_data = np.append(scaled_data, [predicted[0, 0]])
            selling_date -= timedelta(days=1)
        sell_date_price = scaler.inverse_transform(np.array(scaled_data[-1]).reshape(-1, 1))      

    df2 = yf.download(company, '2018-01-01', '2022-3-31')
    data2 = df2.sort_index(ascending=True, axis=0)
    open_data = data2.reset_index()
    open_new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Open'])
    for i in range(0, len(data)):
        open_new_data['Date'][i] = open_data['Date'][i]
        open_new_data['Open'][i] = open_data['Open'][i]

    # setting index
    open_new_data.index = open_new_data.Date
    open_new_data.drop('Date', axis=1, inplace=True)

    # creating train and test sets
    open_dataset = open_new_data.values

    train_open = open_dataset[0:int(0.7 * len(df2)), :]
    valid_open = open_dataset[int(0.7 * len(df2)):, :]

    # converting dataset into x_train and y_train
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data_open = scaler.fit_transform(dataset)

    open_x_train, open_y_train = [], []
    for i in range(100, len(train_open)):
        open_x_train.append(scaled_data_open[i - 100:i, 0])
        open_y_train.append(scaled_data_open[i, 0])
    open_x_train, open_y_train = np.array(open_x_train), np.array(open_y_train)

    open_x_train = np.reshape(open_x_train, (open_x_train.shape[0], open_x_train.shape[1], 1))

    valid_open = scaled_data_open[int(0.7 * len(df2)):, :]

    open_model = Sequential()
    open_model.add(LSTM(units=50, activation='relu', return_sequences=True,
                        input_shape=(open_x_train.shape[1], 1)))
    open_model.add(Dropout(0.2))

    open_model.add(LSTM(units=60, activation='relu', return_sequences=True))
    open_model.add(Dropout(0.3))

    open_model.add(LSTM(units=120, activation='relu'))
    open_model.add(Dropout(0.4))

    open_model.add(Dense(units=1))

    open_model.compile(optimizer='adam', loss='mean_squared_error')
    open_model.fit(open_x_train, open_y_train, epochs=50)

    for i in df2.index:
        if Buying_date == i.date():
            buy_date_price = df2.loc[i]['Open']
    if buy_date_price == 0:
        while (Buying_date - our_end_date).days != 0:
            sample = scaled_data_open[scaled_data_open.shape[0] - 100:scaled_data_open.shape[0]]
            sample = sample.reshape(1, 100, 1)
            predicted = open_model.predict(sample)
            scaled_data_open = np.append(scaled_data_open, [predicted[0, 0]])
            Buying_date -= timedelta(days=1)
        buy_date_price = scaler.inverse_transform(np.array(scaled_data_open[-1]).reshape(-1, 1))    

    Profit_loss = (sell_date_price - buy_date_price) * int(quantity)
    result=Profit_loss
    return render_template('index.html',result1=result)

if __name__== '__main__':
    app.run(debug=True)
