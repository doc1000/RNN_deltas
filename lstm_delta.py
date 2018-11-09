import os
import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import pg
import pandas as pd
import matplotlib.pyplot as plt
#import ipdb


qry = '''select * from
(select tradedate_spp,member, max(case when hourending_spp=1 then trunc(speed80,1) end) as g1
,max(case when hourending_spp=7 then trunc(speed80,1) end) as g7
,max(case when hourending_spp=13 then trunc(speed80,1) end) as g13
,max(case when hourending_spp=19 then trunc(speed80,1) end) as g19
from gefs.v_gefs_2_day
where grid_id = 84503
and tradedate_spp > '1/1/18'
and model_ts_utc = (tradedate_spp - interval '1 day' + interval '12 hour')
group by tradedate_spp,member
) gf
inner join
(select tradedate_spp,max(case when hourending_spp=1 then trunc(speed80,1) end) as HE1,max(case when hourending_spp=2 then trunc(speed80,1) end) as HE2,max(case when hourending_spp=3 then trunc(speed80,1) end) as HE3,max(case when hourending_spp=4 then trunc(speed80,1) end) as HE4,max(case when hourending_spp=5 then trunc(speed80,1) end) as HE5,max(case when hourending_spp=6 then trunc(speed80,1) end) as HE6,max(case when hourending_spp=7 then trunc(speed80,1) end) as HE7,max(case when hourending_spp=8 then trunc(speed80,1) end) as HE8,max(case when hourending_spp=9 then trunc(speed80,1) end) as HE9,max(case when hourending_spp=10 then trunc(speed80,1) end) as HE10,max(case when hourending_spp=11 then trunc(speed80,1) end) as HE11,max(case when hourending_spp=12 then trunc(speed80,1) end) as HE12,max(case when hourending_spp=13 then trunc(speed80,1) end) as HE13,max(case when hourending_spp=14 then trunc(speed80,1) end) as HE14,max(case when hourending_spp=15 then trunc(speed80,1) end) as HE15,max(case when hourending_spp=16 then trunc(speed80,1) end) as HE16,max(case when hourending_spp=17 then trunc(speed80,1) end) as HE17,max(case when hourending_spp=18 then trunc(speed80,1) end) as HE18,max(case when hourending_spp=19 then trunc(speed80,1) end) as HE19,max(case when hourending_spp=20 then trunc(speed80,1) end) as HE20,max(case when hourending_spp=21 then trunc(speed80,1) end) as HE21,max(case when hourending_spp=22 then trunc(speed80,1) end) as HE22,max(case when hourending_spp=23 then trunc(speed80,1) end) as HE23,max(case when hourending_spp=24 then trunc(speed80,1) end) as HE24
from hrrr.v_realtime
where grid_id = 876991
and tradedate_spp > '1/1/18'
group by tradedate_spp
) hr
on gf.tradedate_spp=hr.tradedate_spp
'''


def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_shape=(layers[1], layers[0]),
        output_dim=layers[1],
        return_sequences=True))
    model.add(Activation("tanh"))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    #model.add(Activation("tanh"))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("> Compilation Time : ", time.time() - start)
    return model

def split_data(X,y,seq_len):
    '''simple standarization by div by max values'''
    X = X.T.fillna(method='ffill').T
    y = y.T.fillna(method='ffill').T
    X = np.array(X)
    y = np.array(y)
    X_max = np.max(X)
    y_max = np.max(y)
    X = X/X_max
    y = y/y_max

    sequence_length = seq_len + 1-1
    result = []
    for index in range(len(X) - sequence_length):
        result.append(X[index: index + sequence_length])

#    if normalise_window:
#        result = normalise_windows(result)

    result = np.array(result)

    row = round(0.9 * result.shape[0])
    X_train = result[:int(row), :]
    y_train = y[:int(row), :]
    X_test = result[int(row):, :]
    y_test = y[int(row)+seq_len:, :]

    X_train, y_train = shuffle(X_train, y_train)

    #X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    #X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return [X_train, y_train, X_test, y_test,X_max,y_max]

def run_model():
        cn = pg.Connection()
        cn.reset()
        data = cn.query(qry)
        cols = cn.column_names
        data = pd.DataFrame(data,columns =cols)

        X = data[['g1','g7','g13','g19']].astype('float')
        y = data[['he1','he2','he3','he4','he5','he6','he7','he8','he9','he10','he11','he12','he13','he14','he15','he16','he17','he18','he19','he20','he21','he22','he23','he24']].astype('float')
        #
        # X = np.random.randint(25,size=(100,4))
        # y = np.random.randint(25,size=(100,24))
        #
        epochs  = 20 # suggest 100 for sine wave, 10 for stock
        seq_len = 2

        X_train, y_train, X_test, y_test,X_max,y_max = split_data(X,y,seq_len)
        # '''layers shoule equate to #features per time period (day), seq_len,2nd LSTM neurons, outputs per day'''
        layers = [X.shape[1],seq_len,100,y.shape[1]]

        model = build_model(layers) # 1 input layer, layer 1 has seq_len neurons, layer 2 has 100 neurons, 1 output

        model.fit(
            X_train,
            y_train,
            batch_size=16,
            nb_epoch=epochs,
            validation_split=0.05)

        predicted = model.predict(X_test)
        print(np.sqrt(mean_squared_error(y_test,predicted)))
        predicted_reconvert = predicted*y_max

def member_plot(member,df):
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        if member == -1:
            members = [x for x in range(22)]
        else: members = [member]
        for member in members:
            df2 = df[df['member']==member]
            X = df2[['g1','g7','g13','g19']].astype('float')
            y = df2[['he1','he2','he3','he4','he5','he6','he7','he8','he9','he10','he11','he12','he13','he14','he15','he16','he17','he18','he19','he20','he21','he22','he23','he24']].astype('float')
            X = np.array(X)
            y = np.array(y)
            xy = [x for x in range(y.ravel().shape[0])]
            xp = [x*6 for x in range(X.ravel().shape[0])]
            x_interp = np.interp(xy, xp, X.ravel())
            y_rav = y.ravel()
            ax.plot(x_interp)
            ax.plot(y_rav)
            ax2.plot(y_rav-x_interp)




if __name__ == '__main__':
        cn = pg.Connection()
        cn.reset()
        data = cn.query(qry)
        cols = cn.column_names
        data = pd.DataFrame(data,columns =cols)

        X = data[['g1','g7','g13','g19']].astype('float')
        y = data[['he1','he2','he3','he4','he5','he6','he7','he8','he9','he10','he11','he12','he13','he14','he15','he16','he17','he18','he19','he20','he21','he22','he23','he24']].astype('float')
