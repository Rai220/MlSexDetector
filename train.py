import os, sys

from keras.engine import Input
from keras.engine import Model
from keras.layers import Dropout, Dense, Flatten, Conv1D, MaxPooling1D, Concatenate, LSTM, MaxPooling2D
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf

from utils import *

# Without LSTM: loss: 0.0070 - acc: 0.9987 - val_loss: 0.0103 - val_acc: 0.9982

with tf.device("/gpu:0"):
    letters = "@ABCDEFGHIJKLMNOPQRSTUVWXYZАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ0123456789.,!? -"
    dictSize = len(letters)  # 26 letters + eos


    main_input = Input(shape=(50, dictSize), dtype='float32', name='main_input')
    #conv = Conv1D(200, 5, strides=1, padding='same', dilation_rate=1, activation='relu',
    #              use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
    #              kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
    #              kernel_constraint=None, bias_constraint=None)(main_input)
    #
    #conv2 = Conv1D(200, 3, strides=1, padding='same', dilation_rate=1, activation='relu',
    #              use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
    #              kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
    #              kernel_constraint=None, bias_constraint=None)(main_input)

    lstm = LSTM(128)(main_input)


    #added = Concatenate()([conv, conv2])

    #p = MaxPooling1D(pool_size=4)(added)
    #f = Flatten()(p)

    #added2 = Concatenate()([f, lstm])
    added2 = lstm

    d = Dropout(0.2, noise_shape=None, seed=None)(added2)
    hidden = Dense(128, activation='relu')(d)
    d = Dropout(0.5, noise_shape=None, seed=None)(hidden)
    # hidden = Dense(1024, activation='reluовн')(d)
    # d = Dropout(0.5, noise_shape=None, seed=None)(hidden)

    sex = Dense(len(SEX_DICT), init='uniform', activation='softmax', name='sex')(d)
    model = Model(input=[main_input], output=[sex])


    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    data = loadFile("data.csv")
    for i in range(1, 20):
        x_train, x_test, out_train, out_test = getPatch(data, 70000)
        lr = 0.001 / (i / 10)
        print("Epoch: " + str(i) + " lr: " + str(lr))
        opt = Adam(lr=lr)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        model.fit(x_train, [out_train], nb_epoch=6, batch_size=5000,
                  validation_data=(x_test, [out_test]), verbose=2)

    out_dir = './output'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    K.set_learning_phase(0)
    with open(os.path.join(out_dir, 'model.json'), "w") as json_file:
        json_file.write(model.to_json())
    model.save_weights(os.path.join(out_dir, 'model.h5'))

    print('ok')