import os

from keras.engine import Input
from keras.engine import Model
from keras.layers import Dropout, Dense, Flatten, Conv1D, MaxPooling1D
from keras.optimizers import Adam

from utils import *

letters = "@ABCDEFGHIJKLMNOPQRSTUVWXYZАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ0123456789.,!? -"
dictSize = len(letters)  # 26 letters + eos


main_input = Input(shape=(50, dictSize), dtype='float32', name='main_input')
conv = Conv1D(200, 5, strides=1, padding='same', dilation_rate=1, activation='relu',
              use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
              kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
              kernel_constraint=None, bias_constraint=None)(main_input)

p = MaxPooling1D(pool_size=4)(conv)
f = Flatten()(p)
d = Dropout(0.2, noise_shape=None, seed=None)(f)
hidden = Dense(1024, activation='relu')(d)
d = Dropout(0.5, noise_shape=None, seed=None)(hidden)
hidden = Dense(1024, activation='relu')(d)
d = Dropout(0.5, noise_shape=None, seed=None)(hidden)

sex = Dense(len(SEX_DICT), init='uniform', activation='softmax', name='sex')(d)
model = Model(input=[main_input], output=[sex])


model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

data = loadFile("dataMini.csv")
for i in range(1, 10):
    x_train, x_test, out_train, out_test = getPatch(data, 70000)
    lr = 0.001 / (i / 10)
    print("Epoch: " + str(i) + " lr: " + str(lr))
    opt = Adam(lr=lr)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.fit(x_train, [out_train], nb_epoch=3, batch_size=10000,
              validation_data=(x_test, [out_test]), verbose=2)

out_dir = './output'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

with open(os.path.join(out_dir, 'model.json'), "w") as json_file:
    json_file.write(model.to_json())
model.save_weights(os.path.join(out_dir, 'model.h5'))

print('ok')