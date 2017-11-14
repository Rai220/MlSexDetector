from keras import backend as K
from keras.models import model_from_json

from utils import *

import os

model_dir = './output'
K.set_learning_phase(0)
with open(os.path.join(model_dir, 'model.json'), 'r') as fp:
    model = model_from_json(fp.read())

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
model.load_weights(os.path.join(model_dir, 'model.h5'))

def format_out(out, dict):
    stringify = np.vectorize(lambda x: '{}: {:0.2f}'.format(dict[x], out[x]))
    return ', '.join(stringify(np.argsort(-out)))
    return ""

def predict(m, word):
    out = m.predict(np.array([word2input(word, 50)]))
    indMax = np.argmax(out[0])
    print('Sex: ', SEX_DICT_REVERSE[indMax])
    print(out)

while True:
    name = str(input("Input name: "))
    predict(model, name)