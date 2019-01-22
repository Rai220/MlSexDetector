import csv
import re
import random
from random import randint

from random import shuffle

import numpy as np
from nltk.corpus import stopwords

letters = "@ABCDEFGHIJKLMNOPQRSTUVWXYZАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ0123456789.,!? -"
LETTERS_FOR_RANDOM = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ0123456789.,!?    -"
dictSize = len(letters)  # 26 letters + eos

SEX_DICT = {'unknown': 0, 'М': 1, 'Ж': 2}
SEX_DICT_REVERSE = {0: 'unknown', 1: 'М', 2: 'Ж'}


def sparse(n, size):
    out = [0.0] * size
    if int(n) >= size:
        print("{} {}".format(n, size))
    out[int(n)] = 1.0
    return out

def chartoindex(c):
    c = c.upper()
    if (c not in letters):
        print("Incorrect letter: " + c)
        return 0
    return letters.index(c)


def word2input(word, maxsize):
    word = word.upper()
    word = re.sub('[^0-9A-ZА-Я ]+', '', word)
    input = list(map(lambda c: sparse(chartoindex(c), dictSize), word.upper().replace(" ", "")))
    input += [sparse(dictSize - 1, dictSize)] * (maxsize - len(input))
    return list(input)

def loadFile(file):
    with open(file, encoding='utf-8') as fp:
        data = csv.reader(fp, delimiter=',')
        data = list(data)
        return data

def getPatch(data, count):
    print("Preparing dataSet...")
    max_input = 0
    sexOutput = list()
    input = list()
    stopWords = stopwords.words('russian')
    counter = 0

    shuffle(data)
    for row in data:
        if (counter > count):
            break
        if (row[3].strip() in SEX_DICT ):
            max_input = max(max_input, len(row[0]))

            r = randint(0, 10)
            r2 = randint(0, 1)
            if r < 7:
                if r == 0:
                    text = row[1].strip()
                elif r < 4:
                    if (r2 == 0):
                        text = (row[0] + " " + row[1] + " " + row[2]).strip()
                    else:
                        text = (row[1] + " " + row[0] + " " + row[2]).strip()
                elif r < 6:
                    if (r2 == 0):
                        text = (row[0] + " " + row[1]).strip()
                    else:
                        text = (row[1] + " " + row[0]).strip()
                else:
                    if (r2 == 0):
                        text = (row[1] + " " + row[2]).strip()
                    else:
                        text = (row[2] + " " + row[1]).strip()

                if len(text) < 50:
                    input.append(word2input(text, 50))
                    sexOutput.append(sparse(SEX_DICT[row[3].strip()], len(SEX_DICT)))
            else:
                if r == 7:
                    text = random.choice(stopWords)
                elif r == 8:
                    text = random.choice(stopWords) + " " + random.choice(stopWords)
                elif r == 9: # Random string
                    N = r = randint(2, 15)
                    text = ''.join(random.choices(LETTERS_FOR_RANDOM, k=N))
                else:
                    text = random.choice(stopWords) + " " + random.choice(stopWords) + " " + random.choice(stopWords)
                text = text.strip().upper()
                if len(text) < 50:
                    input.append(word2input(text, 50))
                    sexOutput.append(sparse(SEX_DICT["unknown"], len(SEX_DICT)))

            counter += 1
            if (counter % 10000 == 0):
                print(text)

    x = np.array(input)
    x = x.astype('float32')
    slices = [int(0.8 * len(x)), len(x)]
    x_train, x_test, _ = np.split(x, slices)
    action_output_train, action_output_test,            _ = np.split(sexOutput, slices)

    return [x_train, x_test,
            action_output_train, action_output_test]