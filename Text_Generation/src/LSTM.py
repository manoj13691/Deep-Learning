
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed





SEQ_LENGTH = 20
HIDDEN_DIM = 3
LAYER_NUM = 3
DATA_FILE = "/home/manoj/Deep-Learning/Text_Generation/data/shakespeare.txt"


data = open(DATA_FILE, 'r').read() 
chars = list(set(data))
VOCAB_SIZE = len(chars)

ix_to_char = {ix:char for ix, char in enumerate(chars)}
char_to_ix = {char:ix for ix, char in enumerate(chars)}
print('Data length: {} characters'.format(len(data)))
print('Vocabulary size: {} characters'.format(VOCAB_SIZE))


X = np.zeros((len(data)/SEQ_LENGTH, SEQ_LENGTH, VOCAB_SIZE))
y = np.zeros((len(data)/SEQ_LENGTH, SEQ_LENGTH, VOCAB_SIZE))

end = 0
start = 0
for i in range(0, len(data)/SEQ_LENGTH):
    start = end
    end = start + SEQ_LENGTH
    X_sequence = data[start : end]
    X_sequence_ix = [char_to_ix[char] for char in X_sequence]
    input_sequence = np.zeros((SEQ_LENGTH, VOCAB_SIZE))
    for j in range(SEQ_LENGTH):
        input_sequence[j][X_sequence_ix[j]] = 1.
    X[i] = input_sequence

    y_sequence = data[start+1 : end+1]
    y_sequence_ix = [char_to_ix[value] for value in y_sequence]
    target_sequence = np.zeros((SEQ_LENGTH, VOCAB_SIZE))
    for j in range(SEQ_LENGTH):
        target_sequence[j][y_sequence_ix[j]] = 1.
    y[i] = target_sequence





model = Sequential()
model.add(LSTM(HIDDEN_DIM, input_shape=(None, VOCAB_SIZE), return_sequences=True))
for i in range(LAYER_NUM - 1):
    model.add(LSTM(HIDDEN_DIM, return_sequences=True))
model.add(TimeDistributed(Dense(VOCAB_SIZE)))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer="adam")
model.summary()

nb_epoch = 1
while nb_epoch:
    print("\n")
    print("Epoch=" + str(nb_epoch))
    model.fit(X, y, batch_size=32, verbose=1, nb_epoch=1)
    nb_epoch -= 1
    #generate_text(model, GENERATE_LENGTH)
    #if nb_epoch % 10 == 0:
        #model.save_weights('checkpoint_{}_epoch_{}.hdf5'.format(HIDDEN_DIM, nb_epoch))


def generate_text(model, length):
    ix = [np.random.randint(VOCAB_SIZE)]
    y_char = [ix_to_char[ix[-1]]]
    X = np.zeros((1, length, VOCAB_SIZE))
    for i in range(length):
        X[0, i, :][ix[-1]] = 1
        print(ix_to_char[ix[-1]], end="")
        ix = np.argmax(model.predict(X[:, :i+1, :])[0], 1)
        y_char.append(ix_to_char[ix[-1]])
    return ('').join(y_char)


print (generate_text(model, 100))
