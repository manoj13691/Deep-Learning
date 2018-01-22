import numpy as np

SEQ_LENGTH = 20
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


for i in range(0, len(data)/SEQ_LENGTH):
    X_sequence = data[i*SEQ_LENGTH:(i+1)*SEQ_LENGTH]
    X_sequence_ix = [char_to_ix[value] for value in X_sequence]
    input_sequence = np.zeros((SEQ_LENGTH, VOCAB_SIZE))
    for j in range(SEQ_LENGTH):
        input_sequence[j][X_sequence_ix[j]] = 1.
    X[i] = input_sequence

    y_sequence = data[i*SEQ_LENGTH+1:(i+1)*SEQ_LENGTH+1]
    y_sequence_ix = [char_to_ix[value] for value in y_sequence]
    target_sequence = np.zeros((SEQ_LENGTH, VOCAB_SIZE))
    for j in range(SEQ_LENGTH):
        target_sequence[j][y_sequence_ix[j]] = 1.
    y[i] = target_sequence
