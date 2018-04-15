
# coding: utf-8

# In[184]:

import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout
from keras.models import Model
from keras import backend as K    
K.set_image_dim_ordering('th') 


# In[156]:

corpus = pd.read_pickle('../data/MR.pkl')
corpus= corpus.sample(frac=1)
sentences, labels = list(corpus.sentence), list(corpus.label)
print len(sentences)


# In[157]:

corpus.head(5)


# In[158]:

#Increasing the value will increase sequence length in many sentences. Captures more words
TOP_N_WORDS = 5000


# In[159]:

tokenizer = Tokenizer(nb_words=TOP_N_WORDS)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


# In[160]:

#Tokenizer.word_index shows the word and its index in the dictionary
#These indices are fed as a sequence
print (tokenizer.word_index)


# In[161]:

#This will show how the first sentence has been converted to numeric sequence
print sequences[0]
#This will show the first sentence itself
print "Sentence: "+sentences[0]
#This will loop through every word of the first sentence and see which word is not added in the sequence.
print "Comment: Words not added from the first sentence along with their ID`s"
words = sentences[0].split(" ")
for i in words:
    if i in word_index:
        if word_index[i] not in sequences[0]:
            print i, word_index[i]


# In[162]:

max_sequence_length = 0
min_sequence_length = -1
j = -1
for i in sequences:
    seq_len = len(i)
    
    if min_sequence_length == -1:
        min_sequence_length = seq_len
        
    if seq_len > max_sequence_length:
        max_sequence_length = seq_len
    
    if seq_len < min_sequence_length and min_sequence_length!=-1:
        min_sequence_length = seq_len
        j = i
        
print min_sequence_length
print max_sequence_length


# In[163]:

data = pad_sequences(sequences, maxlen=max_sequence_length)
import numpy as np
data_labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', data_labels.shape)


# In[164]:

data[0]
data_labels[0]


# In[165]:

import os
GLOVE_DIR = "/home/manoj/Downloads/glove.6B/"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print "Loaded "+str(len(embeddings_index))+" word embeddings from GLOVE"


# In[166]:

EMBEDDING_DIM = len(embeddings_index["the"])


# In[167]:

#+1 for bias.
#len(word_index) because we have so many unique tokens after all the filtering.

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be random numbers.
        embedding_matrix[i] = embedding_vector


# In[168]:

embedding_matrix.shape


# In[225]:

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=max_sequence_length,
                            trainable=True)


# In[234]:

sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(2, activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])


# In[235]:

model.summary()


# In[241]:

model.fit(data, data_labels,epochs=2, batch_size=128)


# In[242]:

model.fit(data, data_labels,epochs=2, batch_size=128, validation_split= 0.1)


# In[ ]:



