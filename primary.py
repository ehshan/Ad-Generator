import os
import time
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Input, LSTM, Embedding
from keras.optimizers import RMSprop

from clean_data import tokenize_dir, clean_tokens
from embed_words import train_word_model, dictionary_lookups, vectorize_words

# working directory
path = os.getcwd()

# define data file and file extension
data_path = os.path.abspath(os.path.join(os.getcwd(), '../../Data/Twitter-Data/Clean'))
extension = 'csv'

# START
# ----

# LOAD DATA
# --

print('\nLoading data...')

print('Start-Time: ', time.ctime(time.time()))
corpus = tokenize_dir(data_path, extension)
print('End-Time: ', time.ctime(time.time()))

# clean tokenize corpus
sentences, max_sentence, max_sentence_len = clean_tokens(corpus)

print("max: %d " % max_sentence_len)

print('Num sentences in original corpus:', len(corpus))
print('Num sentences for model:', len(sentences))

# print('\nTRAINING CORPUS: \n' + corpus)


# GENERATE EMBEDDINGS
# ---------------

print('\nCreating word embeddings...')
# train and save the embedding model
word_model = train_word_model(corpus, 'word_model')

# get the initial model weight
embed_weights = word_model.wv.syn0
# get the vocab size and embedding shape for model
vocab_size, embedding_size = embed_weights.shape

# get the dictionary lookup functions
word_to_index, index_to_word = dictionary_lookups(word_model)

# VECTORIZE WORDS
# ----------------

print('\nVectorizing words...')
# define the shape of input & output matrices
# input shape (no sentences, max-sentence-size)
train_input = np.zeros([len(sentences), max_sentence_len], dtype=np.int32)

# output shape (no sentences)
train_output = np.zeros([len(sentences)], dtype=np.int32)

# populate model vectors with word embedding data
train_input, train_output = vectorize_words(sentences, train_input, train_output, index_to_word)

print('\ntrain_input shape:', train_input.shape)
print('train_output shape:', train_output.shape)

# MODEL SETUP
# ------------------
print('\nConstructing Model...')

# define the model layers
model_input = Input(shape=(None,))
model_embed = Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[embed_weights])
model_lstm = LSTM(units=embedding_size, return_sequences=True, return_state=False)
model_dense = Dense(units=vocab_size)
model_activation = Activation('softmax')

# Connect layers
embedded = model_embed(model_input)
lstm_output = model_lstm(embedded)
dense_output = model_dense(lstm_output)
model_output = model_activation(dense_output)

# Define the model
primary_model = Sequential(model_input, model_output)

# Define optimizer
rms_prop = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

# Compile model
primary_model.compile(
    optimizer=rms_prop,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

# print summary of model layers
print(primary_model.summary())


# TRAINING SETUP
# --------------
print("\nVocab size: %d" % vocab_size)
print("Embedding size: %d" % embedding_size)

batch_size = 128
epochs = 50
validation_split = 0.2
print("\nTraining in batches of: %d" % batch_size)
print("Training epochs: %d" % epochs)