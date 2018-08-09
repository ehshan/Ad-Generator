import os.path
import time

import keras.backend as K
from keras.models import Sequential
from keras.layers import Input, Embedding, LSTM, Dropout, Dense
from keras.optimizers import RMSprop

from clean_data import tokenize_dir, clean_tokens
from label_corpus import tag_corpus
from embed_words import train_word_model

# self path
path = os.getcwd()

# define data file and file extension
data_path = os.path.abspath(os.path.join(os.getcwd(), '../../Data/Twitter-Data/Clean'))
extension = 'csv'

# for saving
version_name = 'conditioned_train'

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

# add start and end tags to each sentence
corpus = tag_corpus(sentences)

# GENERATE EMBEDDINGS
# ---------------

print('\nCreating word embeddings...')
# train and save the embedding model
word_model = train_word_model(corpus, 'word_model')

# get the initial model weight
embed_weights = word_model.wv.syn0
# get the vocab size and embedding shape for model
vocab_size, embedding_size = embed_weights.shape

# DEFINE MODEL LAYERS
# ----------------------

# ENCODER
# -------
# Define the encoder layers
encoder_inputs = Input(shape=(None,), name='label_input')
encoder_embed = Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[embed_weights], trainable=False,
                          name='encoder_embedding')
encoder_lstm = LSTM(embedding_size, return_state=True, name='encoder_lstm')


# DECODER
# -------
# Define the decoder layers
decoder_inputs = Input(shape=(None,), name='sentence_input')
decoder_embed = Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[embed_weights], trainable=False,
                          name='decoder_embedding')
decoder_lstm = LSTM(embedding_size, return_sequences=True, return_state=True, name='decoder_lstm')
dropout = Dropout(0.2)
decoder_dense = Dense(vocab_size, activation='softmax')


# CONNECT LAYERS
# --------------
# Connect the encoder layers
encoder_embedded = encoder_embed(encoder_inputs)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedded)
encoder_states = [state_h, state_c]

# Connect the decoder layers
decoder_embedded = decoder_embed(decoder_inputs)
decoder_lstm_outputs, _, _ = decoder_lstm(decoder_embedded, initial_state=encoder_states)
decoder_dropout = dropout(decoder_lstm_outputs)
decoder_outputs = decoder_dense(decoder_dropout)

# Define the training model
conditioned_model = Sequential([encoder_inputs, decoder_inputs], decoder_outputs)

# Define optimisation function
rms_prop = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)


# Define custom evaluation metrics

def perplexity(y_true, y_pred):
    cross_entropy = K.categorical_crossentropy(y_true, y_pred)
    perplexity = K.pow(2.0, cross_entropy)
    return perplexity


def cross_entropy(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)


# compile the training model
conditioned_model.compile(optimizer=rms_prop, loss='categorical_crossentropy',
                          metrics=['accuracy', cross_entropy, perplexity])
