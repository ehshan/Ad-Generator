import os
import time
import numpy as np

from clean_data import tokenize_dir, clean_tokens
from embed_words import train_word_model, dictionary_lookups

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
