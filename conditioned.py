import os.path
import time

from keras.layers import Input, Embedding, LSTM

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
