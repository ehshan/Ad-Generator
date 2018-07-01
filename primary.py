import os
import time

from clean_data import tokenize_dir
from embed_words import train_word_model

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

print('\nTRAINING CORPUS: \n' + corpus)


# GENERATE EMBEDDINGS
# ---------------

# train and save the embedding model
word_model = train_word_model(corpus, 'word_model')
