import os.path
import time

from clean_data import tokenize_dir, clean_tokens

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
