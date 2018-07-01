import os.path
import time

from gensim.models import FastText

path = os.getcwd()


# trains a faxtText model with training corpus
# Limits the vocab to 10000 words
# Ignores word that occur less than twice
def train_word_model(sentences, name):
    print('\nTraining fastText model...')

    print('Start-Time: ', time.ctime(time.time()))
    word_model = FastText(sentences, size=256, min_count=2, window=5, iter=100, workers=4, max_vocab_size=10000)
    print('End-Time: ', time.ctime(time.time()))

    print("\nSaving fastText model to..." + path + "/Embeddings/")
    word_model.save(path + "/Embeddings/" + name + ".model")
    return word_model
