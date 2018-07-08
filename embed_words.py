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


# return dictionary lookup functions when passed an embedding model
def dictionary_lookups(model):
    # get model data
    embed_weights = model.wv.syn0
    vocab_size, embedding_size = embed_weights.shape

    # DICTIONARY/REVERSE DICTIONARY LOOKUPS

    # Get the index for word in vocab
    def word_to_index(word):
        return model.wv.vocab[word].index

    # Get index for similar word if not in vocab
    def similar_word_to_index(word):
        # most similar word
        dict_word = model.wv.most_similar(word)[0][0]
        return model.wv.vocab[dict_word].index

    # check if word  and apply correct index function
    def word_index(word):
        if word in model.wv.vocab:
            return word_to_index(word)
        else:
            # if no ngrams present return <ukn>
            try:
                return similar_word_to_index(word)
            except KeyError:
                return vocab_size - 1

    # lookup word for index with  coded special character for unknown
    def index_to_word(index):
        if index == vocab_size - 1:
            return '<UNK>'
        else:
            return model.wv.index2word[index]

    return word_index, index_to_word
