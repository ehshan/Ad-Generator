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


# populate model vectors with word embedding data
def vectorize_words(sentences, train_input, train_output, word_to_index):
    print('Start-Time: ', time.ctime(time.time()))
    # populate vectors
    for i, sentence in enumerate(sentences):
        for t, word in enumerate(sentence[:-1]):
            train_input[i, t] = word_to_index(word)
            if t > 0:
                train_output[i, t - 1] = word_to_index(word)
    print('End-Time: ', time.ctime(time.time()))
    return train_input, train_output


# populate model vectors with word embedding data
def vectorize_embed(labels, sentences, encoder_input, decoder_input, decoder_target, word_to_index):
    print('Start-Time: ', time.ctime(time.time()))
    # populate labels
    for i, label in enumerate(labels):
        for t, word in enumerate(label[:-1]):
            encoder_input[i, t] = word_to_index(word)
    # populate sentences
    for i, sentence in enumerate(sentences):
        for t, word in enumerate(sentence[:-1]):
            decoder_input[i, t] = word_to_index(word)
            if t > 0:
                decoder_target[i, t - 1] = word_to_index(word)
    print('End-Time: ', time.ctime(time.time()))
    return encoder_input, decoder_input, decoder_target
