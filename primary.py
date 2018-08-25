import os
import time
import numpy as np
import csv
import pickle
import json
import keras.backend as K

from keras.models import Sequential
from keras.layers import Dense, Activation, Input, LSTM, Embedding, Dropout, TimeDistributed
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback, CSVLogger, History, ModelCheckpoint

from clean_data import tokenize_dir, clean_tokens
from embed_words import train_word_model, dictionary_lookups, vectorize_words

# working directory
path = os.getcwd()

# define data file and file extension
data_path = os.path.abspath(os.path.join(os.getcwd(), '../../Data/Twitter-Data/Clean'))
extension = 'csv'

# for saving
version_name = 'primary_train'

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

# output shape (no sentences, max-sentence-size, 1)
train_output = np.zeros([len(sentences), max_sentence_len, 1], dtype=np.int32)

# populate model vectors with word embedding data
train_input, train_output = vectorize_words(sentences, train_input, train_output, word_to_index)

print('\ntrain_input shape:', train_input.shape)
print('train_output shape:', train_output.shape)

# MODEL SETUP
# ------------------
print('\nConstructing Model...')

# define the model layers
model_input = Input(shape=(None,))
model_embed = Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[embed_weights])
model_lstm_1 = LSTM(units=embedding_size, return_sequences=True, return_state=False)
model_dropout_1 = Dropout(0.2)
model_lstm_2 = LSTM(units=embedding_size, return_sequences=False, return_state=False)
model_dropout_2 = Dropout(0.2)
model_dense = TimeDistributed(Dense(units=vocab_size))
model_activation = Activation('softmax')
# Connect layers
embedded = model_embed(model_input)
lstm_1_output = model_lstm_1(embedded)
dropout_1_output = model_dropout_1(lstm_1_output)
lstm_2_output = model_lstm_2(dropout_1_output)
dropout_2_output = model_dropout_2(lstm_2_output)
dense_output = model_dense(dropout_2_output)
model_output = model_activation(dense_output)


# Define the model
primary_model = Sequential(model_input, model_output)

# Define optimizer
rms_prop = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)


# Define custom evaluation metrics
def perplexity(y_true, y_pred):
    cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
    perplexity = K.pow(2.0, cross_entropy)
    # perplexity = 2 ** cross_entropy
    return perplexity


def crossentropy(y_true, y_pred):
    return K.sparse_categorical_crossentropy(y_true, y_pred)


# Compile model
primary_model.compile(
    optimizer=rms_prop,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', crossentropy, perplexity])

# print summary of model layers
print(primary_model.summary())

# TRAINING SETUP
# --------------
print("\nVocab size: %d" % vocab_size)
print("Embedding size: %d" % embedding_size)

batch_size = 32
epochs = 25
validation_split = 0.2
print("\nTraining in batches of: %d" % batch_size)
print("Training epochs: %d" % epochs)

# start point for generated text
start_words = ['the', 'there', 'from', 'have', 'can',
               'engine', 'body', 'speed', 'elegance', 'safety',
               'fun', 'love', 'excite', 'joy', 'curious', ]


# apply temperature to each model sample
def temp_sample(predictions, temperature=1.0):
    # value 0 return argmax sampling
    if temperature <= 0:
        return np.argmax(predictions)
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions) / temperature
    exp_predictions = np.exp(predictions)
    predictions = exp_predictions / np.sum(exp_predictions)
    probability = np.random.multinomial(1, predictions, 1)
    return np.argmax(probability)


# generate sentence one word at a time - limiting to 10 words
def generate_next_word(text, temp, sentence_length=10):
    word_indices = [word_to_index(word) for word in text.lower().split()]
    for n in range(sentence_length):
        prediction = primary_model.predict(x=np.array(word_indices))
        index = temp_sample(prediction[0, -1, :], temperature=temp)
        word_indices.append(index)
    return ' '.join(index_to_word(index) for index in word_indices)


# writes prediction to file for each epoch
def on_epoch_end(epoch, _):
    # declare csv objects for both sampling styles
    wr = csv.writer(f, dialect='excel', lineterminator='\n')
    for text in start_words:
        sentence = generate_next_word(text, 0)
        wr.writerow(sentence)


# TRAIN MODEL
# -----------
print('\nTraining Start-Time: ', time.ctime(time.time()))

# calls function on every epoch end
generate_callback = LambdaCallback(on_epoch_end=on_epoch_end)

# writes training stats to file
csv_logger = CSVLogger(path + '/Logs/' + version_name + '.log')

history = History()

model_check = ModelCheckpoint(path + '/Models/' + version_name + '_.{epoch:02d}.hdf5',
                              monitor='val_perplexity',
                              verbose=1,
                              save_best_only=False,
                              save_weights_only=False,
                              mode='auto',
                              period=1)

with open(path + '/Output/' + version_name + '.csv', 'w') as f:
    hist = primary_model.fit(train_input,
                             train_output,
                             batch_size=batch_size,
                             verbose=1,
                             shuffle='batch',
                             epochs=epochs,
                             validation_split=validation_split,
                             callbacks=[generate_callback, csv_logger, history])

print('\nTraining Finish Time: ', time.ctime(time.time()))

# SAVE MODEL
# -----------

with open(path + '/Logs/' + version_name + '_train_history.pkl', 'wb') as file:
    pickle.dump(hist.history, file)

print("\nSaving trained model...")
primary_model.save(path + '/Models/' + version_name + '.h5')

print("\nSaving model weights...")
primary_model.save_weights(path + '/Models/' + version_name + '_weights.h5')

print("\nSaving model to JSON...")
model_json_string = primary_model.to_json()
with open(path + '/Models/' + version_name + '.json', "w") as f_j:
    json.dump(json.loads(model_json_string), f_j, indent=4)

print("\nAll done!")
