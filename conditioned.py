import os.path
import time
import csv
import pickle
import json
import numpy as np
from random import shuffle, sample, randint
from nltk import word_tokenize

import keras.backend as K
from keras.models import Sequential
from keras.layers import Input, Embedding, LSTM, Dropout, Dense
from keras.optimizers import RMSprop
from keras.callbacks import History, CSVLogger, LambdaCallback, ModelCheckpoint

from clean_data import tokenize_dir, clean_tokens
from label_corpus import tag_corpus, clean_and_label
from embed_words import train_word_model, dictionary_lookups, vectorize_conditioned, vectorize_test_labels

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

# add start and end tags to each sentence
corpus = tag_corpus(sentences)

# CREATE LABELS
# -------------

print('\nLabelling data...')

tags = ['engine', 'body', 'speed', 'elegance', 'safety', 'transmission', 'fuel', 'electric', 'drive', 'reliable',
        'dynamic' 'road', 'urban', 'car', 'fun', 'love', 'excite', 'joy', 'curious', 'pleasure', 'roadster',
        'sophisticated', 'style', 'adventure', 'trip', 'special', 'journey', 'beautiful', 'launch', 'steering',
        'travel', 'design', 'future', 'advanced', 'driving', 'tech', 'imagine', 'control', 'amazing', 'wheel',
        'impressive', 'distinctive', 'diesel', 'petrol', 'celebrating', 'perfect', 'balance', 'interior', 'versatile',
        'practical']

corpus, labels = clean_and_label(corpus, tags)
max_sentence_len = len(max(corpus, key=len))
max_label_len = len(max(labels, key=len))

# Print data sizes
print("\nLabel Size: %d" % len(labels))
print("Max label length: %d" % max_label_len)
print("Corpus Size: %d" % len(corpus))
print("Max sentence length: %d" % max_sentence_len)

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

# CREATE WORD VECTORS FOR MODEL INPUT
# -----------------------------------

# Encoder Model
# encoder input shape (no sentences in labels, max-label-length)
encoder_input = np.zeros([len(labels), max_label_len], dtype=np.int32)

# Decoder Model
# decoder input shape (no sentences in corpus, max sentence length)
decoder_input = np.zeros([len(corpus), max_sentence_len], dtype=np.int32)
# decoder output shape (no sentences in corpus, max sentence length, 1)
decoder_output = np.zeros([len(corpus), max_sentence_len, 1], dtype=np.int32)

encoder_input_data, decoder_input_data, decoder_target_data = vectorize_conditioned(labels, corpus, encoder_input,
                                                                                    decoder_input, decoder_output,
                                                                                    word_to_index)

print('\nEncoder input shape: %s ' % str(encoder_input.shape))
print('Decoder input shape: %s ' % str(decoder_input.shape))
print('Decoder output shape: %s ' % str(decoder_output.shape))

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

# CREATE INFERENCE MODEL
# ----------------------

print("\nCreating the inference model...")

encoder_model = Sequential(encoder_inputs, encoder_states)

# Input states for model
decoder_state_input_h = Input(shape=(embedding_size,))
decoder_state_input_c = Input(shape=(embedding_size,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

final_embedded = decoder_embed(decoder_inputs)

# hidden layer will output states with prediction 
decoder_outputs2, state_h2, state_c2 = decoder_lstm(final_embedded, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)

decoder_model = Sequential([decoder_inputs] + decoder_states_inputs, [decoder_outputs2] + decoder_states2)

# CREATE TEST DATA AND EVALUATE MODEL
# -----------------------------------

print("\nCreating test data & evaluating model...")


def make_test_set(tags, num):
    all_labels = []
    for i in range(num):
        len = randint(2, 5)
        label = sample(tags, len)
        shuffle(label)
        label[:0] = ['<START>']
        label.append('<END>')
        all_labels.append(label)
    return all_labels


test_labels = make_test_set(tags, 50)

# assign the max length for vectorization
max_test_label_len = len(max(test_labels, key=len))

# define the shape of the encoded vector
encoder_vector = np.zeros([len(test_labels), max_test_label_len], dtype=np.int32)

# populate the vector with embeddings for label data
encoder_test_input = vectorize_test_labels(test_labels, encoder_vector, word_to_index)

# TRAINING SETUP
# --------------
print("\nVocab size: %d" % vocab_size)
print("Embedding size: %d" % embedding_size)

batch_size = 32
epochs = 25
validation_split = 0.2
print("\nTraining in batches of: %d" % batch_size)
print("Training epochs: %d" % epochs)


# GENERATOR FUNCTIONS

# apply temperature to each model output
def temp_sample(predictions, temperature=1.0):
    # print(len(predictions))
    if temperature <= 0:
        return np.argmax(predictions)
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions) / temperature
    expected_predictions = np.exp(predictions)
    predictions = expected_predictions / np.sum(expected_predictions)
    probability = np.random.multinomial(1, predictions, 1)
    return np.argmax(probability)


# Function to generate sequences
def generate_sequence(input_seq, temp_value):
    # the context vector from encoder
    states = encoder_model.predict(input_seq)
    # empty array for first target value
    target_seq = np.zeros((1, 1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = word_to_index('<START>')

    previous_word = '<START>'

    terminate = False
    sentence = ''
    while not terminate:
        # predicted word using previous word and state
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states)

        # generate a word with sampling
        predicted_word_index = temp_sample(output_tokens[0, -1, :], temperature=temp_value)
        current_word = index_to_word(predicted_word_index)
        # append to sentence so far
        sentence += ' ' + current_word

        # Sequence generation terminates if <END> or LINK token predicted
        # Or the same word consecutively predicted
        if (current_word == '<END>' or current_word == 'LINK' or previous_word == current_word or
                len(word_tokenize(sentence)) > max_sentence_len):
            terminate = True

        # Assign the current word to variable fot input at next time-step
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = predicted_word_index

        # Update states for next step
        states = [h, c]

        # temp assignment to keep track of sentence
        previous_word = current_word

    return sentence


# function to save output on each epoch
def save_results(epoch, _):
    # print results to file
    print("\nSaving greedy results to file...")
    print('Start-Time: ', time.ctime(time.time()))
    with open(path + '/Output/Conditioned/CSV/' + version_name + '_greedy_ep_' + str(epoch) + '.csv', 'w')as gcf:

        wrg = csv.writer(gcf, dialect='excel', lineterminator='\n')

        for i, label in enumerate(test_labels):
            greedy_output = generate_sequence(encoder_test_input[i:i + 1], 0)
            wrg.writerow([greedy_output])

    print('End-Time: ', time.ctime(time.time()))

    print("\nSaving random sampling results to file...")
    print('Start-Time: ', time.ctime(time.time()))
    with open(path + '/Output/Conditioned/CSV/' + version_name + '_sample_ep_' + str(epoch) + '.csv', 'w')as tcf:

        wrt = csv.writer(tcf, dialect='excel', lineterminator='\n')

        for i, label in enumerate(test_labels):
            temp_output = generate_sequence(encoder_test_input[i:i + 1], 0.6)
            wrt.writerow([temp_output])

    print('End-Time: ', time.ctime(time.time()))


results_callback = LambdaCallback(on_epoch_end=save_results)

model_check = ModelCheckpoint(path + '/Models/' + version_name + '_.{epoch:02d}.hdf5',
                              monitor='val_perplexity',
                              verbose=1,
                              save_best_only=False,
                              save_weights_only=False,
                              mode='auto',
                              period=1)

history = History()

csv_logger = CSVLogger(path + '/Logs/' + version_name + '.log')

# TRAIN MODEL
# -----------
hist = conditioned_model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                             batch_size=batch_size,
                             epochs=epochs,
                             shuffle='batch',  # check param
                             verbose=1,
                             validation_split=validation_split,
                             callbacks=[history, csv_logger, results_callback, model_check])

print('\nTraining Finish Time: ', time.ctime(time.time()))

with open(path + '/Logs/' + version_name + '_train_history.pkl', 'wb') as file:
    pickle.dump(hist.history, file)

print("\nSaving trained model...")
conditioned_model.save(path + '/Models/' + version_name + '.h5')

print("\nSaving model weights...")
conditioned_model.save_weights(path + '/Models/' + version_name + '_weights.h5')

print("\nSaving model to JSON...")
model_json_string = conditioned_model.to_json()
with open(path + '/Models/' + version_name + '.json', "w") as f_j:
    json.dump(json.loads(model_json_string), f_j, indent=4)

print("\nAll done!")
