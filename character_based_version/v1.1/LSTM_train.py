# improved LSTM_train.py -> bigger dataset, formatted data (specified vocabulary), bigger NN, more parameters...

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.models import load_model
import os
import pickle
import sys

MODEL = '/home/novak_luka93/chatbot-unk/character_based_version/v1.1/LSTM_model.h5'
MODEL_WEIGHTS = '/home/novak_luka93/chatbot-unk/character_based_version/v1.1/LSTM_model_weights.h5'
ROOTDIR = '/home/novak_luka93/wikidump'
VOCABULARY = '/home/novak_luka93/chatbot-unk/character_based_version/v1.0/vocab'
CHAR_DICT = '/home/novak_luka93/chatbot-unk/character_based_version/v1.1/char_dict.pkl'

# # create vocabulary with all ascii characters
# L = list(range(128))
# VOCAB = list(''.join(map(chr, L)))

# list of all allowed characters
with open(VOCABULARY, 'r', encoding='utf-8') as v:
    VOCAB = eval(v.read())

VOCAB = sorted(VOCAB)
print(VOCAB)

# hyperparameters
NUM_EPOCH = 50
BATCH_SIZE = 32
NUM_HIDDEN = 256
VERBOSE = 1
DATA_SIZE = 900000
SEQ_LENGTH = 100
VOCAB_SIZE = len(VOCAB)
INPUT_SHAPE = ((DATA_SIZE - SEQ_LENGTH), VOCAB_SIZE)

# create mapping of unique chars to integers
char_to_int = dict((c, i) for i, c in enumerate(VOCAB))

# saving dictionary for model prediction
with open(CHAR_DICT, 'wb') as f:
    pickle.dump(char_to_int, f, pickle.HIGHEST_PROTOCOL)

# summarize the character vocabulary
n_vocab = len(VOCAB)
print("Total Vocab: ", n_vocab)

# define the LSTM model
model = Sequential()
model.add(LSTM(NUM_HIDDEN, input_shape=INPUT_SHAPE, return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(NUM_HIDDEN, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(NUM_HIDDEN))
model.add(Dropout(0.2))
model.add(Dense((DATA_SIZE - SEQ_LENGTH), activation='softmax'))
model.summary()

adam_optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-06)

model.compile(loss='categorical_crossentropy', optimizer=adam_optimizer, verbose=VERBOSE)

model.save(MODEL)

for subdir, dirs, files in os.walk(ROOTDIR):
    name = str(subdir).split('/')[-1]
    print(subdir)
    list_files = files
    completed = []
    for f in files:
        print('Working on file:', f)
        SOURCE = str(subdir) + '/' + str(f)

	# load text and covert to lowercase
        raw_text = open(SOURCE, encoding='utf-8').read()
        raw_text = raw_text.lower()

        text_list = raw_text.split(' ')
        i = 0
        while i < len(text_list):
            if text_list[i] == '':
                text_list.pop(i)
                continue
            else:
                text_list[i] = text_list[i].strip()
                i += 1

        raw_text = ' '.join(text_list)

        if len(raw_text) >= DATA_SIZE:
            raw_text = raw_text[:DATA_SIZE]
        else:
            continue

        # i = 0
        # while i < len(raw_text):
        #     if raw_text[i] == '':
        #         raw_text[i] == '*'
        #     i += 1

		# # create mapping of unique chars to integers
        # chars = sorted(list(set(raw_text)))
        # char_to_int = dict((c, i) for i, c in enumerate(chars))

		# summarize the loaded data
        n_chars = len(raw_text)
        # n_vocab = len(chars)
        print("Total Characters in Article: ", n_chars)
        # print("Total Vocab: ", n_vocab)

		# prepare the dataset of input to output pairs encoded as integers
        dataX = []
        dataY = []
        for i in range(0, n_chars - SEQ_LENGTH, 1):
        	seq_in = raw_text[i:i + SEQ_LENGTH]
            # print('Seq_in_' + str(i) + ': ' + seq_in)
        	seq_out = raw_text[i + SEQ_LENGTH]
            # print('Seq_out_' + str(i) + ': ' + seq_out)
        	dataX.append([char_to_int[char] for char in seq_in])
        	dataY.append(char_to_int[seq_out])
        n_patterns = len(dataX)
        print("Total Patterns: ", n_patterns)

        # reshape X to be [samples, time steps, features]
        X = numpy.reshape(dataX, (n_patterns, SEQ_LENGTH, 1))
        print(X.shape)

        # normalize
        X = X / float(n_vocab)

        # one hot encode the output variable
        y = np_utils.to_categorical(dataY)
        print('Y shape:', y.shape)

        model = load_model(MODEL)

        # fit the model
        model.fit(X, y, epochs=NUM_EPOCH, batch_size=BATCH_SIZE, verbose=VERBOSE)

        print('\nSaving model...\n')
        model.save(MODEL)
        model.save_weights(MODEL_WEIGHTS)
        print('Saved!\n')

print('Done.')
