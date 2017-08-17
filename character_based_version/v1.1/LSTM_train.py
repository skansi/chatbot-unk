
# improved LSTM_train.py -> bigger dataset, formatted data (specified vocabulary), bigger NN, more parameters...

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Dropout
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.optimizers import Adam, Nadam, RMSprop
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.models import load_model
import numpy as np
import os
import os.path
import pickle
import sys
from hyphen import Hyphenator, dict_info
from hyphen.dictools import *

MODEL = '/home/novak_luka93/chatbot-unk/character_based_version/v1.1/LSTM_model.h5'
MODEL_WEIGHTS = '/home/novak_luka93/chatbot-unk/character_based_version/v1.1/LSTM_model_weights.h5'
ROOTDIR = '/home/novak_luka93/wikidump'
VOCABULARY = '/home/novak_luka93/chatbot-unk/character_based_version/v1.1/syllable_vocab.pkl'
SYLLABLE_DICT = '/home/novak_luka93/chatbot-unk/character_based_version/v1.1/syllable_dict.pkl'

# list of all allowed syllables
with open(VOCABULARY, 'rb') as v:
    VOCAB = pickle.load(v)

# adding space character to vocabulary
if ' ' not in VOCAB:
    VOCAB = [' '] + VOCAB

# hyperparameters
NUM_EPOCH = 10
BATCH_SIZE = 32
NUM_HIDDEN = 128
VERBOSE = 1
CONTEXT = 100
DATA_SIZE = 5120 + CONTEXT
VOCAB_SIZE = 10000 # len(VOCAB)
# OPTIMIZER = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-03)
OPTIMIZER = RMSprop(decay=1e-03)
METRICS = ['accuracy']
INPUT_SHAPE = (CONTEXT, VOCAB_SIZE)

VOCAB = VOCAB[:VOCAB_SIZE]

print('Input shape:', INPUT_SHAPE)

# specify and create syllable splitter
h_en = Hyphenator('en_US')

# check if dictionary already exists then load it, else create it and save it for the future
if os.path.isfile(SYLLABLE_DICT):
    with open(SYLLABLE_DICT, 'rb') as f:
        syllable_to_int = pickle.load(f)
else:
    # create mapping of unique chars to integers
    syllable_to_int = dict((c, i) for i, c in enumerate(VOCAB))

    # saving dictionary for model prediction
    with open(SYLLABLE_DICT, 'wb+') as f:
        pickle.dump(syllable_to_int, f, pickle.HIGHEST_PROTOCOL)

# define the LSTM model, compile it and save it
model = Sequential()
model.add(LSTM(NUM_HIDDEN, input_shape=INPUT_SHAPE, batch_size=BATCH_SIZE, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(NUM_HIDDEN, return_sequences=True))
model.add(Dropout(0.25))
model.add(LSTM(NUM_HIDDEN))
model.add(Dropout(0.2))
model.add(Dense(units=VOCAB_SIZE, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, verbose=VERBOSE, metrics=METRICS)

model.save(MODEL)

# walk the filetree, split the data in syllables, create saples and labels for each sample, load the model and train it
for subdir, dirs, files in os.walk(ROOTDIR):
    name = str(subdir).split('/')[-1]
    print(subdir)
    list_files = files
    completed = []
    for f in files:
        print('Working on file:', f)
        SOURCE = str(subdir) + '/' + str(f)

        # load text and covert to lowercase
        text = open(SOURCE, encoding='utf-8').read()
        text = text.lower()
        text_list = text.split()

        syllables_list = []

        # split the data on syllables
        print('\n> Splitting data to syllables...')
        for word in text_list:
            try:
                l = h_en.syllables(word)
                for s in l:
                    if l == []:
                        s = ' '
                    if l.index(s) == (len(l) - 1):
                        s = s + ' '
                    syllables_list = syllables_list + [s]
            except ValueError:
                print(word)
        print('Done!\n')

        print('> Changing data to be written only with syllables from vocabulary...')
        syllables_list = [i for i in syllables_list if i in VOCAB]
        print('Done!\n')

        # check the size of the data and see if splitting is needed
        repeat = 1

        if len(syllables_list) >= (2*DATA_SIZE + 100):
            repeat = 2
            print('Data split in 2 because of its size!\n')
        elif len(syllables_list) > DATA_SIZE:
            print('Data shrinked to certain size to fit the net!\n')
        else:
            print('Data too small! Skipping...\n')
            continue

        for i in range(repeat):

            if i == 0:
                raw_text = syllables_list[:DATA_SIZE]
            else:
                raw_text = syllables_list[DATA_SIZE:2*DATA_SIZE]

    		# summarize the loaded data
            n_syllables = len(raw_text)
            print("Total Syllables in Article: ", n_syllables)

            print('\n> Preparing the dataset...')
    		# prepare the dataset of input to output pairs encoded as integers
            dataX = []
            dataY = []
            for i in range(0, n_syllables - CONTEXT):
            	seq_in = raw_text[i:i + CONTEXT]
                # print('Seq_in_' + str(i) + ': ' + seq_in)
            	seq_out = raw_text[i + CONTEXT]
                # print('Seq_out_' + str(i) + ': ' + seq_out)
            	dataX.append([syllable_to_int[syllable] for syllable in seq_in])
            	dataY.append(syllable_to_int[seq_out])
            N_SAMPLES = len(dataX)
            print("Done.\n\nTotal Number Of Samples: ", N_SAMPLES)

            # normalize and one hot encode every syllable from the context
            print('\n> One-hot-encoding the training data...')
            list_samples = []
            for x in dataX:
            	# x = [(i / VOCAB_SIZE) for i in x]
            	list_samples.append(np_utils.to_categorical(x, num_classes=VOCAB_SIZE))
            print('Done!\n\n')

            # reshape X to be [samples, time steps, features]
            X = np.reshape(np.array(list_samples),(N_SAMPLES, CONTEXT, VOCAB_SIZE))
            print('X:', X.shape)

            # one hot encode the labels
            print('\nDataY size:', len(dataY))
            print()
            y = np_utils.to_categorical(dataY, num_classes=VOCAB_SIZE)
            y = np.reshape(y, (N_SAMPLES, VOCAB_SIZE))
            print('y:', y.shape)

            # load the model
            model = load_model(MODEL)

            # fit the model = train it on given data
            print('\n> Training the model...')
            model.fit(X, y, epochs=NUM_EPOCH, batch_size=BATCH_SIZE, verbose=VERBOSE)
            print('Done!\n')

            # save the model so that is possible to resume training when loaded again
            print('\n> Saving model...\n')
            model.save(MODEL)
            model.save_weights(MODEL_WEIGHTS)
            print('Saved!\n')

print('Done.')
