from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Dropout, Activation
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.optimizers import Adam, Nadam, RMSprop
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.models import load_model
from keras import regularizers
from keras.layers.wrappers import TimeDistributed
import numpy as np
import os
import os.path
import pickle
import sys

MODEL = '/home/novak_luka93/chatbot-unk/character_based_version/v1.0/MLP_model.h5'
ROOTDIR = '/home/novak_luka93/wikidump'
VOCABULARY = '/home/novak_luka93/chatbot-unk/character_based_version/v1.1/vocab2'
CHAR_DICT = '/home/novak_luka93/chatbot-unk/character_based_version/v1.1/char_dict.pkl'

# list of all allowed syllables
with open(VOCABULARY, 'r') as v:
    VOCAB = eval(v.read())

# check if dictionary already exists then load it, else create it and save it for the future
if os.path.isfile(CHAR_DICT):
    with open(CHAR_DICT, 'rb') as f:
        char_to_int = pickle.load(f)
else:
    # create mapping of unique chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(VOCAB))

    # saving dictionary for model prediction
    with open(CHAR_DICT, 'wb+') as f:
        pickle.dump(char_to_int, f, pickle.HIGHEST_PROTOCOL)

# hyperparameters
NUM_EPOCH = 10
BATCH_SIZE = 32
NUM_HIDDEN = 128
VERBOSE = 1
CONTEXT = 100
DATA_SIZE = 1024 + CONTEXT
VOCAB_SIZE = len(VOCAB)
# OPTIMIZER = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-03)
OPTIMIZER = RMSprop(decay=1e-03)
METRICS = ['accuracy']
INPUT_SHAPE = (CONTEXT, VOCAB_SIZE)

print('Input shape:', INPUT_SHAPE)

model = Sequential()
model.add(Dense(500, input_shape = INPUT_SHAPE))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(250))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('linear'))
model.compile(optimizer='adam', loss='mse', verbose=1)

model.save(MODEL)

for subdir, dirs, files in os.walk(ROOTDIR):
    name = str(subdir).split('/')[-1]
    print(subdir)
    for f in files:
        print('Working on file:', f)
        SOURCE = str(subdir) + '/' + str(f)

        # load text and covert to lowercase
        text = open(SOURCE, encoding='utf-8').read()
        text = text.lower()
        text_list = list(text)

        print('\n> Preparing the dataset...')
        # prepare the dataset of input to output pairs encoded as integers
        dataX = []
        dataY = []
        for index in range(0, len(text_list) - CONTEXT):
            seq_in = text_list[index:index + CONTEXT]
            # print('Seq_in_' + str(i) + ': ' + seq_in)
            seq_out = text_list[index + CONTEXT]
            # print('Seq_out_' + str(i) + ': ' + seq_out)
            dataX.append([word_to_int[word] for word in seq_in])
            dataY.append(word_to_int[seq_out])
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
