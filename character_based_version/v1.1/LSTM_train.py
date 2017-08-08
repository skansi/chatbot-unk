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

MODEL = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/character_based_version/v1.0/LSTM_model.h5'
ROOTDIR = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/Resources/wikidump'
VOCABULARY = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/character_based_version/v1.0/vocab'

# # create vocabulary with all ascii characters
# L = list(range(128))
# VOCAB = list(''.join(map(chr, L)))

with open(VOCABULARY, 'r') as v:
    VOCAB = list(v.read())

# hyperparameters
NUM_EPOCH = 50
BATCH_SIZE = 32
NUM_HIDDEN = 256
VERBOSE = 1
DATA_SIZE = 900000
SEQ_LENGTH = 100
VOCAB_SIZE = len(VOCAB)
# VOCAB_SIZE = len(VOCAB)
INPUT_SHAPE = ((DATA_SIZE - SEQ_LENGTH), VOCAB_SIZE)

# create mapping of unique chars to integers
char_to_int = dict((c, i) for i, c in enumerate(chars))

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
    list_files = files
    completed = []
    for f in files:
        print('Working on file:', f)
        SOURCE = str(subdir) + '/' + str(f)

		# load ascii text and covert to lowercase
        raw_text = open(SOURCE).read()
        raw_text = raw_text.decode('utf-8').encode('ascii', 'replace')
        raw_text = raw_text.lower()
        if len(raw_text) >= DATA_SIZE:
            raw_text = raw_text[:DATA_SIZE]
        else:
            continue

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
        	seq_out = raw_text[i + SEQ_LENGTH]
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
        print('Saved!\n')

print('Done.')
