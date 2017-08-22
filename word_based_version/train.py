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
from nltk.stem import WordNetLemmatizer
import numpy as np
import os
import os.path
import pickle
import sys

MODEL = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/word_based_version/model.h5'
MODEL_WEIGHTS = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/word_based_version/model_weights.h5'
ROOTDIR = '/home/prometej/Workspaces/PythonWorkspace/Resources/wikidump'
VOCABULARY_LEMMATIZED = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/word_based_version/words_lemmatized_vocab.pkl'
WORD_DICT_LEMMA = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/word_based_version/lemma_word_dict.pkl'

NUM_WORDS = 10000

# list of all allowed words
with open(VOCABULARY_LEMMATIZED, 'rb') as v:
    VOCAB = pickle.load(v)

VOCAB = sorted(VOCAB.keys())

VOCAB = VOCAB[:NUM_WORDS]

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

lemmatizer = WordNetLemmatizer()

# check if dictionary already exists then load it, else create it and save it for the future
if os.path.isfile(WORD_DICT_LEMMA):
    with open(WORD_DICT_LEMMA, 'rb') as f:
        word_to_int = pickle.load(f)
else:
    # create mapping of unique chars to integers
    word_to_int = dict((c, i) for i, c in enumerate(VOCAB))

    # saving dictionary for model prediction
    with open(WORD_DICT_LEMMA, 'wb+') as f:
        pickle.dump(word_to_int, f, pickle.HIGHEST_PROTOCOL)


# define the LSTM model, compile it and save it
model = Sequential()
model.add(LSTM(NUM_HIDDEN, input_shape=INPUT_SHAPE, batch_size=BATCH_SIZE, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(NUM_HIDDEN, return_sequences=True))
model.add(Dropout(0.25))
model.add(LSTM(NUM_HIDDEN))
model.add(Dropout(0.2))
model.add(Dense(units=VOCAB_SIZE))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, verbose=VERBOSE, metrics=METRICS)

model.save(MODEL)

print('\n\n' + '*'*46)
print('> Starting.... Walking the file tree and learning on every file')
# walk the filetree, split the data in syllables, create saples and labels for each sample, load the model and train it
for subdir, dirs, files in os.walk(ROOTDIR):
    name = str(subdir).split('/')[-1]
    print(subdir)
    for f in files:
        print('Working on file:', f)
        SOURCE = str(subdir) + '/' + str(f)

        # load text and covert to lowercase
        text = open(SOURCE, encoding='utf-8').read()
        text = text.lower()
        text_list = text.split()

        print('\n\n' + '*'*46)
        print('> Changing data to be written only with words from vocabulary...')
        text_list = [i for i in text_list if i in VOCAB]
        print('Done!')
        print('*'*46 + '\n\n')

        print('*'*46)
        print('> Lemmatizing data....')
        # lemmatizing data
        for w in range(len(text_list)):
            word = lemmatizer.lemmatize(text_list[w])
            if word not in VOCAB:
                text_list.remove(text_list[w])
            else:
                text_list[w] = word
        print('Done!')
        print('*'*46 + '\n\n')

        # check the size of the data and see if splitting is needed
        repeat = 1

        if len(text_list) >= (2*DATA_SIZE):
            repeat = 2
            print('Data split in 2 because of its size!\n\n')
        elif len(text_list) > DATA_SIZE:
            print('Data shrinked to certain size to fit the net!\n\n')
        else:
            print('Data too small! Skipping...\n\n')
            continue

        for i in range(repeat):

            if i == 0:
                raw_text = text_list[:DATA_SIZE]
            else:
                raw_text = text_list[DATA_SIZE:2*DATA_SIZE]

    		# summarize the loaded data
            n_words = len(raw_text)
            print("Total Number of Words in Article: ", n_words)

            print('\n\n' + '*'*46)
            print('> Preparing the dataset...')
    		# prepare the dataset of input to output pairs encoded as integers
            dataX = []
            dataY = []
            for index in range(0, n_words - CONTEXT):
            	seq_in = raw_text[index:index + CONTEXT]
                # print('Seq_in_' + str(i) + ': ' + seq_in)
            	seq_out = raw_text[index + CONTEXT]
                # print('Seq_out_' + str(i) + ': ' + seq_out)
            	dataX.append([word_to_int[word] for word in seq_in])
            	dataY.append(word_to_int[seq_out])
            N_SAMPLES = len(dataX)
            print("Done.\n\nTotal Number Of Samples: ", N_SAMPLES)
            print('*'*46 + '\n\n')

            print('*'*46)
            # normalize and one hot encode every syllable from the context
            print('> One-hot-encoding the training data...')
            list_samples = []
            for x in dataX:
            	# x = [(i / VOCAB_SIZE) for i in x]
            	list_samples.append(np_utils.to_categorical(x, num_classes=VOCAB_SIZE))
            print('Done!')
            print('*'*46 + '\n\n')

            print('*'*46)
            print('Data:\n')
            # reshape X to be [samples, time steps, features]
            X = np.reshape(np.array(list_samples),(N_SAMPLES, CONTEXT, VOCAB_SIZE))
            print('X:', X.shape)

            # one hot encode the labels
            y = np_utils.to_categorical(dataY, num_classes=VOCAB_SIZE)
            y = np.reshape(y, (N_SAMPLES, VOCAB_SIZE))
            print('y:', y.shape)
            print('*'*46 + '\n\n')

            # load the model
            model = load_model(MODEL)

            print('*'*46)
            # fit the model = train it on given data
            print('> Training the model...')
            model.fit(X, y, epochs=NUM_EPOCH, batch_size=BATCH_SIZE, verbose=VERBOSE)
            print('Done!\n')
            print('*'*46 + '\n\n')

            # save the model so that is possible to resume training when loaded again
            print('\n> Saving model...\n')
            model.save(MODEL)
            model.save_weights(MODEL_WEIGHTS)
            print('Saved!\n')

print('Done.')
