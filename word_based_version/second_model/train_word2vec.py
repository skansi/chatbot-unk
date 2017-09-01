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
import gensim

MODEL = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/word_based_version/second_model/model.h5'
MODEL_WEIGHTS = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/word_based_version/second_model/model_weights.h5'
ROOTDIR = '/home/prometej/Workspaces/PythonWorkspace/Resources/wikidump_pom'
VOCABULARY_LEMMATIZED = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/word_based_version/file_resources/words_lemmatized_vocab.pkl'
WORD_TO_INT = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/word_based_version/file_resources/word_to_int.pkl'
INT_TO_WORD = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/word_based_version/file_resources/int_to_word.pkl'

NUM_WORDS = 8000

# Load Google's pre-trained Word2Vec model.
model_gensim = gensim.models.KeyedVectors.load_word2vec_format('/home/prometej/Workspaces/PythonWorkspace/Resources/GoogleNews-vectors-negative300.bin', binary=True)

# hyperparameters
NUM_EPOCH = 10
BATCH_SIZE = 64
NUM_HIDDEN = 128
VERBOSE = 1
CONTEXT = 100
DATA_SIZE = 1024 + CONTEXT
# VOCAB_SIZE = len(VOCAB)
EMBEDDING_SIZE = 300
# OPTIMIZER = Adam(lr=0.001, decay=1e-03)
OPTIMIZER = RMSprop(decay=1e-03)
METRICS = ['accuracy']
INPUT_SHAPE = (CONTEXT, EMBEDDING_SIZE)

print('Input shape:', INPUT_SHAPE)

lemmatizer = WordNetLemmatizer()

# define the LSTM model, compile it and save it
model = Sequential()
model.add(LSTM(NUM_HIDDEN, input_shape=INPUT_SHAPE, batch_size=BATCH_SIZE, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(NUM_HIDDEN, return_sequences=True))
model.add(Dropout(0.25))
model.add(LSTM(NUM_HIDDEN))
model.add(Dropout(0.2))
model.add(Dense(units=EMBEDDING_SIZE))
# model.add(Activation('softmax'))
model.summary()

model.compile(loss='cosine_proximity', optimizer=OPTIMIZER, verbose=VERBOSE, metrics=METRICS)

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

        print('*'*46)
        print('> Lemmatizing data....')
        # lemmatizing data
        for w in range(len(text_list)):
            word = lemmatizer.lemmatize(text_list[w])
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
                for x in seq_in:
                    if x not in model_gensim.wv.vocab:
                        dataX.append(np.zeros(300))
                    else:
                        dataX.append(model_gensim.wv[x])

                if seq_out not in model_gensim.wv.vocab:
                    dataY.append(np.zeros(300))
                else:
                    dataY.append(model_gensim.wv[seq_out])
            	# dataX.append([word_to_int[word] for word in seq_in])
            	# dataY.append(word_to_int[seq_out])

            N_SAMPLES = int(len(dataX)/CONTEXT)
            print("Done.\n\nTotal Number Of Samples: ", N_SAMPLES)
            print('*'*46 + '\n\n')

            print('*'*46)
            print('Data:\n')
            dataX = np.array(dataX).flatten()
            # reshape X to be [samples, time steps, features]
            X = np.reshape(np.array(dataX),(N_SAMPLES, CONTEXT, EMBEDDING_SIZE))
            print('X:', X.shape)

            dataY = np.array(dataY).flatten()
            y = np.reshape(np.array(dataY), (N_SAMPLES, EMBEDDING_SIZE))
            print('y:', y.shape)
            # print('*'*46 + '\n\n')

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
