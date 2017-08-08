import numpy as np
from keras.models import load_model
from keras.layers import Dense, Activation, Flatten
from keras.layers import Dropout
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.models import Sequential
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import os

number = input('How many distinct words? ')
CUT_INDEX = int(number)

# references
MODEL = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/LSTM_model_test.h5'
ROOTDIR = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/Resources/wikidump/test'
WORDS = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/distinct_words.txt'

# hypereparameters
CONTEXT = 4
NUM_HIDDEN = 256
BATCH_SIZE = 60
NUM_CYCLES = 2
NUM_EPOCHS_PER_CYCLES = 2 # default 1

# shift the context list so that previously predicted word is in the context for the next prediction
def shift(l, n):
	return l[n:] + l[:n]

# select which word to predict based on the probability for the word to match the context
# this way we ensure that not always for the same context will the same word be selected
def select_one(choices, probs):
	l = 0
	while(l < 2):
		repeat = randint(0, 1)

		# selecting one element from choices with probability p from probs
		# the type has to be changed from np.ndarray of size 1 to int in order to be used as an index
		the_one = np.asscalar(np.random.choice(choices, 1, probs))
		print("Choices:")
		for i in choices:
			print(index2word[i])

		# if predicted word is an end of sentence token select a new token in 50% of the cases
		if(index2word[the_one] == END_OF_SENTENCE):
			if(repeat == 1):
				the_one = np.asscalar(np.random.choice(choices, 1, probs))

		print('\nSelected: ', index2word[the_one])
		print()
		return the_one

def create_word_indices_for_text(text_list):
	input_words = []
	label_word = []
	for i in range(0,len(text_list) - CONTEXT):
		input_words.append((text_list[i:i+CONTEXT]))
		label_word.append((text_list[i+CONTEXT]))
	return input_words, label_word

# most frequent distinct words
with open(WORDS, 'r+') as f:
    distinct_words = f.read().split()

distinct_words = distinct_words[:CUT_INDEX-1]

distinct_words.append('unknown')

NUM_WORDS = len(distinct_words)
print('Number of distinct words:', NUM_WORDS)
print()

# create dictionary to get ID for given word and word for given ID
word2index = dict((w, i) for i, w in enumerate(distinct_words))
index2word = dict((i, w) for i, w in enumerate(distinct_words))

# define the LSTM model
model = Sequential()
model.add(LSTM(NUM_HIDDEN, input_shape=(CONTEXT, NUM_WORDS), activation='tanh', recurrent_activation='hard_sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(NUM_WORDS))
model.add(Activation("softmax"))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam')

# save the model with option to continue training
model.save(MODEL)

# walk throught files
for subdir, dirs, files in os.walk(ROOTDIR):
    print(subdir)
    name = str(subdir).split('/')[-1]
    for f in files:
        print('Working on file:', f)
        SOURCE = str(subdir) + '/' + str(f)
        with open(SOURCE, 'r+') as file_i:
            data = file_i.read()

        text_list = data.lower().split()
        for i in range(len(text_list)):
            if text_list[i] not in distinct_words:
                text_list[i] = 'unknown'

        input_words, label_word = create_word_indices_for_text(text_list)
        input_vectors = np.zeros((len(input_words), CONTEXT, NUM_WORDS), dtype=np.float32)
        vectorized_labels = np.zeros((len(input_words), NUM_WORDS), dtype=np.float32)

        for i, input_w in enumerate(input_words):
        	for j, w in enumerate(input_w):
        		input_vectors[i, j, word2index[w]] = 1
        		vectorized_labels[i, word2index[label_word[i]]] = 1

        model = load_model(MODEL)

        for cycle in range(NUM_CYCLES):
        	print(">-<" * 44)
        	print(" Cycle: %d" % (cycle+1))
        	# print(" Size of the answer:", str(10 + NUM_PREDICTING))
        	model.fit(input_vectors, vectorized_labels, batch_size = BATCH_SIZE, epochs = NUM_EPOCHS_PER_CYCLES)
        	test_index = np.random.randint(len(input_words))
        	test_words = input_words[test_index]

        # save model (architecture, weights, train_config(loss, optimizer), state_of_the_optimizer)
        model.save(MODEL)

model.save(MODEL)
