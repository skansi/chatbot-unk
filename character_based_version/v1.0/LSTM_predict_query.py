import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils
from keras.models import load_model

NUM_CHARS_TO_GENERATE = 100

# load ascii text and covert to lowercase
filename = "/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/Resources/wikidump/A/AA/wiki_00"
raw_text = open(filename).read()
raw_text = raw_text.lower()

# create mapping of unique chars to integers, and a reverse mapping
chars = sorted(list(set(raw_text)))
print(chars)
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []

for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)

print("Total Patterns: ", n_patterns)

# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 1))

# normalize
X = X / float(n_vocab)

model = load_model('/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/character_based_version/v1.0/LSTM_model.h5')

# instead of random seed, user will input the query and the network will answer
query = input('Hello! How can I help you?\n\n')
query = query.lower()
size = len(query)
l = list(query)
list_char = []
for i in range(len(l)):
	list_char.append(char_to_int[l[i]])

pattern = [0] * (100)
pattern[100-size:] = list_char
pattern = list(pattern)

print('\nAnswer: ')

# generate characters
for i in range(NUM_CHARS_TO_GENERATE):
	x = np.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = np.argmax(prediction)
	result = int_to_char[index]
	if i == 0:
		result = result.upper()
	seq_in = [int_to_char[value] for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]

print("\nDone.")
