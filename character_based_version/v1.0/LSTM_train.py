import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import Adam

BATCH_SIZE = 32

# load ascii text and covert to lowercase
filename = "/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/Resources/wikidump/A/AA/wiki_00"
raw_text = open(filename).read()
raw_text = raw_text.lower()

# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

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

# transform the data for stateful LSTM
size = (len(dataX))//BATCH_SIZE
dataX = dataX[:size//10000*BATCH_SIZE]
dataY = dataY[:size//10000*BATCH_SIZE]

n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))

# normalize
X = X / float(n_vocab)

# one hot encode the output variable
y = np_utils.to_categorical(dataY)

# define the LSTM model
model = Sequential()
model.add(LSTM(2, input_shape=(X.shape[1], X.shape[2]), stateful=True, batch_input_shape=(BATCH_SIZE ,X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(2, stateful=True))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.summary()

adam_optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-06)

model.compile(loss='categorical_crossentropy', optimizer=adam_optimizer)

# fit the model
model.fit(X, y, epochs=50, batch_size=BATCH_SIZE)

print('\nSaving model:\n')
model.save('/home/prometej/Workspaces/PythonWorkspace/project_chatbot/LSTM_model.h5')
