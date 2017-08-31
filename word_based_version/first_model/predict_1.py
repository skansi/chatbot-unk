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

MODEL = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/word_based_version/file_resources/model.h5'
MODEL_WEIGHTS = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/word_based_version/file_resources/model_weights.h5'
ROOTDIR = '/home/prometej/Workspaces/PythonWorkspace/Resources/wikidump'
VOCABULARY_LEMMATIZED = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/word_based_version/file_resources/words_lemmatized_vocab.pkl'
WORD_TO_INT = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/word_based_version/file_resources/word_to_int.pkl'
INT_TO_WORD = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/word_based_version/file_resources/int_to_word.pkl'

NUM_WORDS = 8000
CONTEXT = 100

# list of all allowed words
with open(VOCABULARY_LEMMATIZED, 'rb') as v:
    VOCAB = pickle.load(v)

VOCAB = sorted(VOCAB.keys())

VOCAB = VOCAB[:NUM_WORDS]

lemmatizer = WordNetLemmatizer()

# load dictionary
with open(WORD_TO_INT, 'rb') as f1:
    word_to_int = pickle.load(f1)

with open(INT_TO_WORD, 'rb') as f2:
    int_to_word = pickle.load(f2)


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
			if int_to_word[i] in VOCAB:
				print(int_to_word[i])
			else:
				print('unknown')

		# if predicted word is an end of sentence token select a new token in 50% of the cases
		if(int_to_word[the_one] == END_OF_SENTENCE):
			if(repeat == 1):
				the_one = np.asscalar(np.random.choice(choices, 1, probs))

		print('\nSelected: ', int_to_word[the_one])
		print()
		return the_one

# load the model
model = load_model(MODEL)
model.summary()

query = input('Hi! How can I help you?\n\n> ')

query = query.lower()

query_list = query.split()

query_list = [word for word in query_list if word in VOCAB]

print(len(query_list))

query_int = []
for i in range(len(query_list)):
    query_list[i] = lemmatizer.lemmatize(query_list[i])
    query_int.append(word_to_int[query_list[i]])

# padding
input_sequence = [word_to_int['unk']]*(CONTEXT*63)
tmp = ([word_to_int['unk']]*(CONTEXT-len(query_int)))

# append actual query
for i in query_int:
    tmp.append(i)

input_sequence += tmp

print(len(input_sequence))

input_matrix = []
for i in input_sequence:
    input_matrix.append(np_utils.to_categorical(i, num_classes=len(VOCAB)))

predict_input = np.reshape(np.array(input_matrix), (64, CONTEXT, len(VOCAB)))

print(predict_input.shape)

result = []

for i in range(10):

    # predict next word
    predicted = model.predict(predict_input, batch_size=64, verbose=1)

    # # select 5 words with the biggest probability to be matching the context
    # top_5_indices = np.argsort(predicted)[-5:]
    # top_5_probs = []
    # for i in range(5):
    #     top_5_probs.append(predicted[top_5_indices[i]])
    #
    # selected = select_one(top_5_indices, top_5_probs)
    #
    # word = int_to_word[selected]

    selected = np.argmax(predicted[-1])

    if int_to_word[selected] == 'a':
        selected = np.argmax(predicted[-2])

    # append result to the results array
    result.append(selected)

    print(int_to_word[selected])

    s = np_utils.to_categorical(selected, num_classes=len(VOCAB))

    predict_input = np.append(predict_input, s)

    predict_input = predict_input[len(VOCAB):]

    predict_input = np.reshape(predict_input, (64, CONTEXT, len(VOCAB)))

ans_1 = []
for i in result:
    ans_1.append(int_to_word[i])

answer_1 = ' '.join(ans_1)

print('\n' + '*'*50)
print('\nAnswer whole: ' + answer_1)


if '$#' in ans_1:
    index = ans_1.index('$#')
    ans_2 = ans_1[:index]

    answer_2 = ' '.join(ans_2)

    print('\n' + '*'*50)
    print('\nAnswer cutoff: ' + answer_2)

# result_split = result.split()
# if '$#' in result_split:
#     index = result_split.index('$#')
#     result_ = result_split[:index]
#
#     result_.append('.')
#
#     answer_2 = ' '.join(result_)
#
#     print('\n' + '*'*50)
#     print('\nAnswer cutoff: ' + answer_2)

print('\n\nDone!')
