import numpy as np
from random import randint
from keras. models import load_model

number = input('How many distinct words? ')
CUT_INDEX = int(number)

# references
MODEL = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/LSTM_model_test.h5'
ROOTDIR = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/Resources/wikidump'
WORDS = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/distinct_words.txt'

model = load_model(MODEL)
FILE = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/Resources/wikidump/test/wiki_00'

# NUM_PREDICTING = 6
NUM_PREDICTING = randint(2, 5) # number of words generated
CONTEXT = 4
NUM_HIDDEN = 50
BATCH_SIZE = 60
NUM_CYCLES = 50
NUM_EPOCHS_PER_CYCLES = 5 #default 1
END_OF_SENTENCE = '$#'
CUTOFF_PROB = 0.5

# remove END_OF_SENTENCE symbol and put dot symbol instead
def remove_EOS(last_word):
	last_word = last_word[:len(last_word)-len(END_OF_SENTENCE)]
	return last_word

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
			if index2word[i] in distinct_words:
				print(index2word[i])
			else:
				print('unknown')

		# if predicted word is an end of sentence token select a new token in 50% of the cases
		if(index2word[the_one] == END_OF_SENTENCE):
			if(repeat == 1):
				the_one = np.asscalar(np.random.choice(choices, 1, probs))

		print('\nSelected: ', index2word[the_one])
		print()
		return the_one

# most frequent distinct words
with open(WORDS, 'r+') as f:
    distinct_words = f.read().split()

distinct_words = distinct_words[:CUT_INDEX-1]

distinct_words.append('unknown')

NUM_WORDS = len(distinct_words)
print('Number of distinct words:', NUM_WORDS)
print()

word2index = dict((w, i) for i, w in enumerate(distinct_words))
index2word = dict((i, w) for i, w in enumerate(distinct_words))

def create_word_indices_for_text(text_list):
	input_words = []
	label_word = []
	for i in range(0,len(text_list) - CONTEXT):
		input_words.append((text_list[i:i+CONTEXT]))
		label_word.append((text_list[i+CONTEXT]))
	return input_words, label_word

with open(FILE, 'r+') as file_i:
	data = file_i.read()

text_list = data.lower().split()
for i in range(len(text_list)):
	if text_list[i] in distinct_words:
		continue
	else:
		text_list[i] = 'unknown'

# creating dataset, input_words are the context and the label_word is the word that has to be predicted
input_words, label_word = create_word_indices_for_text(text_list)
input_vectors = np.zeros((len(input_words), CONTEXT, NUM_WORDS), dtype=np.float32)
vectorized_labels = np.zeros((len(input_words), NUM_WORDS), dtype=np.float32)

for i, input_w in enumerate(input_words):
	for j, w in enumerate(input_w):
		input_vectors[i, j, word2index[w]] = 1
		vectorized_labels[i, word2index[label_word[i]]] = 1

test_index = np.random.randint(len(input_words))
test_words = input_words[test_index]
print(test_words)

complete_sentence = (" ".join(test_words))

print('\n\nNumber of words to predict:', NUM_PREDICTING)

for i in range(NUM_PREDICTING):
	# CONTEXT_ITER = CONTEXT + i
	print("\n\nGenerating test from test index %s with words %s: " % (test_index, test_words))
	input_for_test = np.zeros((1, CONTEXT, NUM_WORDS))
	for i, w in enumerate(test_words):
		input_for_test[0, i, word2index[w]] = 1
	predictions_all_matrix = model.predict(input_for_test, verbose = 0)[0]

	# select 5 words with the biggest probability to be matching the context
	top_5_indices = np.argsort(predictions_all_matrix)[-5:]
	top_5_probs = [0, 0, 0, 0, 0]
	for i in range(5):
		top_5_probs[i] = predictions_all_matrix[top_5_indices[i]]

	selected = select_one(top_5_indices, top_5_probs)

	predicted_word = index2word[selected]

	# append selected word to the answer; answer = context + predicted_words
	complete_sentence = complete_sentence + ' ' + predicted_word

# check if predicted words have END_OF_SENTENCE seqence and if they do cut the answer at the given
# spot with 0.5 probability
sentence_list = complete_sentence.split(' ')
predicted = sentence_list[len(sentence_list)-NUM_PREDICTING:]
original_sentence = sentence_list[:len(sentence_list)-NUM_PREDICTING]

cutoff_index = -1

for i in range(len(predicted)):
	if predicted[i][len(predicted[i])-2:] == END_OF_SENTENCE:
		# remove END_OF_SENTENCE symbol
		predicted[i] = remove_EOS(predicted[i])
		keep_prob = randint(0, 1)
		if keep_prob == 0: # remove the rest
			if cutoff_index == -1:
				cutoff_index = i
			# print('\nRemoving after:', predicted[i])
			# print('NEW predicted:', predicted[:i+1])
			break
		else:
			# print('\nLeaving:', predicted[i])
			continue

# if the answer was shortened because of END_OF_SENTENCE, shorten the original answer
if cutoff_index != -1:
	predicted = predicted[:cutoff_index + 1]

complete_sentence = " ".join(original_sentence) + ' ' + " ".join(predicted) + '.'

complete_sentence = complete_sentence.strip().split()

for i in complete_sentence:
	if i == 'unknown':
		complete_sentence.remove(i)

print("\nTHE COMPLETE RESULTING SENTENCE IS:", ' '.join(complete_sentence))

# Writing only results to the file
# result = (" ".join(test_words) + ' ' + predicted_word)
# SAVE_FILE.write(str(cycle) + ". THE COMPLETE RESULTING SENTENCE IS: " + str(result) + '\n')
# print("\nTHE COMPLETE RESULTING SENTENCE IS: %s %s" % (" ".join(test_words), predicted_word))
print()#put more NUM_CYCLES in if what you see here is gibberish

# complete_sentence = complete_sentence + ' ' + str(predicted_word)

# test_words.append(index2word[selected])
# test_words = shift(test_words, 1)
# test_words[9] = index2word[selected]

# print("Complete answer: " + complete_sentence + '\n')
