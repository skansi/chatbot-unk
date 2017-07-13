from keras.layers import Dense, Activation, Flatten
from keras.layers.recurrent import SimpleRNN
from keras.models import Sequential
import numpy as np
from random import randint

# NUM_PREDICTING = 3
NUM_PREDICTING = randint(2, 5) # number of words generated
CONTEXT = 10
NUM_HIDDEN = 50
BATCH_SIZE = 60
NUM_CYCLES = 250
NUM_EPOCHS_PER_CYCLES = 50 #default 1

# shift the context list so that previously predicted word is in the context for the next prediction
def shift(l, n):
	return l[n:] + l[:n]

# select which word to predict based on the probability for the word to match the context
# this way we ensure that not always for the same context will the same word be selected
def select_one(choices, probs):
	# selecting one element from choices with probability p from probs
	# the type has to be changed from np.ndarray of size 1 to int in order to be used as an index
	the_one = np.asscalar(np.random.choice(choices, 1, probs))
	print("Choices:")
	for i in choices:
		print(index2word[i])
	print('\nSelected: ', index2word[the_one])
	return the_one

def create_text_from_file(textfile="/home/prometej/Workspaces/PythonWorkspace/project_chatbot/sample_changed.txt"):
	clean_text_chunks = []
	with open(textfile, 'r', encoding='utf-8') as text:
		for line in text:
			clean_text_chunks.append(line)
	clean_text = (" ".join(clean_text_chunks)).lower()
	text_as_list = clean_text.split()
	return text_as_list

text_list = create_text_from_file()
distinct_words = set(text_list)
NUM_WORDS = len(distinct_words)

word2index = dict((w, i) for i, w in enumerate(distinct_words))
index2word = dict((i, w) for i, w in enumerate(distinct_words))

def create_word_indices_for_text(text_list):
	input_words = []
	label_word = []
	for i in range(0,len(text_list) - CONTEXT):
		input_words.append((text_list[i:i+CONTEXT]))
		label_word.append((text_list[i+CONTEXT]))
	return input_words, label_word

input_words, label_word = create_word_indices_for_text(text_list)
input_vectors = np.zeros((len(input_words), CONTEXT, NUM_WORDS), dtype=np.float32)
vectorized_labels = np.zeros((len(input_words), NUM_WORDS), dtype=np.float32)

for i, input_w in enumerate(input_words):
	for j, w in enumerate(input_w):
		input_vectors[i, j, word2index[w]] = 1
		vectorized_labels[i, word2index[label_word[i]]] = 1

model = Sequential()
model.add(SimpleRNN(NUM_HIDDEN, return_sequences=False, input_shape=(CONTEXT, NUM_WORDS), unroll=True))
model.add(Dense(NUM_WORDS))
model.add(Activation("softmax"))
model.summary()

model.compile(loss="mean_squared_error", optimizer="sgd")

SAVE_FILE = open('/home/prometej/Workspaces/PythonWorkspace/project_chatbot/resulting_sentences.txt', 'w')

for cycle in range(NUM_CYCLES):
	print(">-<" * 44)
	print(" Cycle: %d" % (cycle+1))
	print(" Size of the answer:", str(10 + NUM_PREDICTING))
	model.fit(input_vectors, vectorized_labels, batch_size = BATCH_SIZE, epochs = NUM_EPOCHS_PER_CYCLES)
	test_index = np.random.randint(len(input_words))
	test_words = input_words[test_index]

# save model (architecture, weights, train_config(loss, optimizer), state_of_the_optimizer)
model.save('/home/prometej/Workspaces/PythonWorkspace/project_chatbot/model.h5')
del model

# # save model and weights
# model_json = model.to_json()
# open('chatbot_architecture.json', 'w+').write(model_json)
# model.save_weights('chatbot_weights.h5', overwrite=True)

	# complete_sentence = (" ".join(test_words))
	#
	# for i in range(NUM_PREDICTING):
	# 	# CONTEXT_ITER = CONTEXT + i
	# 	print("Generating test from test index %s with words %s: " % (test_index, test_words))
	# 	input_for_test = np.zeros((1, CONTEXT, NUM_WORDS))
	# 	for i, w in enumerate(test_words):
	# 		input_for_test[0, i, word2index[w]] = 1
	# 	predictions_all_matrix = model.predict(input_for_test, verbose = 0)[0]
	#
	# 	top_5_indices = np.argsort(predictions_all_matrix)[-5:]
	# 	top_5_probs = [0, 0, 0, 0, 0]
	# 	for i in range(5):
	# 		top_5_probs[i] = predictions_all_matrix[top_5_indices[i]]
	#
	# 	selected = select_one(top_5_indices, top_5_probs)
	#
	# 	predicted_word = index2word[selected]
	#
	# 	# Writing only results to the file
	# 	# result = (" ".join(test_words) + ' ' + predicted_word)
	# 	# SAVE_FILE.write(str(cycle) + ". THE COMPLETE RESULTING SENTENCE IS: " + str(result) + '\n')
	# 	print("\nTHE COMPLETE RESULTING SENTENCE IS: %s %s" % (" ".join(test_words), predicted_word))
	# 	print()#put more NUM_CYCLES in if what you see here is gibberish
	#
	# 	complete_sentence = complete_sentence + ' ' + str(predicted_word)
	#
	# 	# test_words.append(index2word[selected])
	# 	test_words = shift(test_words, 1)
	# 	test_words[9] = index2word[selected]
	#
	# print("Complete answer: " + complete_sentence + '\n')

# # extracting model and weights for saving
# merge_layer = model.layers[0]
# word_model = merge_layer.layers[0]
# word_embed_layer = word_model.layers[0]
# weights = word_embed_layer.get_weights()[0]

# # save model and weights
# model_json = model.to_json()
# open('chatbot_architecture.json', 'w+').write(model_json)
# model.save_weights('chatbot_weights.h5', overwrite=True)
