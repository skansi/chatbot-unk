import os
import pickle
from nltk.stem import WordNetLemmatizer

ROOTDIR = '/home/prometej/Workspaces/PythonWorkspace/Resources/wikidump'
DEST_L = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/word_based_version/file_resources/words_lemmatized_vocab.pkl'

lemmatizer = WordNetLemmatizer()
words_lemma = {}

for subdir, dirs, files in os.walk(ROOTDIR):
    name = str(subdir).split('/')[-1]
    print(subdir)
    list_files = files

    for f in files:
        print('Working on file:', f)
        SOURCE = str(subdir) + '/' + str(f)

        # load text and covert to lowercase
        with open(SOURCE, 'r') as f:
            text = f.read()
        text = text.lower()
        text_as_list = text.split(' ')
        print('Size:', len(text_as_list))

        for word in text_as_list:
            w_l = lemmatizer.lemmatize(word)
            words_lemma[w_l] = words_lemma.setdefault(word, 0) + 1

words_lemma_sorted = sorted(words_lemma.keys(), key=words_lemma.get, reverse=True)

print('Number of distinct lemmatized words: ', len(words_lemma))

with open(DEST_L, 'wb+') as d:
    pickle.dump(words_lemma, d)
