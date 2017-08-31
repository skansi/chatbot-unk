import numpy as np
import os
import operator

ROOTDIR = '/home/prometej/Workspaces/PythonWorkspace/Resources/wikidump_pom'
VOCABULARY_LEMMATIZED = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/word_based_version/file_resources/words_lemmatized_vocab.pkl'

# list of all allowed words
with open(VOCABULARY_LEMMATIZED, 'rb') as v:
    VOCAB = pickle.load(v)

VOCAB = sorted(VOCAB.items(), key = operator.itemgetter(1))

with open('/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/word_based_version/file_resources/vocab2', 'r') as v:
    VOCAB = eval(v.read())

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

        for k in range(len(text_list)):
            text_list[k] = ''.join([i for i in text_list[k] if i in VOCAB])


        for i in text_list:
            if i == '$#':
                flag = np.random.choice(2, 1, p=[0.2, 0.8])
                if flag == 1:
                    text_list.remove(i)

        data = ' '.join(text_list)

        with open(SOURCE, 'w+') as s:
            s.write(data)
