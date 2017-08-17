import os
import pickle

ROOTDIR = '/home/novak_luka93/wikidump/'
DEST = '/home/novak_luka93/chatbot-unk/word_based_version/words_vocab.pkl'
N_MOST_COMMON = 20000

words = {}

for subdir, dirs, files in os.walk(ROOTDIR):
    name = str(subdir).split('/')[-1]
    print(subdir)
    list_files = files

    for f in files:
        print('Working on file:', f)
        SOURCE = str(subdir) + '/' + str(f)

        # load text and covert to lowercase
        text = open(SOURCE, encoding='utf-8').read()
        text = text.lower()
        print('Size:', len(text))

        text_as_list = text.split(' ')

        for word in text_as_list:
            words[word] = words.setdefault(word, 0) + 1

words_sorted = sorted(words.keys(), key=words.get, reverse=True)

words_top_n = words_sorted[:N_MOST_COMMON]

print('Top n words:', words_top_n)
print('N: ', len(words_top_n))
print('len(words):', len(words.keys()))

with open(DEST, 'wb+') as d:
    pickle.dump(words_top_n, d)
