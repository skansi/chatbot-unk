from hyphen import Hyphenator, dict_info
from hyphen.dictools import *
import os
import pickle

ROOTDIR = '/home/novak_luka93/wikidump/'
DEST = '/home/novak_luka93/chatbot-unk/character_based_version/v1.1/syllable_vocab.pkl'
N_MOST_COMMON = 20000

syllables = {}

for subdir, dirs, files in os.walk(ROOTDIR):
    name = str(subdir).split('/')[-1]
    print(subdir)
    list_files = files

    h_en = Hyphenator('en_US')

    for f in files:
        print('Working on file:', f)
        SOURCE = str(subdir) + '/' + str(f)

        # load text and covert to lowercase
        text = open(SOURCE, encoding='utf-8').read()
        text = text.lower()
        print('Size:', len(text))

        text_as_list = text.split(' ')

        for word in text_as_list:
            try:
                l = h_en.syllables(word)
                for s in l:
                    if l == []:
                        s = ' '
                    if l.index(s) == (len(l) - 1):
                        s = s + ' '
                    syllables[s] = syllables.setdefault(s, 0) + 1
            except ValueError:
                print(word)

syllables_sorted = sorted(syllables.keys(), key=syllables.get, reverse=True)

syllables_top_n = syllables_sorted[:N_MOST_COMMON]

print('Top n syllables:', syllables_top_n)
print('N: ', len(syllables_top_n))
print('len(syllables):', len(syllables.keys()))

with open(DEST, 'wb+') as d:
    pickle.dump(syllables_top_n, d)
