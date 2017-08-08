import os
from string import punctuation
from operator import itemgetter

rootdir = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/Resources/wikidump'

# extraction of distinct words that appear more than once
dist_words = set()
distinct_words = set()
DEST = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/distinct_words.txt'

# with open(DEST, 'r+') as f_a:
#     d = f_a.read()
#
# print(d)
# distinct_words = set(d)
# print('num elem:', len(d))

# -*- coding: utf-8 -*-
def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

for subdir, dirs, files in os.walk(rootdir):
    name = str(subdir).split('/')[-1]
    print(str(subdir))

    c = {}
    for f in files:
        print('Working on file:', str(subdir).split('/')[-1] + '/' + str(f))
        SOURCE = str(subdir) + '/' + str(f)
        with open(SOURCE, 'r+') as file_i:
            for line in file_i:
                for word in line.lower().split():
                    key = word.rstrip(punctuation)
                    c[key] = c.get(key, 0) + 1

        for k, v in sorted(c.items(),key=itemgetter(1),reverse=True):
            if v <= 5:
                del c[k]

        words = set(c.keys())

        for i in words:
            if not isEnglish(i):
                del i

        dist_words = dist_words.union(words)

    if not os.path.isfile(DEST):
        with open(DEST, 'w+') as f_a:
            f_a.write(str(dist_words))
    else:
        with open(DEST, 'a+') as f_a:
            d = f_a.read().split()
            distinct_words = set(d)
            distinct_words = distinct_words.union(dist_words)
            f_a.write(str(distinct_words))



with open('/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/distinct_words.txt') as f:
    d = f.read()
    print(d)
    print('\n\nNumber of elements:', len(set(d)))
