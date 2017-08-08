import os
import re

DEST = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/distinct_words.txt'

p_a = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/distinct_words_A.txt'
p_b = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/distinct_words_B.txt'
p_c = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/distinct_words_C.txt'
p_d = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/distinct_words_D.txt'
p_e = '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/distinct_words_E.txt'

distinct_words = set()

for i in [p_a, p_b, p_c, p_d, p_e]:
    with open(i, 'r+') as f:
        d = f.read().split()
        dist = set(d)
        distinct_words = distinct_words.union(dist)

p = re.compile('\"')
d = p.sub('', str(distinct_words))
r = re.compile('\{')
q = re.compile('\}')
t = re.compile('\'')
d = r.sub('', d)
d = q.sub('', d)
d = t.sub('', d)

s = re.compile(',,')
d = s.sub(',', d)

for i in range(len(d)):
    if d[i] == '':
        del d[i]

with open(DEST, 'w+') as dest:
    dest.write(str(d))
