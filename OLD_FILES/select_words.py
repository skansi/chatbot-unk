import os
from string import punctuation
from operator import itemgetter

from multiprocessing import Process
from collections import Counter

def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def proc(SRC, DEST, TOP_N_WORDS):

    print('Process started:', os.getpid())

    c = {}
    for subdir, dirs, files in os.walk(SRC):
        name = str(subdir).split('/')[-1]

        print('Process \'' + str(os.getpid()) + '\' working on directory: ' + str(name))

        for f in files:
            # print('Working on file:', str(subdir).split('/')[-1] + '/' + str(f))
            SOURCE = str(subdir) + '/' + str(f)
            with open(SOURCE, 'r+') as file_i:
                for line in file_i:
                    for word in line.lower().split():
                        key = word.rstrip(punctuation)
                        c[key] = c.get(key, 0) + 1

    words = []

    d = Counter(c)

    for k, v in d.most_common(TOP_N_WORDS):
        words.append(k)

    for i in words:
        if not isEnglish(i):
            del i

    with open(DEST, 'w+') as f_a:
        f_a.write(str(words))

if __name__ == '__main__':
    p_a = Process(target=proc, args=('/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/Resources/wikidump/A', '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/distinct_words_A.txt', 30000))
    p_b = Process(target=proc, args=('/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/Resources/wikidump/B', '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/distinct_words_B.txt', 30000))
    p_c = Process(target=proc, args=('/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/Resources/wikidump/C', '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/distinct_words_C.txt', 30000))
    p_d = Process(target=proc, args=('/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/Resources/wikidump/D', '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/distinct_words_D.txt', 30000))
    p_e = Process(target=proc, args=('/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/Resources/wikidump/E', '/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/distinct_words_E.txt', 15000))

    # start all processes
    p_a.start()
    p_b.start()
    p_c.start()
    p_d.start()
    p_e.start()

    # wait for all processes to finish
    p_a.join()
    p_b.join()
    p_c.join()
    p_d.join()
    p_e.join()
