# -*- coding: utf-8 -*-
from gensim.models import word2vec
import os
import logging

rootdir ='/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/Resources/wikidump'
sentences_list = []
l = []

# fist train set
with open ('/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/Resources/wikidump/AA/wiki_00') as f:
    data = f.read()

sen_list = []
s = data.split('\$\#')
for s_i in s:
    sen = s_i.split(' ')
    while '' in sen:
        sen.remove('')
    sen_list += sen

model = word2vec.Word2Vec(sentences=sen_list, size=100, window=5, sg=0, min_count=1)
model.save('/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/wiki_model')

for subdir, dirs, files in os.walk(rootdir):
    print(subdir)
    name = str(subdir).split('/')[-1]
    l.append(str(name))
    for f in files:
        print('Working on file:', f)
        SOURCE = str(subdir) + '/' + str(f)
        with open(SOURCE, 'r+') as file_i:
            data = file_i.read()

        sentences = data.split('\$\#')
        for sentence in sentences:
            sentence = sentence.split(' ')
            while '' in sentence:
                sentence.remove('')
            sentences_list += sentence

        model = word2vec.Word2Vec.load('/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/wiki_model')
        model.train(sentences_list, total_words=len(sentences_list), epochs=1)

        model.save('/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/wiki_model')
        sentences_list = []

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#
# # DATA_DIR = "/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/Resources/text8/"
# # sentences = Text8Sentences(os.path.join(DATA_DIR, "text8"), 50)
#
# # sg=0 (default) is CBOW model and sg=1 is skipgram model
# model = word2vec.Word2Vec(sentences=sentences_list, size=100, alpha=0.025, window=5, sg=1)

# saving the model
# it can be loaded by: model = Word2Vec.load("/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/word2vec_gensim.bin")
# model.init_sims(replace=True)
# word_vectors = model.wv
# model.wv.save_word2vec_format("/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/word2vec_gensim_skipgram.bin", binary=True)
