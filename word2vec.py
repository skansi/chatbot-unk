# -*- coding: utf-8 -*-
from gensim.models import word2vec
import os
import logging

class Text8Sentences(object):
    def __init__(self, fname, maxlen):
        self.fname = fname
        self.maxlen = maxlen

    def __iter__(self):
        with open(os.path.join(DATA_DIR, "text8"), "r") as ftext:
            text = ftext.read().split(" ")
            words = []
            for word in text:
                if len(words) >= self.maxlen:
                    yield words
                    words = []
                words.append(word)
            yield words

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

DATA_DIR = "/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/Resources/text8/"
sentences = Text8Sentences(os.path.join(DATA_DIR, "text8"), 50)
model = word2vec.Word2Vec(sentences, size=300, min_count=30, sg=1)

# saving the model
# it can be loaded by: model = Word2Vec.load("/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/word2vec_gensim.bin")
# model.init_sims(replace=True)
# word_vectors = model.wv
model.wv.save_word2vec_format("/home/prometej/Workspaces/PythonWorkspace/chatbot-unk/word2vec_gensim_skipgram.bin", binary=True)
