from keras.models import Sequential
from keras.layers.recurrent import *
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.core import *
from keras import backend as K
#from layers import AttentionWithContext 
#from keras.engine.topology import Layer
#import numpy as np

class Attention(Dense):
    """
        Attention operation, with an optional context/query vector, for temporal data.
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The number of units has to be the same as the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention(64, activation='tanh'))
    """
    def __init__(self, use_context=True, **kwargs):
        self.use_context = use_context
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        if self.use_context:
            self.context = self.add_weight(name='context', 
                                      shape=(self.units,1), # (self.units,)
                                      initializer='glorot_uniform',
                                      trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, inputs):
        output = super(Attention, self).call(inputs)
        # Implements soft attention mechanism
        if self.use_context:
            output = K.dot(output, self.context)
            #output = K.reshape(output, output.shape[:-1])
        weights = K.softmax(output)
        return K.sum(weights*inputs, axis=1) # dim reduction step
        """
        # MemN2N approach
        match = dot([output, self.context], axes=(2, 2))
        match = Activation('softmax')(match)
        response = add([match, inputs])
        response = Permute((2,1))(response) # (samples, query_maxlen, story_maxlen)
        """
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

HIDDEN_SIZE = 50
BATCH_SIZE = 128
CHARS = 44

"""
Our message-level sentiment analysis model
(MSA) consists of a 2-layer bidirectional LSTM
(BiLSTM) with an attention mechanism, for identifying
the most informative words.
"""
print('Build model...')
model = Sequential()
model.add(Embedding(input_dim=300, output_dim=HIDDEN_SIZE))
# GRU instead of LSTM
model.add(Bidirectional(layer=GRU(HIDDEN_SIZE, return_sequences=True), merge_mode='concat'))
#model.add(AttentionWithContext()) # word attention
model.add(Bidirectional(layer=GRU(HIDDEN_SIZE, return_sequences=True), merge_mode='concat')) # TimeDistributed???
#model.add(AttentionWithContext()) # sentence attention
model.add(Attention(units=2*HIDDEN_SIZE, activation='tanh'))
model.add(Dense(units=CHARS))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
print("All checks passed")
