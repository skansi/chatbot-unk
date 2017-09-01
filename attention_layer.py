import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers.recurrent import *
from keras.layers.embeddings import *
from keras.layers.wrappers import *
from keras.layers.core import *

class Attention(Dense):
    """
    Keras Layer that implements an Attention mechanism,
    with an optional context/query vector,for temporal data.
    Supports Masking.
    Follows the work of Baziotis et al. [aclweb.org/anthology/S17-2126]
    # Input shape
        3D tensor with shape `(batch_size, timesteps, features)`
    # Output shape
        2D tensor with shape `(batch_size, features)`
    :param kwargs:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Example:
        model.add(GRU(100, return_sequences=True))
        model.add(Attention(activation='tanh'))
    """
    def __init__(self, use_context=False, **kwargs):
        self.use_context = use_context
        super(Attention, self).__init__(units=1, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.units = input_shape[-1]
        if self.use_context:
            self.context = self.add_weight(name='context',
                                      shape=(self.units,),
                                      initializer='glorot_uniform',
                                      trainable=True)
        else:
            self.context = None
        super(Attention, self).build(input_shape)

    def call(self, inputs):
        outputs = super(Attention, self).call(inputs)
        if self.use_context:
            outputs = K.squeeze(K.dot(outputs, K.expand_dims(self.context)), axis=-1)
        weights = K.softmax(outputs)
        return K.sum(inputs*weights, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

if __name__ =='__main__':

    HIDDEN_SIZE = 300
    BATCH_SIZE = 32
    VOCAB_SIZE = 10000
    CONTEXT = 100
    INPUT_SHAPE = (CONTEXT, VOCAB_SIZE)

    """
    The deep attentive language model
    consists of a 2-layer bidirectional GRU
    with an attention mechanism for identifying
    the most informative words.
    """
    model = Sequential()
    #model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=HIDDEN_SIZE, input_shape=INPUT_SHAPE))
    model.add(Bidirectional(layer=GRU(HIDDEN_SIZE, return_sequences=True), merge_mode='concat', input_shape=INPUT_SHAPE))
    model.add(Dropout(0.3))
    model.add(Bidirectional(layer=GRU(HIDDEN_SIZE, return_sequences=True), merge_mode='concat'))
    model.add(Dropout(0.3))
    model.add(Attention(activation='tanh'))
    model.add(Dense(units=VOCAB_SIZE, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
