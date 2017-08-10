from keras.models import Sequential
from keras.layers.recurrent import *
from keras.layers.embeddings import *
from keras.layers.wrappers import *
from keras.layers.core import *
from keras import backend as K
import numpy as np

class Attention(Dense):
    """
        Attention operation, with an optional context/query vector, for temporal data.
        # Arguments
            use_context: Boolean. Whether to use an optional context vector
            return_sequences: Boolean. Whether to return the contexts only,
             or the full sequence with concatenated matching contexts.
        # Input shape
            3D tensor with shape: `(batch_size, timesteps, input_dim)`.
        # Output shape
            - if `return_sequences`: 3D tensor with shape `(batch_size, timesteps, 2*input_dim)`.
            - else, 2D tensor with shape `(batch_size, input_dim)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True,
        and units=output_dim from the previous RNN layer.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention(64, activation='tanh'))
    """
    def __init__(self, use_context=True, return_sequences=True, **kwargs):
        self.use_context = use_context
        self.return_sequences = return_sequences
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        assert input_shape[-1] == self.units
        # Create a trainable weight variable for this layer.
        if self.use_context:
            self.context = self.add_weight(name='context', 
                                      shape=(self.units,1),
                                      initializer='glorot_uniform',
                                      trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, inputs):
        outputs = super(Attention, self).call(inputs)
        # Implements a soft attention mechanism
        if self.use_context:
            outputs = K.dot(outputs, self.context)
        weights = K.softmax(outputs)
        contexts = K.sum(weights*inputs, axis=1)

        if self.return_sequences:
            # non functional!!!
            # Returns matching context appended to each timestep (i.e. word) in a sequence
            input_shape = K.int_shape(inputs)
            contexts = K.repeat_elements(contexts, input_shape[1], axis=0)
            contexts = K.reshape(contexts, input_shape)
            return K.concatenate([contexts, inputs])
        else:
            # Returns matching contexts for each input sequence
            return contexts

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 3
        batch_size, timesteps, input_dim = input_shape

        if self.return_sequences:
            output_shape = (batch_size, timesteps, 2*self.input_dim)
        else:
            output_shape = (batch_size, input_dim)
        return output_shape

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
model.add(Bidirectional(layer=GRU(HIDDEN_SIZE, return_sequences=True), merge_mode='concat'))
model.add(Attention(return_sequences=False, units=2*HIDDEN_SIZE, activation='tanh'))
#model.add(GRU(units=HIDDEN_SIZE))
model.add(Dense(units=CHARS))  # add Timedistributed wrapper?
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
print("All checks passed")
