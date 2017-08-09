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
            - if `return_sequences`: 3D tensor with shape `(batch_size, timesteps, units+input_dim)`.
            - else, 2D tensor with shape `(batch_size, units)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention(64, activation='tanh'))
    """
    def __init__(self, use_context=True, return_sequences=True, **kwargs):
        self.use_context = use_context
        self.return_sequences = return_sequences
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
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
            # Appends matching context to each timestep (i.e. word) in a sequence
            new_contexts = []
            for i in range(contexts.shape[0]):
                context_i = K.transpose(contexts[i])
                new_contexts.append([context_i for _ in range(contexts.shape[1])])
            new_contexts = K.variable(value=np.array(new_contexts))
            return K.concatenate([new_contexts, inputs], axis=3)
        else:
            # Returns matching contexts for each sequence
            return contexts

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
model.add(Bidirectional(layer=GRU(HIDDEN_SIZE, return_sequences=True), merge_mode='concat'))
model.add(Attention(return_sequences=False ,units=2*HIDDEN_SIZE, activation='tanh'))
model.add(Dense(units=CHARS))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
print("All checks passed")
