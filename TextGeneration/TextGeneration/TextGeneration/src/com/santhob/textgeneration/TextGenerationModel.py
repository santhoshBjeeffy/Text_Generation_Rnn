'''
Created on Nov 17, 2018

@author: santhosh
'''

import tensorflow as tf
import functools
import os
from com.santhob.textgeneration import TextDataConverter as TDC


class TextGenModel(object):
    uniqueCharsSize = None
    embedding_dim = None
    rnn_units = None
    batch_size = None

    def __init__(self, uniqueCharsSize, embedding_dim,
                 rnn_units,
                 batch_size):
        self.uniqueCharsSize = uniqueCharsSize
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.batch_size = batch_size
        
    def buildModel(self):
        rnn = None
        if tf.test.is_gpu_available():
            rnn = tf.keras.layers.CuDNNGRU
        else:
            rnn = functools.partial(tf.keras.layers.GRU, recurrent_activation='sigmoid')

        embedding_layer = tf.keras.layers.Embedding(self.uniqueCharsSize,
                                                   self.embedding_dim,
                              batch_input_shape=[self.batch_size, None])
        rnn_layer = rnn(self.rnn_units, return_sequences=True,
                        recurrent_initializer='glorot_uniform',
                        stateful=True)
        dense_layer = tf.keras.layers.Dense(self.uniqueCharsSize);
        
        model = tf.keras.Sequential([embedding_layer, rnn_layer,
                                     dense_layer ])
        return model


class TrainModelWithDataSet(object):
    model = None
    dataset = None

    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
    
    #textDataLen = Total Number of chars in text file/Source
    def trainModelWithEpoch(self, epochNumber, textDataLen):
        self.model.compile(optimizer=tf.train.AdamOptimizer(),
                           loss=tf.losses.sparse_softmax_cross_entropy)
        checkpoint_dir = '/home/santhob/ProjectWorkspace/Dataset/training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
        checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                                               save_weights_only=True)
        history = self.model.fit(self.dataset.repeat(), epochs=epochNumber, steps_per_epoch=textDataLen, callbacks=[checkpoint_callback])
        
        return history
    
    def genrateText(self, start_string, char2idx, idx2char):
        # Evaluation step (generating text using the learned model)

        # Number of characters to generate
        num_generate = 1000
        
        # You can change the start string to experiment
        #start_string = 'ROMEO'
        
        # Converting our start string to numbers (vectorizing) 
        input_eval = [char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)
        
        # Empty string to store our results
        text_generated = []
        
        # Low temperatures results in more predictable text.
        # Higher temperatures results in more surprising text.
        # Experiment to find the best setting.
        temperature = 1.0
        
        # Here batch size == 1
        self.model.reset_states()
        for i in range(num_generate):
            predictions = self.model(input_eval)
            # remove the batch dimension
            predictions = tf.squeeze(predictions, 0)
        
            # using a multinomial distribution to predict the word returned by the model
            predictions = predictions / temperature
            predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()
              
            # We pass the predicted word as the next input to the model
            # along with the previous hidden state
            input_eval = tf.expand_dims([predicted_id], 0)
            text_generated.append(idx2char[predicted_id])
        
        return (start_string + ''.join(text_generated))
