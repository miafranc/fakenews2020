import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from collections import Counter

from keras import backend as K

from utils import word_tokenizer
from utils import NNBase


class AttentionLayer(tf.keras.layers.Layer):
    
    def __init__(self, dim):
        super(AttentionLayer, self).__init__()
        self.w = tf.Variable(
            initial_value=K.random_normal([dim,1], stddev=1.),
            trainable=True
        )
        
    def call(self, inputs):
        return tf.matmul(inputs, self.w)

class BLSTM(NNBase):
    
    def __init__(self, data, text_length, texts_i):
        self.embedding_dim = 100
        self.data = data
        self.text_length = text_length
        self.texts_i = texts_i
        self.word_map = self._get_words(self.data)
    
    def _get_words(self, data):
        if self.texts_i == 0:
            X = [d['text'] for d in data]
        elif self.texts_i == 1:
            X = [d['title'] for d in data]
        elif self.texts_i == 2:
            X = [d['title'] + ' ' + d['text'] for d in data]
        counts = Counter(word_tokenizer(" ".join(X)))
        return {w:(i+1) for i, w in enumerate(counts.keys())} 
    
    def _map_text(self, text, word_map, text_length):
        v = np.zeros((text_length,))
        tokens = word_tokenizer(text)
        for i, c in enumerate(tokens[:text_length]):
            v[i] = word_map.get(c, 0)
        return v

    def getX(self):
        if self.texts_i == 0:
            X = np.asarray([self._map_text(d['text'], self.word_map, self.text_length) for d in self.data])
        elif self.texts_i == 1:
            X = np.asarray([self._map_text(d['title'], self.word_map, self.text_length) for d in self.data])
        elif self.texts_i == 2:
            X = np.asarray([self._map_text(d['title'] + ' ' + d['text'], self.word_map, self.text_length) for d in self.data])
        return X
    
    def gety(self):
        y = np.asarray([(d['label']+1)/2 for d in self.data])
        return y
        
    def build_model(self):
        input_layer = layers.Input(shape=(self.text_length,))
        emb_layer = layers.Embedding(input_dim=len(self.word_map)+1, output_dim=self.embedding_dim, input_length=self.text_length)(input_layer)
        
        H = layers.Bidirectional(layers.LSTM(100, return_sequences=True), merge_mode='sum')(emb_layer)
        M = layers.Activation('tanh')(H)
        
        w = AttentionLayer(M.shape[2])(M)
        
        a = layers.Activation('softmax')(w)
        r = layers.Dot(axes=(1, 1))([H, a])
        h = layers.Activation('tanh')(r)
        
        layer = layers.Flatten()(h)
        layer = layers.Dense(1, activation='sigmoid')(layer)
        
        model = tf.keras.Model(inputs=[input_layer], outputs=[layer])
        
        return model
        