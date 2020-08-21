import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from utils import NNBase


class CNNChar(NNBase):
    
    def __init__(self, data, text_length, texts_i):
        self.embedding_dim = 100
        self.chars = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\"'/\|_@#$%^&*~`+=<>()[]{} \n"
        self.char_map = {self.chars[i]:i+1 for i in range(len(self.chars))}
        self.data = data
        self.text_length = text_length
        self.texts_i = texts_i
        
    def _map_text(self, text):
        v = np.zeros((self.text_length,))
        for i, c in enumerate(text.lower()[:self.text_length]):
            v[i] = self.char_map.get(c, 0)
        return v
    
    def getX(self):
        if self.texts_i == 0:
            X = np.asarray([self._map_text(d['text']) for d in self.data])
        elif self.texts_i == 1:
            X = np.asarray([self._map_text(d['title']) for d in self.data])
        elif self.texts_i == 2:
            X = np.asarray([self._map_text(d['title'] + ' ' + d['text']) for d in self.data])
        return X
    
    def gety(self):
        y = np.asarray([(d['label']+1)/2 for d in self.data])
        return y
    
    def build_model(self):
        kernels = [256] * 6
        kernel_sizes = [7, 7, 3, 3, 3, 3]
        pool_sizes = [3, 3, 0, 0, 0, 3]
        neurons = [1024, 1024]
        dropouts = [0.5, 0.5]
         
        input_layer = layers.Input(shape=(self.text_length))
        layer = layers.Embedding(input_dim=len(self.chars)+1, output_dim=self.embedding_dim, input_length=self.text_length)(input_layer)
         
        for i, num_kernels, kernel_size, pool_size in zip(range(len(kernels)), kernels, kernel_sizes, pool_sizes):
            if i == 0:
                layer = layers.Conv1D(num_kernels, kernel_size, input_shape=(self.text_length, len(self.chars)))(layer)
            else:
                layer = layers.Conv1D(num_kernels, kernel_size)(layer)
            layer = layers.Activation('relu')(layer)
            if pool_size > 0:
                layer = layers.MaxPooling1D(pool_size)(layer)
             
        layer = layers.Flatten()(layer)
         
        for nn, do in zip(neurons, dropouts):
            layer = layers.Dense(nn, activation='relu')(layer)
            layer = layers.Dropout(do)(layer)
         
        layer = layers.Dense(1, activation='sigmoid')(layer)
        model = tf.keras.Model(inputs=[input_layer], outputs=[layer])
        
        return model
