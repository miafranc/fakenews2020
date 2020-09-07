import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, initializers, regularizers
from sklearn.model_selection._split import StratifiedKFold
from keras import backend as K

from sklearn.model_selection import train_test_split

from cyclical_learning_rate import CyclicLR

from utils import ExperimentBase
from utils import scoring_functions

from nn.cnn_char import CNNChar
from nn.cnn_word import CNNWord
from nn.lstm import LSTM
from nn.blstm import BLSTM


class NNTC(ExperimentBase):
    
    def __init__(self):
        self.lr = 0.001
        self.num_epochs = 1000
        self.patience = 50
        self.batch_size = 50
        
        self.min_lr = 1e-5
        self.max_lr = 1e-3
        self.step_lr = 8.
        
        self.cv_k = 5
        self.cv_shuffle = False
        
        self.nnet = None
        self.model = None
    
    def build_model(self, texts_i, exps_i):
        if exps_i == 9: # Character-level CNN
            self.nnet = CNNChar(self.data, self.data_prop['length_chars'], texts_i)
        elif exps_i == 10: # Word-based CNN
            self.nnet = CNNWord(self.data, self.data_prop['length_words'], texts_i)
        elif exps_i == 11: # LSTM
            self.nnet = LSTM(self.data, self.data_prop['length_words'], texts_i)
        elif exps_i == 12: # BLSTM
            self.nnet = BLSTM(self.data, self.data_prop['length_words'], texts_i)
            
    def run(self, texts_i, exps_i):
        self.build_model(texts_i, exps_i)
        
        cv = StratifiedKFold(n_splits=self.cv_k, shuffle=self.cv_shuffle)
        scores = []
    
        X = self.nnet.getX()
        y = self.nnet.gety()
    
        for train_index, test_index in cv.split(X, y):
            X_train, X_test = X[train_index,:], X[test_index,:]
            y_train, y_test = y[train_index], y[test_index]
            
            X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train)
            
            self.model = self.nnet.build_model()
            print(self.model.summary())
             
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            clr = CyclicLR(base_lr=self.min_lr, max_lr=self.max_lr, mode='triangular', 
                           step_size=self.step_lr * X_tr.shape[0] / self.batch_size)
             
            self.model.fit(X_tr, y_tr, epochs=self.num_epochs,
                           batch_size=self.batch_size,
                           validation_data=(X_val, y_val),
                           callbacks=[clr,
                                      callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=self.patience, verbose=True, restore_best_weights=True)]
                           )
             
            y_pred = (self.model.predict(X_test) > 0.5).astype("int32")
            
            K.clear_session()
             
            score = [f(y_test, y_pred) for f in scoring_functions]
            scores.append(score)
            print("-> SCORES: {}".format(scores))
            
#             break # only one iteration
    
        means = np.mean(scores, axis=0)
        stds = np.std(scores, axis=0)
    
        for (i, f) in enumerate(scoring_functions):
            print("\t{:20}: {:.6f} (+/- {:.6f})".format(f.__name__, means[i], stds[i]))
            

if __name__ == "__main__":
    pass
