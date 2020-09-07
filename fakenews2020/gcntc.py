import codecs
import pickle
import json
from collections import Counter
import numpy as np

from scipy.sparse import csr_matrix
import scipy

import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection._split import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras import backend as K

import os.path

from cyclical_learning_rate import CyclicLR

from utils import word_tokenizer
from utils import ExperimentBase
from utils import scoring_functions
from utils import binary_crossentropy, sparse_dropout


DIR_GRAPH = 'gcn'  # Local directory to store the generated graphs.

class WordDocGraphs:

    def __init__(self, data, texts_i):
        self.data = data
        self.texts_i = texts_i

    def tokenize(self, d):
        if self.texts_i == 0:
            return word_tokenizer(d['text'], lowercase=True)
        elif self.texts_i == 1:
            return word_tokenizer(d['title'], lowercase=True)
        elif self.texts_i == 2:
            return word_tokenizer(d['title'] + ' ' + d['text'], lowercase=True)

    def get_vocabulary(self, start_index=0):
        if self.texts_i == 0:
            texts = " ".join([d['text'] for d in self.data])
        elif self.texts_i == 1:
            texts = " ".join([d['title'] for d in self.data])
        elif self.texts_i == 2:
            texts = " ".join([d['title'] + ' ' + d['text'] for d in self.data])
        tokens = list(set(word_tokenizer(texts, lowercase=True)))
        return {tokens[i]:i+start_index for i in range(len(tokens))}

    def get_doc_map(self, start_index=0):
        return {d['id']:i+start_index for i, d in enumerate(self.data)}
    
    def get_df(self):
        """DF for calculating IDF
        """
        word_list = []
        for d in self.data:
            word_list.extend(list(set(self.tokenize(d))))
        return Counter(word_list)
    
    def sorted_tuple(self, x, y):
        return tuple(sorted((x, y)))
    
    def pmi_window(self, window):
        return [self.sorted_tuple(window[i], window[j]) for i in range(len(window)-1) for j in range(i+1, len(window))]
    
    def word_cooccurrences(self, window_size=20):
        coocc = {}
        occ = {}
        w_num = 0
        for d in self.data:
            text = self.tokenize(d)
            for window in [text[i:i+window_size] for i in range(len(text)-window_size+1)]:
                for sp in self.pmi_window(window):
                    coocc[sp] = coocc.get(sp, 0) + 1
                for w in window:
                    occ[w] = occ.get(w, 0) + 1
                w_num += 1
            
        return (coocc, occ, w_num)
    
    def build_word_word_graph(self, word_map, window_size=20):
        rows = []
        cols = []
        vals = []
        
        coocc, occ, w_num = self.word_cooccurrences(window_size)
        w_num = float(w_num)
        for w1, w2 in coocc.keys():
            if w1 != w2:
                p = np.log2((coocc[(w1, w2)] * w_num) / (occ[w1] * occ[w2]))
                if p > 0:
                    rows.append(word_map[w1])
                    cols.append(word_map[w2])
                    vals.append(p)
                    # symmetrize:
                    rows.append(word_map[w2])
                    cols.append(word_map[w1])
                    vals.append(p)
        
        return rows, cols, vals
    
    def build_doc_word_graph(self, word_map, doc_map):
        """Rows: doc_id, cols: word_id, vals: tf*idf
        """
        rows = []
        cols = []
        vals = []
        
        idf = self.get_df()
        N = len(self.data)
        for d in self.data:
            tf = Counter(self.tokenize(d))
            for w in tf.keys():
                rows.append(doc_map[d['id']])
                cols.append(word_map[w])
                vals.append(tf[w] * np.log2(N / idf[w]))
                # symmetrize:
                rows.append(word_map[w])
                cols.append(doc_map[d['id']])
                vals.append(tf[w] * np.log2(N / idf[w]))
            
        return rows, cols, vals

    def normalize(self, A):
        """A = D^-0.5 * A * D^-0.5
        """
        D = np.asarray(np.sum(A, axis=0)).flatten() ** -0.5
#         print(D)
        nonzeros = A.nonzero()
        for i, r in enumerate(nonzeros[0]):
            A[r, nonzeros[1][i]] *= D[r] * D[nonzeros[1][i]]

def build_graph(data, data_prop, texts_i):
    if os.path.isfile(os.path.join(DIR_GRAPH, '_A_' + data_prop['abbrv'].lower())):
        print('File already exists!')
        return
    
    g = WordDocGraphs(data, texts_i)
    
    V = g.get_vocabulary()
    print("Word count={}".format(len(V)))
    G = g.build_word_word_graph(V, window_size=20)
    print("Nonzeros={}".format(len(G[0])))
    D = g.get_doc_map(start_index=len(V))
    print("Doc count={}".format(len(D)))
    G2 = g.build_doc_word_graph(V, D)
    A = csr_matrix((G[2]+G2[2], (G[0]+G2[0], G[1]+G2[1])), dtype=np.float64)
    print("Total nonzeros={}".format(A.count_nonzero()))
    
    g.normalize(A)
     
    f = codecs.open(os.path.join(DIR_GRAPH, '_A_' + data_prop['abbrv'].lower()), 'wb')
    pickle.dump(A, f)
    f.close()
    
    f = codecs.open(os.path.join(DIR_GRAPH, '_A_meta_' + data_prop['abbrv'].lower()), 'w')
    json.dump({'word_count': len(V),
               'doc_count': len(D)}, f, indent=2)
    f.close()    

def load_graph(data, data_prop, texts_i):
    build_graph(data, data_prop, texts_i)
    
    f = codecs.open(os.path.join(DIR_GRAPH, '_A_' + data_prop['abbrv'].lower()), 'rb')
    A = pickle.load(f)
    f.close()
    
    f = codecs.open(os.path.join(DIR_GRAPH, '_A_meta_' + data_prop['abbrv'].lower()), 'r')
    meta = json.load(f)
    f.close()

    return (A, meta)

class GCL(tf.keras.layers.Layer):
    
    def __init__(self, input_dim, output_dim, bias=True):
        super(GCL, self).__init__()
        self.W = tf.Variable(
            initial_value=K.random_normal(shape=(input_dim, output_dim), stddev=1.),
            dtype='float32',
            trainable=True
        )
        self.b = tf.Variable(
            initial_value=K.zeros(shape=(output_dim,)), 
            dtype='float32',
            trainable=True
        )
        self.bias = bias

    def call(self, inputs):
        """
        inputs[0]: X (features)
        inputs[1]: A (adj. matrix)
        """
        if self.bias:
            return K.dot(inputs[1], K.dot(inputs[0], self.W)) + self.b
        return K.dot(inputs[1], K.dot(inputs[0], self.W))

class GCNTC(ExperimentBase):
    
    def __init__(self):
        self.lr = 0.02
        self.num_epochs = 200
        self.patience = 10

        self.min_lr = 1e-5
        self.max_lr = 1e-3
        self.step_lr = 8.
        
        self.cv_k = 5
        self.cv_shuffle = False
        
        self.nnet = None
        self.model = None

        self.embedding_size = 200
    
    def build_model(self, n): # n = X.shape[0]
        G = layers.Input(shape=(None,), sparse=True)
        
        X_in = layers.Input((n,), sparse=True)
#         H = tf.keras.layers.Lambda(lambda x: sparse_dropout(x, 0.5, (n,)))(X_in)
        H = GCL(n, self.embedding_size, bias=True)([X_in, G])
        H = layers.Activation('relu')(H)
        H = layers.Dropout(0.5)(H)
        H = GCL(self.embedding_size, 1, bias=True)([H, G])
        
        out = layers.Activation('sigmoid')(H)
        model = tf.keras.Model(inputs=[X_in, G], outputs=[out])
        
        return model
    
    def run(self, texts_i, exps_i):
        A, meta = load_graph(self.data, self.data_prop, texts_i)
        
        y = np.asarray([(d['label']+1)/2 for d in self.data])  # true labels
        
        X = csr_matrix(scipy.sparse.eye(meta['word_count'] + meta['doc_count']))  # identity matrix
        X_unlab = X[:meta['word_count'],:]
        X_lab = X[meta['word_count']:,:]
        y_ok = np.concatenate(([0]*X_unlab.shape[0], y))  # "all" labels (0 for the [unlabeled] words -- these will be masked out)
        
        cv = StratifiedKFold(n_splits=self.cv_k, shuffle=self.cv_shuffle)
        
        scores = []
    
        for train_index, test_index in cv.split(X_lab, y):
            train_index += meta['word_count']  # because only documents have labels (words come first though)
            test_index += meta['word_count']
            
            tr_index, val_index, y_tr, y_val = train_test_split(train_index, y_ok[train_index], test_size=0.25, stratify=y_ok[train_index])
            
            train_mask = np.array([1 if i in tr_index else 0 for i in range(X.shape[0])])
            
            self.model = self.build_model(meta['word_count'] + meta['doc_count'])
            
            print(self.model.summary())
            
            opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
            self.model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

            best_val_loss = 10**10  # best loss so far
            best_preds = []  # best predictions (for evaluation)
            wait = 0
            
            for epoch in range(self.num_epochs):
                print('Epoch {}/{}'.format(epoch+1, self.num_epochs))
                
                self.model.fit([X, A], y_ok, batch_size=X.shape[0], epochs=1, sample_weight=train_mask, shuffle=False, verbose=1)
                
                y_pred = self.model.predict([X, A], batch_size=X.shape[0])  # predictions
                
                y_pred_labels = (y_pred > 0.5).astype('int32')  # binary class labels (0 and 1) 
                
                train_acc = accuracy_score(y_ok[tr_index], y_pred_labels[tr_index])
                train_loss = binary_crossentropy(y_ok[tr_index], y_pred[tr_index], X.shape[0])
                val_acc = accuracy_score(y_ok[val_index], y_pred_labels[val_index])
                val_loss = binary_crossentropy(y_ok[val_index], y_pred[val_index], X.shape[0])
                print('{:15} accuracy={:.6f}, loss={:.6f}'.format('Train:', train_acc, train_loss))
                print('{:15} accuracy={:.6f}, loss={:.6f}'.format('Validation:', val_acc, val_loss))
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_preds = y_pred
                    wait = 0
                else:
                    if wait >= self.patience:
                        print('Epoch {}: early stopping'.format(epoch))
                        break
                    wait += 1
                
            
            best_pred_labels = (best_preds > 0.5).astype('int32')  # binary class labels (0 and 1) 
            
#             test_acc = accuracy_score(y_ok[test_index], best_pred_labels[test_index])
#             test_loss = binary_crossentropy(y_ok[test_index], best_preds[test_index], X.shape[0])
#             print('Test: accuracy={:.6f}, loss={:.6f}'.format(test_acc, test_loss))
    
            K.clear_session()  # if retrain!
            score = [f(y_ok[test_index], best_pred_labels[test_index]) for f in scoring_functions]
            scores.append(score)
            print("-> SCORES: {}".format(scores))
               
#             break # only one iteration
       
        means = np.mean(scores, axis=0)
        stds = np.std(scores, axis=0)
    
        for (i, f) in enumerate(scoring_functions):
            print("\t{:20}: {:.6f} (+/- {:.6f})".format(f.__name__, means[i], stds[i]))


if __name__ == "__main__":
    pass
