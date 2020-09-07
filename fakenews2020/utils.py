from scipy.sparse import csr_matrix
import codecs
import json
import regex
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import tensorflow as tf    
import numpy as np


class ExperimentBase:
    def set_data(self, data, data_prop):
        self.data = data
        self.data_prop = data_prop
    
    def run(self, texts_i, exps_i):
        pass
    
class NNBase:
    def getX(self):
        pass
    
    def gety(self):
        pass
    
    def build_model(self):
        pass
    
scoring_functions = [accuracy_score, precision_score, recall_score, f1_score]

def combine_sparse_matrices(X, Z):
    """Combines two sparse matrices (horizontally)
    """
    maxCol = max(X.indices)
    maxCol2 = max(Z.indices)
    Z2 = csr_matrix((Z.data, Z.indices + maxCol + 1, Z.indptr))
    X2 = csr_matrix(X, shape=(X.shape[0], maxCol + maxCol2 + 2))
    return X2 + Z2

def load_data(f_name):
    """Loads data from JSON format.
    The data must contain the fields: title, text, label (= +/- 1)
    """
    f = codecs.open(f_name, 'r', encoding='utf8', errors='ignore')
    data = json.load(f)
    f.close()
    fields = {'title', 'text', 'label'}
    if sum([len(fields.intersection(d.keys())) != len(fields) for d in data]) != 0:
        raise Exception('Wrong data format!')
    return data

base_word_tokenizer = CountVectorizer().build_tokenizer() # r"(?u)\b\w\w+\b"

RE_SPLIT = r'\w+|[^\w\s]+'
reSplit = regex.compile(RE_SPLIT, regex.U)

def word_tokenizer(text, lowercase=True):
    """Word tokenizer
    """
    if lowercase:
        tokens = [str.lower(x) for x in regex.findall(reSplit, text)]
    else:
        tokens = [x for x in regex.findall(reSplit, text)]
    return tokens

def sentence_tokenizer(text):
    """Sentence tokenizer: nltk's sentence tokenizer
    """
    return sent_tokenize(text)

def char_tokenizer(text):
    """Character tokenizer
    """
    txt = regex.sub(r'\s+', ' ', text)
    return [t for t in txt]

def binary_crossentropy(y, yh, n):
    """(Because loss is averaged over all training data, regardless of sample_weight.)
    """
    return np.sum([-np.log(yh[i]) if y[i] == 1 else -np.log(1-yh[i]) for i in range(len(y))]) / n

def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors.
    """
    random_tensor = keep_prob
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse.retain(x, dropout_mask)
    return pre_out * (1./keep_prob)



if __name__ == "__main__":
    pass
#     data = load_data('data/kaggle.json')
#     print(data)
