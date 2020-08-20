from scipy.sparse import csr_matrix
import codecs
import json
import regex
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer


def combine_sparse_matrices(X, Z):
    """Combines two sparse matrices
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

def word_tokenizer(text):
    """Word tokenizer
    """
    tokens = [x for x in regex.findall(reSplit, text)]
#     tokens = [str.lower(x) for x in regex.findall(reSplit, text)]
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




if __name__ == "__main__":
    pass
#     data = load_data('data/kaggle.json')
#     print(data)
