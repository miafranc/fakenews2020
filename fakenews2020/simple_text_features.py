import regex
import numpy as np
from scipy.sparse import csr_matrix


class SimpleTextFeaturesVectorizer:
    """Generating simple statistical text features.
    """
    R_ALLCAPS = regex.compile(r"\p{lu}+", regex.U)
    R_CAPSTART = regex.compile(r"^\p{lu}", regex.U)
    R_ABBREV = regex.compile(r"(\p{lu}\W)+", regex.U)
    R_NUM = regex.compile(r"[0-9]+", regex.U)
    
    def __init__(self, word_tokenizer, sentence_tokenizer, w_max=30, s_max=30):
        self.word_tokenizer = word_tokenizer
        self.sentence_tokenizer = sentence_tokenizer
        self.hist_w_max = w_max
        self.hist_s_max = s_max
        self.method_list = sorted([func for func in dir(self) if func.startswith('_f_') and callable(getattr(self, func))])
    
    def _generate(self, text):
        self.tokens = self.word_tokenizer(text)
        self.sentences = self.sentence_tokenizer(text)
        self.word_num = len(self.tokens)
        self.sent_num = len(self.sentences)
        self.word_norm = float(self.word_num) if self.word_num > 0 else 1.
        self.sent_norm = float(self.sent_num) if self.sent_num > 0 else 1.
        
        return np.array([x for method in self.method_list for x in getattr(self, method)()])
        
    def fit_transform(self, X):
        return csr_matrix([self._generate(x) for x in X])
        
    def _f_all_caps(self):
        return [sum([regex.fullmatch(self.R_ALLCAPS, t) != None for t in self.tokens]) / self.word_norm]
     
    def _f_cap_start(self):
        return [sum([regex.match(self.R_CAPSTART, t) != None for t in self.tokens]) / self.word_norm]
   
    def _f_numbers(self):
        return [sum([regex.match(self.R_NUM, t) != None for t in self.tokens]) / self.word_norm]
   
    def _f_abbrev(self):
        return [sum([regex.fullmatch(self.R_ABBREV, t) != None for t in self.tokens]) / self.word_norm]
        
    def _f_word_len_hist(self):
        w_lengths = [len(t) for t in self.tokens]
        h = np.histogram(w_lengths, bins=self.hist_w_max-1, range=(1, self.hist_w_max))
        long_words = sum([l > self.hist_w_max for l in w_lengths])  # if longer than parameter (self.hist_w_max)
        return np.append(h[0], long_words) / self.word_norm
    
    def _f_sent_len_hist(self):
        s_lengths = [len(self.word_tokenizer(s)) for s in self.sentences]
        h = np.histogram(s_lengths, bins=self.hist_s_max-1, range=(1, self.hist_s_max))
        long_sents = sum([l > self.hist_s_max for l in s_lengths])  # if longer than parameter (self.hist_s_max)
        return np.append(h[0], long_sents) / self.sent_norm
