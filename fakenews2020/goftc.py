from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import normalize, binarize
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection._split import StratifiedKFold

from simple_text_features import SimpleTextFeaturesVectorizer
from utils import combine_sparse_matrices, load_data, base_word_tokenizer, word_tokenizer, sentence_tokenizer

    
class GOFTC:
    
    def __init__(self):
        self.n_jobs = -1
        self.cv_k = 5
        self.cv_shuffle = False

    def set_data(self, data, data_name):
        self.data = data
        self.data_name = data_name
        
    def build_model(self, texts_i, exps_i):
        # Experiments:
        if exps_i == 0:
            count_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 1))
        elif exps_i == 1:
            count_vectorizer = CountVectorizer(binary=False, ngram_range=(1, 1))
        elif exps_i == 2:
            count_vectorizer = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        elif exps_i == 3:
            count_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
        elif exps_i == 4:
            count_vectorizer = CountVectorizer(tokenizer=word_tokenizer, binary=True, ngram_range=(1, 2))
        elif exps_i == 5 or exps_i == 6:
            count_vectorizer = CountVectorizer(tokenizer=word_tokenizer, binary=False, ngram_range=(1, 2))
        elif exps_i == 7 or exps_i == 8:
            count_vectorizer = TfidfVectorizer(tokenizer=word_tokenizer, ngram_range=(1, 2), max_df=0.5)
        
        # Data (text, title, both):
        if texts_i == 0:
            train_X = count_vectorizer.fit_transform([d['text'] for d in data])
        elif texts_i == 1:
            train_X = count_vectorizer.fit_transform([d['title'] for d in data])
        elif texts_i == 2:
            train_X = count_vectorizer.fit_transform([d['text'] + ' ' + d['title'] for d in data])
        train_y = [d['label'] for d in data]
    
        # Feature selection:
        if exps_i == 5 or exps_i == 6 or exps_i == 7 or exps_i == 8:
            if texts_i == 1:
                train_X = SelectKBest(chi2, k=1000).fit_transform(train_X, train_y)
            else:
                train_X = SelectKBest(chi2, k=5000).fit_transform(train_X, train_y)
            train_X = binarize(train_X)
    
        # Stat. features:
        if exps_i == 6 or exps_i == 7 or exps_i == 8:
            stf = SimpleTextFeaturesVectorizer(base_word_tokenizer, sentence_tokenizer)
            if texts_i == 0:
                train_X_2 = stf.fit_transform([d['text'] for d in data])
            elif texts_i == 1:
                train_X_2 = stf.fit_transform([d['title'] for d in data])
            elif texts_i == 2:
                train_X_2 = stf.fit_transform([d['text'] + ' ' + d['title'] for d in data])
            train_X = combine_sparse_matrices(train_X, train_X_2)

        # Normalization:
        if exps_i == 8: # tweak2
            train_X = normalize(train_X, norm='l1', axis=0)
    
        self.train_X = train_X
        self.train_y = train_y
    
    def run(self):
        # Classifiers:
        clfs = [LogisticRegression(solver='lbfgs', max_iter=1000, verbose=False),
                LinearSVC(max_iter=10000),
                MultinomialNB()]

        # Classification tests:
        for clf in clfs:
            print(f'{self.data_name} / {clf.__class__.__name__}')
            # Cross validation:
            cv = StratifiedKFold(n_splits=self.cv_k, shuffle=self.cv_shuffle)
            scores = cross_validate(clf, self.train_X, self.train_y, 
                                    cv=cv, scoring=['accuracy', 'precision', 'recall', 'f1'], return_train_score=False,
                                    n_jobs=self.n_jobs)
            print("\t F1: {:.6f} (+/- {:.6f})".format(scores['test_f1'].mean(), scores['test_f1'].std()))
        
        
        
if __name__ == "__main__":
    texts = [
        'text', #0 
        'title', #1
        'text+title' #2
    ]
    
    exps = [
        'baseline_binary', #0 
        'baseline_freq', #1
        'baseline_tfidf', #2
        'uni+bi_binary', #3
        'new_tokenizer', #4
        'chi2_5000', #5
        'stat_features', #6
        'tweak1', #7
        'tweak2' #8
    ] 

    datasets = {
        'FNAMT': 'data/perez_fakenews.json',
        'Celeb': 'data/perez_celebrity.json',
        'BF': 'data/buzzfeed.json',
        'Rand': 'data/random.json',
        'MI': 'data/mcintire.json',
        'UTK': 'data/kaggle.json',
        'GC': 'data/shu_fakenewsnet_gossipcop.json',
        'PF': 'data/shu_fakenewsnet_politifact.json'    
    }

    ds = 'FNAMT'
    
    tc = GOFTC()
    data = load_data(datasets[ds])
    tc.set_data(data, ds)
    
    tc.build_model(2, 6)
    tc.run()
    