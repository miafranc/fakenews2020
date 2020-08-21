from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import normalize, binarize
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection._split import StratifiedKFold
from sklearn.metrics import make_scorer

from simple_text_features import SimpleTextFeaturesVectorizer
from utils import combine_sparse_matrices, base_word_tokenizer, word_tokenizer, sentence_tokenizer
from utils import ExperimentBase
from utils import scoring_functions


class GOFTC(ExperimentBase):
    
    def __init__(self):
        self.n_jobs = -1
        self.cv_k = 5
        self.cv_shuffle = False
        
    def build_model(self):
        # Experiments:
        if self.exps_i == 0:
            count_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 1))
        elif self.exps_i == 1:
            count_vectorizer = CountVectorizer(binary=False, ngram_range=(1, 1))
        elif self.exps_i == 2:
            count_vectorizer = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        elif self.exps_i == 3:
            count_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
        elif self.exps_i == 4:
            count_vectorizer = CountVectorizer(tokenizer=word_tokenizer, binary=True, ngram_range=(1, 2))
        elif self.exps_i == 5 or self.exps_i == 6:
            count_vectorizer = CountVectorizer(tokenizer=word_tokenizer, binary=False, ngram_range=(1, 2))
        elif self.exps_i == 7 or self.exps_i == 8:
            count_vectorizer = TfidfVectorizer(tokenizer=word_tokenizer, ngram_range=(1, 2), max_df=0.5)
        
        # Data (text, title, both):
        if self.texts_i == 0:
            train_X = count_vectorizer.fit_transform([d['text'] for d in self.data])
        elif self.texts_i == 1:
            train_X = count_vectorizer.fit_transform([d['title'] for d in self.data])
        elif self.texts_i == 2:
            train_X = count_vectorizer.fit_transform([d['title'] + ' ' + d['text'] for d in self.data])
        train_y = [d['label'] for d in self.data]
    
        # Feature selection:
        if self.exps_i in [5, 6, 7, 8]:
            if self.texts_i == 1:
                train_X = SelectKBest(chi2, k=1000).fit_transform(train_X, train_y)
            else:
                train_X = SelectKBest(chi2, k=5000).fit_transform(train_X, train_y)
            train_X = binarize(train_X)
    
        # Stat. features:
        if self.exps_i in [6, 7, 8]:
            stf = SimpleTextFeaturesVectorizer(base_word_tokenizer, sentence_tokenizer)
            if self.texts_i == 0:
                train_X_2 = stf.fit_transform([d['text'] for d in self.data])
            elif self.texts_i == 1:
                train_X_2 = stf.fit_transform([d['title'] for d in self.data])
            elif self.texts_i == 2:
                train_X_2 = stf.fit_transform([d['title'] + ' ' + d['text'] for d in self.data])
            train_X = combine_sparse_matrices(train_X, train_X_2)

        # Normalization:
        if self.exps_i == 8: # tweak2
            train_X = normalize(train_X, norm='l1', axis=0)
    
        return (train_X, train_y)
    
    def run(self, texts_i, exps_i):
        self.texts_i = texts_i
        self.exps_i = exps_i
        train_X, train_y = self.build_model()
        
        # Classifiers:
        clfs = [LogisticRegression(solver='lbfgs', max_iter=1000, verbose=False),
                LinearSVC(max_iter=10000),
                MultinomialNB()]

        # Classification tests:
        for clf in clfs:
            print(f'{clf.__class__.__name__}')
            # Cross validation:
            cv = StratifiedKFold(n_splits=self.cv_k, shuffle=self.cv_shuffle)
            scores = cross_validate(clf, train_X, train_y, 
                                    cv=cv, 
                                    scoring={sf.__name__:make_scorer(sf) for sf in scoring_functions}, 
                                    return_train_score=False,
                                    n_jobs=self.n_jobs)
            for s in scores.keys():
                if s != 'fit_time' and s != 'score_time':
                    print("\t{:20}: {:.6f} (+/- {:.6f})".format(s[5:], scores[s].mean(), scores[s].std()))
        
        
        
if __name__ == "__main__":
    pass
    