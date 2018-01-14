import math
import statistics
import warnings
import sys

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences

class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError   

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None

class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def score(self, hmm_model, x, x_lengths):
        '''
        Score the model using BIC

        "p is the number of parameters"
        "N is the number of data points"
        p = "increasing model complexity (more parameters),"
        
        Ref:
        https://www.quora.com/Why-can-a-Hidden-Markov-Model-only-remember-a-log-N-bits
        https://en.wikipedia.org/wiki/Bayesian_information_criterion
        https://stats.stackexchange.com/questions/12341/number-of-parameters-in-markov-model
        http://web.science.mq.edu.au/~cassidy/comp449/html/ch12s02.html
        https://ai-nd.slack.com/archives/C3V8A1MM4/p1493912818353455
        '''
        logL = hmm_model.score(x, x_lengths)
        f = len(x[0]) # number of features
        n = hmm_model.n_components # number of states
        logN = np.log(len(x))  
        p = (n * n) + 2 * (n * f) - 1
        bic = (-2 * logL) + (p * logN)
        if self.verbose: 
            print("bic values: ", bic, logL, f, n, p, logN)
        return bic

    def select(self):
        '''
        Select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        Return model with the lowest BIC score
        '''
        best_score = float("inf")
        best_model = None
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            # build model
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False)
            try: 
                hmm_model.fit(self.X, self.lengths)
                score = self.score(hmm_model, self.X, self.lengths)
                if score < best_score:
                    best_score = score
                    best_model = hmm_model
                if self.verbose:
                    print("model created for {} with {} states, withs score: {}".format(self.this_word, num_states, score))
            except ValueError as e: 
                if self.verbose:
                    print("ValueError error({0}):".format(e))
                    print("failure on {} with {} states".format(self.this_word, num_states))

        return best_model

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''
    def score(self, hmm_model, x, x_lengths):
        '''
        Score the model using the DIC value
        '''
        # log(P(X(i))
        logL = hmm_model.score(x, x_lengths)
        # SUM(log(P(X(all but i))
        logL_sum = 0
        for word in self.words:
            if word != self.this_word:
                X, lengths = self.hwords[word]
                logL_sum += hmm_model.score(X, lengths)
        # DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
        dic = logL - 1/((len(self.words) - 1) * logL_sum)
        if self.verbose: 
            print("dic values: ", bic, logL, logL_sum)
        return dic

    def select(self):
        '''
        Return model with the highest DIC score
        '''
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_score = float("-inf")
        best_model = None
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False)
            try: 
                hmm_model.fit(self.X, self.lengths)
                score = self.score(hmm_model, self.X, self.lengths)
                if score > best_score:
                    best_score = score
                    best_model = hmm_model
                if self.verbose:
                    print("model created for {} with {} states, withs score: {}".format(self.this_word, num_states, score))
            except ValueError as e: 
                if self.verbose:
                    print("ValueError error({0}):".format(e))
                    print("failure on {} with {} states".format(self.this_word, num_states))

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    def score(self, hmm_model, x, x_lengths):
        ''' 
        use the log lkelihood value returned by the scored function for evaluaion 
        '''
        return hmm_model.score(x, x_lengths)

    def get_n_splits(self):
        ''' 
        provides the number of splits to be used with KFold 
        '''
        return 2 if len(self.sequences) < 4 else 3

    def select(self):
        '''
        Selects the best model as determined by the average score
        across n_splits cross-validation sets
        '''
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = float("-inf")
        best_model = None
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False)
            scores = []
            try: 
                n_splits = self.get_n_splits()
                split_method = KFold(n_splits=n_splits)
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    # train
                    train_x, train_length = combine_sequences(cv_train_idx, self.sequences)
                    hmm_model.fit(train_x, train_length)
                    # test
                    test_x, test_length = combine_sequences(cv_test_idx, self.sequences)
                    # evaluate
                    score = self.score(hmm_model, test_x, test_length)
                    scores.append(score)
                    if self.verbose:
                        print("model created for {} with {} states, withs score: {}".format(self.this_word, num_states, score))
            except ValueError as e: 
                if self.verbose:
                    print("ValueError error({0}):".format(e))
                    print("failure on {} with {} states".format(self.this_word, num_states))
                scores = []

            if len(scores) > 0:
                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_model = hmm_model

        return best_model

