import numpy as np
import scipy as scipy
import lxmls.classifiers.linear_classifier as lc
import sys
from lxmls.distributions.gaussian import *


class MultinomialNaiveBayes(lc.LinearClassifier):
    def __init__(self, xtype="gaussian"):
        lc.LinearClassifier.__init__(self)
        self.trained = False
        self.likelihood = 0
        self.prior = 0
        self.smooth = False
        self.smooth_param = 1

    def train(self, x, y):
        # n_docs = no. of documents
        # n_words = no. of unique words
        n_docs, n_words = x.shape

        # classes = a list of possible classes
        classes = np.unique(y)
        # n_classes = no. of classes
        n_classes = np.unique(y).shape[0]

        # initialization of the prior and likelihood variables
        prior = np.zeros(n_classes)
        likelihood = np.zeros((n_words, n_classes))

        # TODO: This is where you have to write your code!
        # You need to compute the values of the prior and likelihood parameters
        # and place them in the variables called "prior" and "likelihood".
        # Examples:
        # prior[0] is the prior probability of a document being of class 0
        # likelihood[4, 0] is the likelihood of the fifth(*) feature being
        # active, given that the document is of class 0
        # (*) recall that Python starts indices at 0, so an index of 4
        # corresponds to the fifth feature!

        # calculate prior
        for index, classification_class in enumerate(classes):
            prior[index] = np.sum(y == classification_class) / float(n_docs)

        # calculate likelihood is the same as the conditional probability
        # P(x|y) = P(x,y) / P(y)
        # P(word|class) = P(word, class) / P(class)
        # probability of word x_i when knowing the class is 0
        for index, document in enumerate(x):
            # see the class in y
            class_in_y = y[index][0]
            for word_index, word_occurency in enumerate(document):
                likelihood[word_index, class_in_y] += word_occurency

        # at this point we have the occurency of each word in each class
        # we need to divide by the amount of words independent of the class

        total_word_occurency = np.sum(x)
        print("total_word_occurency: ", total_word_occurency)

        likelihood = likelihood / total_word_occurency
        # need to divide by the prior
        likelihood = likelihood / prior
        print("likelihood: ", likelihood)

        # ----------
        # Solution to Exercise 1.1
        for i in range(n_classes):
            docs_in_class, _ = np.nonzero(
                y == classes[i]
            )  # docs_in_class = indices of documents in class i
            prior[i] = (
                1.0 * len(docs_in_class) / n_docs
            )  # prior = fraction of documents with this class

            # word_count_in_class = count of word occurrences in documents of class i
            word_count_in_class = x[docs_in_class, :].sum(0)
            total_words_in_class = (
                word_count_in_class.sum()
            )  # total_words_in_class = total number of words in documents of class i
            if not self.smooth:
                # likelihood = count of occurrences of a word in a class
                likelihood[:, i] = word_count_in_class / total_words_in_class
            else:
                likelihood[:, i] = (word_count_in_class + self.smooth_param) / (
                    total_words_in_class + self.smooth_param * n_words
                )
        # End solution to Exercise 1.1
        # ----------

        params = np.zeros((n_words + 1, n_classes))
        for i in range(n_classes):
            params[0, i] = np.log(prior[i])
            params[1:, i] = np.nan_to_num(np.log(likelihood[:, i]))
        self.likelihood = likelihood
        self.prior = prior
        self.trained = True
        return params
