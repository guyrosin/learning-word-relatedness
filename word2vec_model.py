import logging
import os
import sys
from gensim.models import Word2Vec
import utils


class SentencesFromFile(object):
    def __init__(self, filename, remove_stopwords=False):
        self.filename = filename
        self.remove_stopwords = remove_stopwords

    def __iter__(self):
        # We assume the file contains a single sentence per line
        for sentence in open(self.filename, encoding='utf8'):
            # tokenize and return each sentence as a list of words
            yield utils.tokenize(sentence, remove_stopwords=self.remove_stopwords)


class Word2VecModel(object):
    def __init__(self, year, dir_path=sys.path[0], sg=0, size=140, min_count=50, window=2, alpha=0.025,
                 remove_stopwords=False):
        """
        :param year:
        :param dir_path:
        :param sg: True for Skipgram, False for CBOW
        :param size: vector size
        :param min_count:
        :param window: context window (from each side)
        :param alpha: learning rate
        :param remove_stopwords: True to remove stopwords
        """
        self.year = year
        corpus_filename_str = 'nyt-%i.txt'
        self.corpus_filename = os.path.join(dir_path, corpus_filename_str % year)
        self.MODEL_PATH = os.path.join(dir_path, 'word2vec-nyt-%i.model' % year)

        if not os.path.exists(self.MODEL_PATH):  # create the model
            logging.info('Creating a word2vec model for %i' % year)
            corpus = SentencesFromFile(self.corpus_filename, remove_stopwords)
            self.model = Word2Vec(corpus, sg=sg, size=size, window=window, min_count=min_count,
                                  alpha=alpha, workers=4, sorted_vocab=True)
            self.model.save(self.MODEL_PATH)
        else:
            logging.info('Loading existing word2vec model')
            self.model = Word2Vec.load(self.MODEL_PATH)  # just load the model
        # Precompute the L2-normalized vectors. This makes the model read-only
        self.model.init_sims(replace=True)

    def contains_all_words(self, words):
        return all(w in self.model for w in words)
