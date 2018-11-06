import logging
import os
import sys

import utils
from gensim.models import Word2Vec
import re


class SentencesFromDirectory(object):
    def __init__(self, dir_path, remove_stopwords=False):
        self.remove_stopwords = remove_stopwords
        self.dir_path = dir_path

    def __iter__(self):
        text_files = filter(lambda f: f.endswith('.txt') and re.match(r'nyt-(\d{4}).*$', f),
                            os.listdir(self.dir_path))
        for file in text_files:
            # We assume that each file contains a single sentence per line
            for sentence in open(os.path.join(self.dir_path, file), encoding='utf8'):
                # tokenize and return each sentence as a list of words
                yield utils.tokenize(sentence, remove_stopwords=self.remove_stopwords)


class Word2VecAllTimeModel(object):
    def __init__(self, dir_path=sys.path[0], sg=0, size=140, min_count=20, window=2, remove_stopwords=True, alpha=0.05):
        self.MODEL_PATH = os.path.join(dir_path, 'word2vec-nyt-9999.model')

        if not os.path.exists(self.MODEL_PATH):  # create the model
            logging.info('Creating a word2vec model for all time')
            corpus = SentencesFromDirectory(dir_path, remove_stopwords)

            self.model = Word2Vec(corpus, sg=sg, size=size, window=window, min_count=min_count, workers=4,
                                  sorted_vocab=True, alpha=alpha)
            # Precompute the L2-normalized vectors. This makes the model read-only
            self.model.init_sims(replace=True)
            self.model.save(self.MODEL_PATH)
        else:
            logging.info('Loading existing word2vec model')
            self.model = Word2Vec.load(self.MODEL_PATH)  # just load the model
            # Precompute the L2-normalized vectors. This makes the model read-only
            self.model.init_sims(replace=True)
