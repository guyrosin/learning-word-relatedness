import re
from collections import OrderedDict
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import utils
from word2vec_model import Word2VecModel


class ModelsBuilder(object):
    def __init__(self, data_dir_name=sys.path[0], from_year=None, to_year=None,
                 global_model_dir=None, sg=0, size=140, window=2, remove_stopwords=True, min_count=30, alpha=0.05):
        """

        :param data_dir_name:
        :param from_year:
        :param to_year:
        :param global_model_dir: directory of the global model (leave blank if it's in the same one)
        """
        self.data_dir_name = data_dir_name
        self.year_to_model = {}
        self.from_year = from_year or 0
        self.to_year = to_year or utils.GLOBAL_YEAR
        self.global_model_dir = global_model_dir
        self.sg = sg
        self.window = window
        self.size = size
        self.remove_stopwords = remove_stopwords
        self.min_count = min_count
        self.alpha = alpha

    def build_model_of_year(self, year):
        logging.info('Building model for %i', year)
        self.year_to_model[year] = Word2VecModel(year, self.data_dir_name, sg=self.sg, size=self.size,
                                                 min_count=self.min_count, remove_stopwords=self.remove_stopwords,
                                                 window=self.window, alpha=self.alpha)

    def build_models_from_files(self, files):
        with ProcessPoolExecutor(max_workers=4) as executor:  # this will wait until completion of all workers
            for f in files:
                # m = re.match(r'nyt-(\d{4})\.txt$', f)
                # look for years in the filename
                m = re.match(r'.*(\d{4}).*$', f)
                if m is None:
                    continue
                year = int(m.group(1))
                if year not in self.year_to_model and self.from_year <= year <= self.to_year:
                    executor.submit(self.build_model_of_year, year)
            if utils.GLOBAL_YEAR not in self.year_to_model and self.global_model_dir is not None:
                self.year_to_model[utils.GLOBAL_YEAR] = Word2VecModel(utils.GLOBAL_YEAR,
                                                                      dir_path=self.global_model_dir)

    def build_all_models(self):
        # build a word2vec model out of each model file in the 'data' folder
        files_list = os.listdir(self.data_dir_name)
        model_files = filter(lambda file: file.endswith('.model'), files_list)
        corpus_files = filter(lambda file: file.endswith('.corpus'), files_list)
        text_files = filter(lambda file: file.endswith('.txt'), files_list)
        self.build_models_from_files(model_files)
        # now look at all the corpus/text files and add new models, if needed
        self.build_models_from_files(corpus_files)
        self.build_models_from_files(text_files)

        # convert to an OrderedDict, sorted by key
        self.year_to_model = OrderedDict(sorted(self.year_to_model.items(), key=lambda t: t[0]))
        return self.year_to_model

    def load_models(self):
        # build a word2vec model out of each model file in the 'data' folder
        files_list = os.listdir(self.data_dir_name)
        files_list = filter(lambda file: file.endswith('.model'), files_list)
        self.build_models_from_files(files_list)

        # convert to an OrderedDict, sorted by key
        self.year_to_model = OrderedDict(sorted(self.year_to_model.items(), key=lambda t: t[0]))
        return self.year_to_model
