import logging
from collections import OrderedDict

import pysolr
from unidecode import unidecode

import utils
from models_builder import ModelsBuilder
from search.qe_multiple_entities import MultipleEntitiesQEMethod
from search.qe_multiple_entities import QEMultipleEntities
from search.qe_single_entity import QESingleEntity


class QueryType(utils.AutoNumber):
    EntityPeriod = ()
    MultipleEntities = ()
    EntityRelationYear = ()


class EvaluationMethod(utils.AutoNumber):
    Absolute = ()
    RelativeToNoQE = ()


class TemporalSearch:
    def __init__(self, query_type, eval_method, models_dir, qe_method=None,
                 two_entities_qe_method=None, start_year=None,
                 end_year=None, precision_at=10, num_of_expanding_terms=2, year_to_model=None, global_model=None,
                 svm_model=None):
        self.solr = pysolr.Solr('http://localhost:8983/solr/nyt')
        self.precision_at = precision_at
        self.k = num_of_expanding_terms
        self.min_year = start_year
        self.max_year = end_year
        self.query_type = query_type
        self.two_entities_qe_method = two_entities_qe_method
        self.eval_method = eval_method
        self.year_to_model = year_to_model
        self.global_model = global_model
        self.svm_model = svm_model
        if self.two_entities_qe_method.value > MultipleEntitiesQEMethod.baseline.value:
            if not self.year_to_model:
                year_to_model = ModelsBuilder(data_dir_name=models_dir, from_year=start_year, to_year=end_year,
                                              global_model_dir=models_dir).load_models()
                # convert to an OrderedDict, sorted by key
                self.year_to_model = OrderedDict(sorted(year_to_model.items(), key=lambda t: t[0]))
            self.global_model = self.year_to_model[utils.GLOBAL_YEAR]

        self.qe_single_entity = QESingleEntity(qe_method=qe_method, k=self.k, year_to_model=self.year_to_model,
                                               global_model=self.global_model, svm_model=self.svm_model)
        self.qe_multiple_entities = QEMultipleEntities(self.qe_single_entity,
                                                       two_entities_qe_method=self.two_entities_qe_method, k=self.k,
                                                       year_to_model=self.year_to_model,
                                                       global_model=self.global_model, svm_model=self.svm_model,
                                                       min_year=self.min_year, max_year=self.max_year)

    def search(self, *args):
        """
        expands a given query using the temporal model, searches NYT and returns a list of articles
        """
        if self.query_type == QueryType.EntityPeriod:
            query = args[0]
            time_start = args[1]
            time_end = args[2]
            middle_year = utils.get_middle_year(time_start, time_end)
            related_tuples = self.qe_single_entity.expand_entity(query, middle_year)
            expanded_query = '{} {}'.format(query, ' '.join([tup[0] for tup in related_tuples]))
        elif self.query_type == QueryType.MultipleEntities:
            entities = args[0]
            time_start = args[1]
            time_end = args[2]
            query = ' '.join(entities)
            expansions = self.qe_multiple_entities.expand(entities)
            if expansions is None:
                return None
            expanded_query = '{} {}'.format(query, expansions)
        else:
            return None

        expanded_query = expanded_query.replace('_nnp', '').replace('_', ' ')  # prepare to search
        msg = unidecode(
            'got query "{}" @ {}-{}, will search for "{}"'.format(query, time_start, time_end, expanded_query))
        logging.warning(msg)
        # search and return results
        try:
            results = self.solr.search(q=expanded_query)
        except:
            logging.exception('Failed searching for "%s"', expanded_query)
            return None
        return results

    def evaluate(self, *args):
        """
        expands a given query using the temporal model, searches NYT and returns a list of articles
        """
        query = None
        expanded_query = None
        time_start = None
        time_end = None
        # parse the arguments
        if self.query_type == QueryType.EntityPeriod:
            query = args[0][0]
            time_start = args[1]
            time_end = args[2]
            middle_year = utils.get_middle_year(time_start, time_end)
            related_tuples = self.qe_single_entity.expand_entity(query, middle_year)
            if related_tuples:
                expanded_query = '{} {}'.format(query, ' '.join([tup[0] for tup in related_tuples]))
        elif self.query_type == QueryType.MultipleEntities:
            entities = args[0]
            time_start = args[1]
            time_end = args[2]
            query = ' '.join(entities)
            if not self.global_model.contains_all_words(entities):
                return None
            expansions = self.qe_multiple_entities.expand(entities)
            if expansions is None:
                return None
            expanded_query = '{} {}'.format(query, expansions)
        elif self.query_type == QueryType.EntityRelationYear:
            logging.error('QueryType.EntityRelationYear is unsupported')
            exit()

        if expanded_query is None:  # if this term wasn't expanded
            return None
        expanded_query = expanded_query.replace('_nnp', '').replace('_', ' ')  # prepare to search

        result = self.search_eval(expanded_query, time_start, time_end)
        if result:
            true_articles_count = result[0]
            total_articles_count = result[1]
        else:
            return None

        if self.eval_method == EvaluationMethod.Absolute:
            pass
        elif self.eval_method == EvaluationMethod.RelativeToNoQE:
            # search without QE
            result = self.search_eval(query, time_start, time_end)
            if result:
                baseline_true_articles_count = result[0]
                baseline_total_articles_count = result[1]
            else:
                return None
            # true count will be the difference between the QE method and the baseline
            true_articles_count -= baseline_true_articles_count
            total_articles_count = max(total_articles_count, baseline_total_articles_count)

        return true_articles_count, total_articles_count

    def search_eval(self, query, time_start, time_end):

        query = query.replace('_nnp', '').replace('_', ' ')  # prepare to search

        if time_start:
            if time_end is None or time_start == time_end:
                msg = 'searching for "{}" @ {}'.format(query, time_start)
            else:
                msg = 'searching for "{}" @ {}-{}'.format(query, time_start, time_end)
        else:
            msg = 'searching for "{}"'.format(query)
        msg = unidecode(msg)

        # search and return results
        try:
            results = self.solr.search(q=query, **{'rows': str(self.precision_at)})
        except:
            logging.exception('Failed searching for "%s"', query)
            return None
        if not results:
            logging.warning(msg + ', no results' if msg else 'searching for "{}", '.format(query))
            return None
        true_articles_count = 0
        total_articles_count = 0
        for article in results:
            url = ''
            pub_year = ''
            try:
                url = article['id']
                logging.debug(url)
                pub_year = article['year']
                # the year field is currently a list (my fault), so check that case, too
                if isinstance(pub_year, list):
                    pub_year = article['year'][0]
                else:
                    pub_year = int(article['year'])
                total_articles_count += 1
                if time_start <= pub_year <= time_end:
                    true_articles_count += 1
            except:
                logging.exception("Article's year couldn't be parsed: {} , {}".format(url, pub_year))
                continue
        logging.warning(msg + ', accuracy: {}/{}'.format(true_articles_count, total_articles_count))
        return true_articles_count, total_articles_count
