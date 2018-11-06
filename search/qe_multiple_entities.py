import logging
import itertools
import numpy as np

import utils
from search import qe_single_entity
from search.max_entities_heap import MaxEntitiesHeap


class MultipleEntitiesQEMethod(utils.AutoNumber):
    none = ()
    baseline = ()
    globally_based_classifier = ()
    temporal_classifier = ()
    temporal_model_classifier = ()
    peak_detection = ()


class QEMultipleEntities:
    def __init__(self, qe_single_entity, two_entities_qe_method=None, k=2, year_to_model=None, global_model=None,
                 svm_model=None, min_year=None, max_year=None):
        self.qe_single_entity = qe_single_entity
        self.two_entities_qe_method = two_entities_qe_method
        self.k = k
        self.global_model = global_model
        self.year_to_model = year_to_model
        self.svm_model = svm_model
        self.min_year = min_year
        self.max_year = max_year

    def expand(self, entities):
        return self.expand_entities(entities)

    def expand_entities(self, entities, year=None):
        """
        applies QE with the selected method and returns a string of expansion terms
        """
        if not year:
            year = self.calc_best_year_method()(entities)
        entities = [ent.replace(' ', '_') for ent in entities]

        if self.two_entities_qe_method == MultipleEntitiesQEMethod.none:
            return ''
        elif self.two_entities_qe_method == MultipleEntitiesQEMethod.baseline:
            return self.expand_baseline(entities)
        elif self.two_entities_qe_method == MultipleEntitiesQEMethod.globally_based_classifier:
            return self.expand_entities_generic(entities, year)
        elif self.two_entities_qe_method == MultipleEntitiesQEMethod.temporal_classifier:
            return self.expand_entities_generic(entities, year)
        elif self.two_entities_qe_method == MultipleEntitiesQEMethod.temporal_model_classifier:
            return self.expand_entities_generic(entities, year)
        elif self.two_entities_qe_method == MultipleEntitiesQEMethod.peak_detection:
            return self.expand_entities_generic(entities, year)
        else:
            raise ValueError('Unknown QE method: {}'.format(self.two_entities_qe_method))

    def calc_best_year_method(self):
        if self.two_entities_qe_method == MultipleEntitiesQEMethod.none:
            return lambda x: 0
        elif self.two_entities_qe_method == MultipleEntitiesQEMethod.baseline:
            return lambda x: 0
        elif self.two_entities_qe_method == MultipleEntitiesQEMethod.globally_based_classifier:
            return self.calc_best_year_globally_based_classifier
        elif self.two_entities_qe_method == MultipleEntitiesQEMethod.temporal_classifier:
            return self.calc_best_year_temporal_classifier
        elif self.two_entities_qe_method == MultipleEntitiesQEMethod.temporal_model_classifier:
            return self.calc_best_year_temporal_model_classifier
        elif self.two_entities_qe_method == MultipleEntitiesQEMethod.peak_detection:
            return self.calc_best_year_peak_detection
        else:
            raise ValueError('Unknown QE method: {}'.format(self.two_entities_qe_method))

    def expand_baseline(self, entities):
        candidates = []
        for entity in entities:
            new_tuples = self.qe_single_entity.expand_entity(entity,
                                                             qe_method=qe_single_entity.QEMethod.global_word2vec)
            if new_tuples:
                candidates += [tup[0] for tup in new_tuples]
        w2v_model = self.global_model
        expansions = self.filter_candidates_by_mutual_sim(candidates, entities, w2v_model)
        return ' '.join(expansions)

    def calc_best_year_globally_based_classifier(self, entities):
        best_year = None
        max_score = 0
        for y in range(self.min_year, self.max_year):
            feature_vector = utils.create_feature_vector(entities, y, global_model=self.global_model,
                                                         include_peak_detection=False)
            if feature_vector is None:
                continue
            score = self.svm_model.predict_proba([feature_vector])[0][1]  # probability for the true class
            if score > max_score:
                max_score = score
                best_year = y
        return best_year

    def expand_entities_generic(self, entities, year):
        if not year:
            return None
        logging.warning('{} @ {}'.format(' , '.join(entities), year))
        candidates = []
        for entity in entities:
            new_tuples = self.qe_single_entity.expand_entity(entity, year, topk=100)
            if new_tuples:
                candidates += [tup[0] for tup in new_tuples]
        w2v_model = self.year_to_model[year]
        expansions = self.filter_candidates_by_mutual_sim(candidates, entities, w2v_model)
        return ' '.join(expansions)

    def calc_best_year_temporal_classifier(self, entities):
        best_year = None
        max_score = 0
        for y in range(self.min_year, self.max_year):
            feature_vector = utils.create_feature_vector(entities, y, year_to_model=self.year_to_model,
                                                         include_peak_detection=False)
            if feature_vector is None:
                continue
            score = self.svm_model.predict_proba([feature_vector])[0][1]  # probability for the true class
            if score > max_score:
                max_score = score
                best_year = y
        return best_year

    def calc_best_year_temporal_model_classifier(self, entities):
        best_year = None
        max_score = 0
        for y in range(self.min_year, self.max_year):
            feature_vector = utils.create_feature_vector(entities, y, year_to_model=self.year_to_model,
                                                         include_peak_detection=True)
            if feature_vector is None:
                continue
            score = self.svm_model.predict_proba([feature_vector])[0][1]  # probability for the true class
            if score > max_score:
                max_score = score
                best_year = y
        return best_year

    def calc_best_year_peak_detection(self, entities):
        # for each pair of entities (unordered):
        pairs = list(itertools.combinations(entities, 2))
        best_years = []
        for pair in pairs:
            peak_years = utils.find_peaks(pair[0], pair[1], self.year_to_model)
            longest_sequence = utils.find_longest_sequence(peak_years)
            if not longest_sequence:  # if didn't find any peak
                return None
            middle_year = utils.get_middle_year(longest_sequence[0], longest_sequence[-1])
            best_years.append(middle_year)
        avg_best_year = round(np.array(best_years).mean())
        return avg_best_year

    def expand_peak_detection_new(self, entity1, entity2):
        peak_years = utils.find_peaks(entity1, entity2, self.year_to_model)
        longest_sequence = utils.find_longest_sequence(peak_years)
        if not longest_sequence:
            return None
        middle_year = utils.get_middle_year(longest_sequence[0], longest_sequence[-1])
        related_tuples1 = self.qe_single_entity.expand_entity(entity1, middle_year, topk=100)
        related_tuples2 = self.qe_single_entity.expand_entity(entity2, middle_year, topk=100)
        if related_tuples1 is None or related_tuples2 is None:  # if one of the terms was not expanded
            return None
        related_tuples = []
        for tup in related_tuples1:
            peak_years = utils.find_peaks(entity1, tup[0], self.year_to_model)
            if utils.is_overlapping(peak_years, longest_sequence):
                related_tuples.append(tup)
        for tup in related_tuples2:
            peak_years = utils.find_peaks(entity2, tup[0], self.year_to_model)
            if utils.is_overlapping(peak_years, longest_sequence):
                related_tuples.append(tup)
        if related_tuples is None:  # if there are no relevant related terms
            return None
        # reorder the related terms by (mutual) similarity with both entities
        heap = MaxEntitiesHeap(self.k * 2, [entity1, entity2])
        w2v_model = self.year_to_model[middle_year]
        for tup in related_tuples:
            term = tup[0]
            sim1 = w2v_model.model.similarity(term, entity1)
            sim2 = w2v_model.model.similarity(term, entity2)
            mutual_similarity = sim1 + sim2
            if term in (entity1, entity2) or term in [tup[1] for tup in heap.heap]:  # if this term already exists
                continue
            heap.add(mutual_similarity, term)
        expansions = []
        for obj in heap.heap:
            expansions.append(obj[1])
        return ' '.join(expansions)

    def filter_candidates_by_mutual_sim(self, candidates, entities, w2v_model):
        # sort the candidate list by mutual similarity, and return the top k
        heap = MaxEntitiesHeap(self.k, entities)
        for term in candidates:
            sim = 0
            for entity in entities:
                if w2v_model.contains_all_words([entity]):
                    sim += w2v_model.model.similarity(term, entity)
            heap.add(sim, term)
        expansions = []
        for item in heap.heap:
            expansions.append(item[1])
        return expansions
