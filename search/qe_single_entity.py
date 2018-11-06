import utils
from search.max_entities_heap import MaxEntitiesHeap


class QEMethod(utils.AutoNumber):
    none = ()
    global_word2vec = ()
    specific_word2vec = ()
    word2vec_with_peak_detection = ()
    word2vec_with_svm = ()
    svm_peak = ()


min_true_confidence = 0.2


class QESingleEntity:
    def __init__(self, qe_method=None, k=2, year_to_model=None, global_model=None, svm_model=None):
        self.qe_method = qe_method
        self.k = k
        self.global_model = global_model
        self.year_to_model = year_to_model
        self.svm_model = svm_model
        pass

    def expand_entity(self, entity, time=None, qe_method=None, topk=None):
        """
        applies QE with the selected method and returns the expanded query
        """
        if qe_method is None:
            qe_method = self.qe_method
        if topk is None:
            topk = self.k
        entity = entity.replace(' ', '_')

        if qe_method == QEMethod.none:
            return entity
        elif qe_method == QEMethod.global_word2vec:
            related_tuples = self.expand_entity_word2vec(entity, utils.GLOBAL_YEAR, topk=topk)
        elif qe_method == QEMethod.specific_word2vec:
            related_tuples = self.expand_entity_word2vec(entity, time, topk=topk)
        elif qe_method == QEMethod.word2vec_with_peak_detection:
            related_tuples = self.expand_entity_word2vec_with_peak(entity, time, topk=topk)
        elif qe_method == QEMethod.word2vec_with_svm:
            related_tuples = self.expand_entity_word2vec_with_svm(entity, time, topk=topk)
        elif qe_method == QEMethod.svm_peak:
            related_tuples = self.expand_entity_svm_peak(entity, time, topk=topk)
        else:
            raise ValueError('Unknown QE method: {}'.format(self.qe_method))
        return related_tuples

    def expand_entity_word2vec(self, entity, time, topk=None):
        """
        get an entity and a timestamp
        return tuples (term, score) of the k closest terms from the word2vec model of that time period
        """
        w2v_model = None
        if time == utils.GLOBAL_YEAR:
            w2v_model = self.global_model if self.global_model else self.year_to_model[time]
        elif self.year_to_model and time in self.year_to_model:
            w2v_model = self.year_to_model[time]
        if not w2v_model:
            return None
        if topk is None:
            topk = self.k
        related_tuples = None
        if entity in w2v_model.model:
            related_tuples = w2v_model.model.similar_by_word(entity, topn=topk + 1)
        return related_tuples

    def expand_entity_word2vec_with_peak(self, entity, time, topk=None):
        """
        get an entity and a timestamp
        find top 10 closest terms from the word2vec model of that time period
        for each term, check if it's a peak
        """
        w2v_model = self.year_to_model[time]
        if topk is None:
            topk = self.k
        expansions = []
        if entity in w2v_model.model:
            related_terms = w2v_model.model.most_similar(positive=entity, topk=10)
            # logging.debug("%i: '%s' is most similar to " % (time, predicted[0]))
            for related_tuple in related_terms:
                related_term = related_tuple[0]
                peak_years = utils.find_peaks(entity, related_term, self.year_to_model)
                # Count this relation as correct if the real year was identified as a peak (or close to a peak)
                if time in peak_years:
                    expansions.append(related_tuple)
                if len(expansions) >= topk:
                    break
        return expansions

    def expand_entity_word2vec_with_svm(self, entity, time, topk=None):
        """
        get an entity and a timestamp
        find top 10 closest terms from the word2vec model of that time period
        then re-order them by the SVM model's ranking, and return the top 2 tuples (term, score)
        """
        w2v_model = self.year_to_model[time]
        if topk is None:
            topk = self.k
        expansions = []
        if entity in w2v_model.model:
            related_terms = w2v_model.model.most_similar(positive=entity, topk=10)
            related_terms = [tup[0] for tup in related_terms]  # extract the terms only
            heap = MaxEntitiesHeap(topk, [entity])
            for related_term in related_terms:
                # find top terms according to the SVM
                feature_vector = utils.create_feature_vector(related_term, entity, time,
                                                             year_to_model=self.year_to_model,
                                                             global_model=self.global_model)
                if feature_vector is None:
                    continue
                score = self.svm_model.predict_proba([feature_vector])[0][1]  # probability for the true class
                if score < min_true_confidence:
                    continue
                heap.add(score, related_term)
            for obj in heap.heap:
                expansions.append(obj[1])
        return expansions

    def expand_entity_svm_peak(self, entity, time, topk=None):
        """
        get an entity and a timestamp
        return tuples (term, score) of top terms according to the SVM model (SVM with peak detection)
        """
        w2v_model = self.year_to_model[time]
        if topk is None:
            topk = self.k
        expansions = []
        if entity in w2v_model.model:
            # find top terms according to the SVM
            heap = MaxEntitiesHeap(topk, [entity])
            for term in w2v_model.model:
                feature_vector = utils.create_feature_vector(term, entity, time,
                                                             year_to_model=self.year_to_model,
                                                             global_model=self.global_model)
                if feature_vector is None:
                    continue
                score = self.svm_model.predict_proba([feature_vector])[0][1]  # probability for the true class
                if score < min_true_confidence:
                    continue
                heap.add(score, term)
            for obj in heap.heap:
                expansions.append(obj)
        return expansions
