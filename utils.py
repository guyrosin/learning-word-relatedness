import csv
import itertools
import logging
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.parsing import preprocessing

import peak_detection

relation_types = ['holdsPoliticalPosition', 'isMarriedTo', 'produced', 'directed', 'playsFor', 'happenedIn']
# convert the relation types to one-hot encoding
relation_types_df = pd.get_dummies(pd.DataFrame(relation_types))
relation_types_dict = {rel_type: relation_types_df.iloc[relation_types.index(rel_type)].tolist() for rel_type in
                       relation_types}

UNKNOWN_TRUE_CLASS = -1
UNKNOWN_FALSE_CLASS = -2
GLOBAL_YEAR = 9999


def show_roc_graph(linewidth=2, show_legend=True):
    plt.plot([0, 1], [0, 1], color='navy', lw=linewidth, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    if show_legend:
        plt.legend(loc="lower right")
    plt.show()


def read_relations(relations_file_path, start_year=None, end_year=None, relation_types=None,
                   includes_relation_type=True):
    """Reads through a relations file."""
    relations = []
    with open(relations_file_path, encoding='utf8') as relations_file:
        rel_reader = csv.reader(relations_file)
        for rel in rel_reader:
            if len(rel) < 5:
                continue
            # each relation contains a list of entities, start & end years, and (possibly) relation name
            start_year_index = -3 if includes_relation_type else -2
            end_year_index = start_year_index + 1
            start = int(rel[start_year_index])
            end = int(rel[end_year_index])
            if (start_year and start < start_year) or (end_year and end > end_year):
                continue
            if relation_types:
                rel_type = rel[4]
                if rel_type not in relation_types:
                    continue
            relations.append(rel)
    return relations


def read_binary_relations_to_features(relations_file_name, year_to_model=None, global_model=None,
                                      include_rel_type=True, include_peak_detection=False, min_year=None,
                                      max_year=None, new_relations_file_name=None, include_unknowns=False,
                                      include_year=True):
    """
    Reads through a binary relations file and creates a feature vector.
    :param relations_file_name:
    :param year_to_model: a dictionary, mapping year to word2vec model
    :param global_model:
    :param include_rel_type: true to use the relation type as a feature (one-hot encoded)
    :param include_peak_detection: true to add a binary "is peak" feature
    :param new_relations_file_name: the relevant binary relations will be written to this file
    :param include_unknowns: false to include only relations that have sufficient data to interpret.
            true to include even insufficient ones, with an UNKNOWN class.
    :return:
    """
    if year_to_model and global_model:
        logging.critical('utils.read_binary_relations_to_features got both global_model and year_to_model!')
        return
    if not year_to_model and not global_model:
        logging.critical("utils.read_binary_relations_to_features didn't receive a model!")
        return

    relations = []
    feature_vectors = []
    with open(relations_file_name, encoding='utf8') as relations_file:
        rel_reader = csv.reader(relations_file)
        # each relation contains a list of entities, year, and relation name
        year_index = -3
        for rel in rel_reader:
            rel_year = int(rel[year_index])
            entities = rel[:year_index]
            if year_to_model and rel_year not in year_to_model:
                continue
            if (min_year and rel_year < min_year) or (max_year and rel_year > max_year):
                continue
            rel_type = rel[-2] if include_rel_type else None
            # parse the result: YES -> 1, o.w. 0
            result = int(rel[-1] == 'True')
            feature_vector = create_feature_vector(entities, rel_year, include_rel_type,
                                                   rel_type, year_to_model, global_model, include_peak_detection,
                                                   include_result=True, result=result, include_year=include_year)
            if feature_vector is not None:
                feature_vectors.append(feature_vector)
                relations.append(rel)
            elif include_unknowns:
                feature_vectors.append(np.array([result]))
                relations.append(rel)

    if include_unknowns:
        num_of_unknowns = len([vec for vec in feature_vectors if len(vec) == 1])
        feature_vector_length = next(len(vec) for vec in feature_vectors if len(vec) > 1)
        unknown_true_vector = np.zeros(feature_vector_length)
        unknown_true_vector[-1] = UNKNOWN_TRUE_CLASS
        unknown_false_vector = np.zeros(feature_vector_length)
        unknown_false_vector[-1] = UNKNOWN_FALSE_CLASS
        feature_vectors = np.array(
            [(unknown_true_vector if vec[0] else unknown_false_vector) if len(vec) == 1 else vec
             for vec in feature_vectors])
        logging.warning("Extracted relations total #: {}. Unknowns #: {}".format(len(feature_vectors), num_of_unknowns))
    else:
        logging.warning("Extracted relations #: {}".format(len(relations)))

    if new_relations_file_name:  # create a file for all (filtered) relations
        with open(new_relations_file_name, 'w', encoding='utf-8', newline='') as out_file:
            out = csv.writer(out_file)
            for rel in relations:
                out.writerow(rel)
    return relations, feature_vectors


def create_feature_vector(entities, year, include_rel_type=None, rel_type=None,
                          year_to_model=None, global_model=None,
                          include_peak_detection=True, include_result=False, result=None, include_year=True):
    """
    Constructs a feature vector for a given relation.
    :param entities:
    :param year:
    :param include_rel_type:
    :param rel_type:
    :param year_to_model:
    :param global_model:
    :param include_peak_detection:
    :param include_result:
    :param result:
    :return:
    """
    if year_to_model and global_model:
        logging.critical('utils.create_feature_vector got both global_model and year_to_model!')
        return
    if not year_to_model and not global_model:
        logging.critical("utils.create_feature_vector didn't receive a model!")
        return
    if include_rel_type and not rel_type:
        logging.critical("utils.create_feature_vector didn't receive a rel_type!")
        return
    if include_result and result is None:
        logging.critical("utils.create_feature_vector didn't receive a result!")
        return
    if year_to_model and year not in year_to_model:
        return None
    if include_rel_type:
        # convert the relation type to an ID
        rel_type = relation_types_dict.get(rel_type)
        if not include_rel_type:
            logging.critical(
                'Unknown relation type in: "%s". Make sure to update the hard-coded array... exiting.', rel_type)
            exit()

    # take the embeddings from the global/specific model
    model = global_model or year_to_model.get(year)
    if not model or not model.contains_all_words(entities):
        return None
    # construct the feature vector
    embeddings = [model.model[ent] for ent in entities]
    if include_year:
        embeddings.append([year])
    vector = np.hstack(embeddings)
    if include_rel_type:
        vector = np.append(vector, rel_type)

    if include_peak_detection:
        hist = {
            y: calc_mutual_similarity(entities, w2v_model)
            for y, w2v_model in year_to_model.items()
            if w2v_model.contains_all_words(entities)
        }

        model_identifies_as_peak = False
        if year in hist:
            peak_years = peak_detection.find_peaks(hist)
            # Count this relation as correct if all of the real years were identified as peaks
            if year in peak_years:
                model_identifies_as_peak = True
        vector = np.append(vector, [model_identifies_as_peak])

    if include_result:
        vector = np.append(vector, [result])

    return vector


def calc_mutual_similarity(entities, w2v_model):
    pairs = list(itertools.combinations(entities, 2))
    return sum(w2v_model.model.similarity(pair[0], pair[1]) for pair in pairs)


def filter_relations_with_binary_relations(relations, binary_relations):
    # go over the relations (not binary relations!)
    # and look for those that are included in the set of the binary ones
    filtered_relations = []
    for rel in relations:
        entity1 = rel[0].lower()
        entity2 = rel[1].lower()
        if any((bin_rel[0] == entity1 and bin_rel[1] == entity2) for bin_rel in binary_relations):
            filtered_relations.append(rel)
    return filtered_relations


def find_peaks(entity1, entity2, year_to_model):
    """
    :param entity1: an entity
    :param entity2: another entity
    :param year_to_model: a mapping between each year and its word2vec model
    :return: a list of peak years
    """
    hist = {
        year: w2v_model.model.similarity(entity1, entity2)
        for year, w2v_model in year_to_model.items()
        if year < GLOBAL_YEAR
        and w2v_model.contains_all_words([entity1, entity2])
    }

    return peak_detection.find_peaks(hist)


def find_longest_sequence(list_of_years):
    sequences = []
    current_seq = []
    for year in list_of_years:
        if current_seq and year > current_seq[-1] + 1:
            sequences.append(current_seq)
            current_seq = []
        current_seq.append(year)
    # save the last sequence
    if current_seq:
        sequences.append(current_seq)
    # return the longest one
    if not sequences:
        return []
    return max(sequences, key=len)


def is_contained(seq1, seq2):
    """Returns true iff the first sequence is contained in the second sequence"""
    total = len(seq1)
    overlapping = sum(year in seq2 for year in seq1)
    return overlapping / total > 0.9


def is_overlapping(seq1, seq2):
    """Returns true iff the two given sequence are overlapping (approximately)"""
    return is_contained(seq1, seq2) and is_contained(seq2, seq1)


class AutoNumber(Enum):
    def __new__(cls):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj


def get_middle_year(time_start, time_end):
    middle_year = time_start
    if time_start != time_end:
        middle_year = (time_end + time_start) // 2
    return middle_year


def preprocess_tokenize_text(text, remove_stopwords=True):
    filters = [preprocessing.strip_non_alphanum,
               preprocessing.strip_numeric,
               lambda s: preprocessing.strip_short(s, minsize=2),
               preprocessing.strip_multiple_whitespaces,
               lambda x: x.lower()]
    if remove_stopwords:
        filters.append(preprocessing.remove_stopwords)
    return preprocessing.preprocess_string(text, filters)


def tokenize(sentence, remove_stopwords=False):
    """Converts a single sentence into a list of tokens.
    This lowercases, tokenizes (to alphabetic characters only) and converts to unicode."""
    return preprocess_tokenize_text(sentence, remove_stopwords)


def to_string(obj, digits=0):
    """Gets a list or dictionary and converts to a readable string"""
    if isinstance(obj, dict):
        return [''.join([str(k), ": ", ("{0:0.%df}" % digits).format(v)]) for k, v in obj.items()]
    return [("{0:0.%df}" % digits).format(i) for i in obj]
