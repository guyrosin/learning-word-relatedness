# Learning Word Relatedness over Time

### Author: Guy Rosin (guyrosin@cs.technion.ac.il)

This repository provides the data and implementation of the paper:
>Learning Word Relatedness over Time<br>
>Guy D. Rosin, Eytan Adar and Kira Radinsky<br>
>EMNLP 2017<br>
>https://arxiv.org/abs/1707.08081

## Code
The main folder contains:
 1. code for creating word embeddings using word2vec, either from a single corpus (`word2vec_model_alltime.py`), or from a temporal corpus (`models_builder.py`)
 2. framework for running and evaluating various types of ML classifiers (`classifier.py`)
 3. a peak detection algorithm that we used (`peak_detection.py`)

`search` contains code for temporal query expansion, in particular:
 1. searching the New York Times archive, using Apache Solr, and evaluating search results (`temporal_search.py`)
 2. performing temporal query expansion. The query can be either a single entity (`qe_single_entity.py`) or multiple entities (`qe_multiple_entities.py`)

## Data
- Relations, in the format of: <entity1, entity2, start_year, end_year, relation_type>
- Binary relations that were generated from the relations file, in the format of: <entity1, entity2, year, true/false>

## Dependencies

- Python 3.5
- gensim
- spacy
- sklearn
- numpy
- scikit-learn
- scipy
- pysolr
- unidecode
- matplotlib
- gensim