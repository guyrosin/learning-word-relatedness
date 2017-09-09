# Learning Word Relatedness over Time

## Author: Guy Rosin (guyrosin@cs.technion.ac.il)

## Under construction... code and more data coming soon!

This repository provides the data and implementation of the paper:
>Learning Word Relatedness over Time<br>
>Guy D. Rosin, Eytan Adar and Kira Radinsky<br>
>EMNLP 2017<br>
>https://arxiv.org/abs/1707.08081


## Data
- Relations, in the format of: <entity1, entity2, start_year, end_year, relation_type>
- Binary relations that were generated from the relations file, in the format of: <entity1, entity2, year, true/false>

## Code Dependencies

- Python 3.5
- gensim
- spacy
- sklearn
