import tensorflow as tf
import numpy as np
from scipy.sparse import csr_matrix
import itertools
import nltk
import math


def make_vocabulary(corpus, case_sensitive, vocabulary_cutoff):
    occurrences = dict()
    print("Traversing corpus to make vocabulary.")
    for text in corpus:
        for token in nltk.wordpunct_tokenize(text if case_sensitive else text.lower()):
            if token not in occurrences:
                occurrences[token] = 0
            occurrences[token] += 1
    return list(sorted(occurrences, key=occurrences.get, reverse=True))[:vocabulary_cutoff]


def process_x(inputs, vocabulary, encoder_inputs):
    result = list()
    for token in nltk.wordpunct_tokenize(inputs):
        if len(result) == encoder_inputs:
            break
        result.append(0 if token not in vocabulary else 1 + vocabulary.index(token))
    while len(result) < encoder_inputs:
        result.append(0)
    return result         


def process_y(outputs, vocabulary, encoder_inputs):
    result = list()
    for token in nltk.wordpunct_tokenize(outputs):
        if len(result) == encoder_inputs:
            break
        y = np.full(len(vocabulary), False)
        if token in vocabulary:
            y[vocabulary.index(token)] = True
        result.append(y)
    while len(result) < encoder_inputs:
        result.append(np.full(len(vocabulary), False))
    return np.asarray(result)


def convert_raw(data, vocabulary, settings):
    result = list()
    for vector in data:
        result.append(vocabulary[sum(vector * np.arange(len(vocabulary),))])
    return " ".join(result)


def load_corpus(path):
    corpus = list()
    with open(path) as f:
        for sample in f:
            corpus.append(sample.replace("\n", str()))
    return corpus


def dataset_generator(x_corpus, y_corpus, x_vocab, y_vocab, batch_size, encoder_inputs):
    def generator():
        for index in range(len(x_corpus) // batch_size):
            start = index * batch_size
            end = (index + 1) * batch_size
            yield [process_x(sample, x_vocab, encoder_inputs) for sample in x_corpus[start:end]],\
                  [process_y(y_corpus[index], y_vocab, encoder_inputs).tolist() for sample in y_corpus[start:end]]
    return tf.data.Dataset.from_generator(generator, (tf.int64, tf.bool), (tf.TensorShape([None, encoder_inputs]), tf.TensorShape([None, encoder_inputs, len(y_vocab)])))