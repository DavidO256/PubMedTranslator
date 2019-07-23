import tensorflow as tf
import numpy as np
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


def process_inputs(inputs, vocabulary, settings):
    result = list()
    for token in nltk.wordpunct_tokenize(inputs):
        result.append(0 if token not in vocabulary else 1 + vocabulary.index(token))
    return np.asarray(result[:settings['encoder_inputs']]
                      + [0] * (settings['encoder_inputs'] - len(result)))

def process_outputs(outputs, vocabulary, settings):
    result = list()
    for token in nltk.wordpunct_tokenize(outputs):
        y = np.zeros(len(vocabulary))
        if token in vocabulary:
            y[vocabulary.index(token)] = 1
        result.append(y)
    return np.asarray(result[:settings['encoder_inputs']]
                      + [np.zeros(len(vocabulary))] * (settings['encoder_inputs'] - len(result)))

def parse_outputs(outputs, vocabulary, settings):
    result = list()
    for vector in outputs:
        result.append(vocabulary[sum(vector * np.arange(len(vocabulary),))])
    return " ".join(result)

def load_corpus(path):
    corpus = list()
    with open(path) as f:
        for sample in f:
            corpus.append(sample.replace("\n", str()))
    return corpus


class Dataset(tf.keras.utils.Sequence):

    def __init__(self, x_corpus, y_corpus, x_vocab, y_vocab, settings):
        self.x_corpus = x_corpus
        self.y_corpus = y_corpus
        self.x_vocab = x_vocab
        self.y_vocab = y_vocab
        self.settings = settings

    def __len__(self):
        return int(math.ceil(len(self.x_corpus) // self.settings['batch_size']))

    def __getitem__(self, index):
        start = index * self.settings['batch_size']
        end = (index + 1) * self.settings['batch_size']
        x = [process_inputs(sample, self.x_vocab, self.settings) for sample in self.x_corpus[start:end]]
        y = [process_inputs(sample, self.y_vocab, self.settings) for sample in self.y_corpus[start:end]]
        for n in range(len(x)):
            if len(x[n]) == 0 or len(y[n]) == 0:
                x.pop(n)
                y.pop(n)
        return np.asarray(x), np.asarray(y)
