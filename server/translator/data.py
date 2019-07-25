import tensorflow as tf
import numpy as np
import nltk


def make_vocabulary(corpus, settings):
    occurrences = dict()
    print("Traversing corpus to make vocabulary.")
    for text in corpus:
        for token in nltk.wordpunct_tokenize(text if settings['case_sensitive'] else text.lower()):
            if token not in occurrences:
                occurrences[token] = 0
            occurrences[token] += 1
    return list(sorted(occurrences, key=occurrences.get, reverse=True))[:settings['vocabulary_cutoff']]


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


class Dataset(tf.keras.utils.Sequence):

    def __init__(self, x_corpus, y_corpus, x_vocab, y_vocab, settings):
        self.x_corpus = x_corpus
        self.y_corpus = y_corpus
        self.x_vocab = x_vocab
        self.y_vocab = y_vocab
        self.settings = settings

    def __len__(self):
        return len(self.x_corpus) // self.settings['batch_size']

    def __getitem__(self, index):
        start = index * self.settings['batch_size']
        end = (index + 1) * self.settings['batch_size']
        x = [process_x(sample, self.x_vocab, self.settings['encoder_inputs']) for sample in self.x_corpus[start:end]]
        y = [process_y(sample, self.y_vocab, self.settings['encoder_inputs']) for sample in self.y_corpus[start:end]]
        assert len(x) == self.settings['batch_size'] and len(y) == self.settings['batch_size']
        return np.asarray(x), np.asarray(y)