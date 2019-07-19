import numpy as np
import nltk

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
    return np.asarray(result + [0] * (settings['encoder_inputs'] - len(result)))

def process_outputs(outputs, vocabulary, settings):
    result = list()
    for token in nltk.wordpunct_tokenize(outputs):
        y = np.zeros(len(vocabulary))
        if token in vocabulary:
            y[vocabulary.index(token)] = 1
        result.append(y)

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
