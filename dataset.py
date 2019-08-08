from xml.etree import ElementTree
from contextlib import closing
import codecs
import gzip
import os


SPLIT_RATIO = 0.9
BASE_DIRECTORY = "./"


def load_target_articles(path):
    with open(path) as f:
        target_articles = [line.replace('\n', str()) for line in f if len(line) > 0]
        f.close()
    return target_articles


def xml_generator(directory):
    for path in filter(lambda s: s.endswith(".xml.gz"), os.listdir(directory)):
        with gzip.open(os.path.join(directory, path), mode='rt', encoding='utf-8', errors='replace') as f:
            print("Loading", path)
            root = ElementTree.parse(f).getroot()
        f.close()
        yield root


def find_article_data(root, target_articles):
    if root.tag != 'PubmedArticleSet':
        print(root)
        raise ValueError("Invalid root provided.")
    collected = dict()
    for article in root:
        try:
            pmid = next(article.iter('PMID')).text
            if pmid in target_articles:
                collected[pmid] = {'ArticleTitle': next(article.iter('ArticleTitle')).text,
                                   'VernacularTitle': next(article.iter('VernacularTitle')).text}
        except StopIteration as error:
            print(error)
    if len(collected) > 0:
        print("Found", len(collected), "target articles.")
    return collected
        

def process_baseline(target_articles, directory):
    found = set()
    with closing(codecs.open(os.path.join(BASE_DIRECTORY, "outputs.txt"), 'w', encoding='utf', errors='replace')) as f:
        for article_set in xml_generator(directory):
            collected = find_article_data(article_set, target_articles)
            if len(collected) > 0:
                for pmid in collected:
                    if pmid not in found:
                        f.write(pmid + "|{ArticleTitle}|{VernacularTitle}\n".format(**collected[pmid]))
                found |= set(collected)
                f.flush()
            if len(found) == len(target_articles):
                break


def split_data(corpus):
    data = tuple(map(lambda l: l.split("|")[1:3], corpus))
    ratio = SPLIT_RATIO * len(data)
    validation = list()
    train = list()
    count = 0
    for sample in data:
        if count < ratio:
            train.append(sample)
        else:
            validation.append(sample)
        count += 1
    return train, validation


def save_dataset(directory, dataset, name):
    def save(path, data):
        with closing(codecs.open(path, 'w', encoding='utf-8', errors='replace')) as output:
            output.write("\n".join(data))
    x, y = zip(*dataset)
    save(os.path.join(directory, name + "_x.txt"), x)
    save(os.path.join(directory, name + "_y.txt"), y)
    

def load_corpus(path):
    with closing(codecs.open(path, encoding='utf-8', errors='replace')) as f:
        return [line.replace('\n', str()) for line in f]


if __name__ == '__main__':
    corpus = baseline(load_target_articles(os.path.join(BASE_DIRECTORY, "articles.txt")), os.path.join(BASE_DIRECTORY, "baseline"))
    train_dataset, validation_dataset = split_data(corpus)
    save_dataset(os.path.join(BASE_DIRECTORY, "datasets"), validation_dataset, "validation")
    save_dataset(os.path.join(BASE_DIRECTORY, "datasets"), train_dataset, "train")

