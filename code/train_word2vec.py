import sys
import logging

import numpy as np

from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec


def main():
    logging.basicConfig(level=logging.INFO)
    sentences = LineSentence(sys.argv[1], max_sentence_length=2000)

    if len(sys.argv) < 4:
        context_window = 1000
    else:
        context_window = int(sys.argv[4])

    model = Word2Vec(sentences, size=int(sys.argv[3]), window=context_window,
        workers=4, iter=10, negative=10, sample=0, trim_rule=None)
    model.save(sys.argv[2])


if __name__ == '__main__' : main()
