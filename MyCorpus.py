from gensim import utils


class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""
    corpus_path = ''

    def __init__(self, path_corpus_processed):
        self.corpus_path = path_corpus_processed

    def __iter__(self):
        # corpus_path = datapath(path_corpus_processed)
        for line in open(self.corpus_path):
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line)