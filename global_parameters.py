import gensim
import numpy as np

path_corpus = './app/static/data/corpus/corpus_10_0000.txt'
path_corpus_processed = './app/static/data/corpus/corpus_10_0000_processed.txt'

# path_corpus = './app/static/data/corpus/corpus_10.txt'
# path_corpus_processed = './app/static/data/corpus/corpus_10_processed.txt'

# entity frequency weights
path_vocab = './app/static/data/corpus/vocab_166_0765'

# doc embedding
path_doc_vec_w2v = './app/static/data/models/word2vec/np_w2v_doc_embedding.txt'
# path_doc_vec_d2v = './app/static/data/models/doc2vec/np_d2v_doc_embedding.txt'
path_doc_vec_fst = './app/static/data/models/fasttext/np_ft_doc_embedding.txt'
path_doc_vec_glv = './app/static/data/models/glove/np_glv_doc_embedding.txt'

# model
path_model_w2v = './app/static/data/models/word2vec/word2vec'
path_model_d2v = './app/static/data/models/doc2vec/doc2vec'
path_model_fst = './app/static/data/models/fasttext/fasttext'
path_model_glv = './app/static/data/models/glove/glove'

path_model_glv_vectors = './app/static/data/corpus/glove/vectors.txt'
path_model_glv_vectors_w2v_pattern = './app/static/data/corpus/glove/vectors_w2v_pattern.txt'


stoplist = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]


print('Loading "word2vec word embedding" start......')
model_w2v = gensim.models.Word2Vec.load(path_model_w2v)
print('Loading "word2vec word embedding" finish.')

print('Loading "doc2vec word embedding" start......')
model_d2v = gensim.models.doc2vec.Doc2Vec.load(path_model_d2v)
print('Loading "doc2vec word embedding" finish.')

print('Loading "fasttext word embedding" start......')
model_fst = gensim.models.fasttext.FastText.load(path_model_fst)
print('Loading "fasttext word embedding" finish.')

print('Loading "GloVe word embedding" start......')
model_glv = gensim.models.KeyedVectors.load(path_model_glv)
print('Loading "GloVe word embedding" finish.')


# doc vectors list
print('Loading "word2vec doc embedding" start......')
doc_vector_list_w2v = np.loadtxt(path_doc_vec_w2v)
print('Loading "word2vec doc embedding" finish.')

print('Loading "fasttext doc embedding" start......')
doc_vector_list_fst = np.loadtxt(path_doc_vec_fst)
print('Loading "fasttext doc embedding" finish.')

print('Loading "GloVe doc embedding" start......')
doc_vector_list_glv = np.loadtxt(path_doc_vec_glv)
print('Loading "GloVe doc embedding finish.')