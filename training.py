import gensim
from gensim.scripts.glove2word2vec import glove2word2vec
import smart_open
from app.MyCorpus import MyCorpus


# word2vec
def run_training_word2vec(path_corpus_processed, path_model_w2v):
    '''
    1. 读corpus
    2. 训练
    3. save
    4. 乘以weights
    '''

    '''训练模型'''
    sentences=MyCorpus(path_corpus_processed)
    model = gensim.models.Word2Vec(sentences=sentences, size=256, window=5, min_count=2, workers=4)
    model.save(path_model_w2v)

    # print(model.most_similar([u'large']))

    return "success!"


# doc2vec 辅助
def read_corpus(fname, tokens_only=False):
    with smart_open.open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            tokens = gensim.utils.simple_preprocess(line)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])


# doc2vec
def run_training_doc2vec(path_corpus_processed, path_model_d2v):
    '''Training the Model'''
    corpus_file = list(read_corpus(path_corpus_processed))
    print('Initilize d2v model.')
    model = gensim.models.doc2vec.Doc2Vec(vector_size=256, min_count=2, epochs=40)
    print('Initilize d2v model done.')

    print('Build vocab start.')
    model.build_vocab(corpus_file)
    print('Build vocab done.')

    # print('Save weighted model.')
    # model.save(path_model_d2v)
    # print('Save weighted model done.')

    print('Training model start.')
    model.train(corpus_file, total_examples=model.corpus_count, epochs=model.epochs)
    print('Training model done.')

    print('Trained d2v model saving.')
    model.save(path_model_d2v)
    print('Trained d2v model saved.')

    return "success!"


# fasttext
def run_training_fasttext(path_corpus_processed, path_model_fst):
    print('ft init start.')
    model = gensim.models.fasttext.FastText(size=256, min_count=2)
    print('ft init done.')

    print('ft build vocab start.')
    model.build_vocab(corpus_file=path_corpus_processed)
    print('ft build vocab done.')

    print('ft train start.')
    model.train(
        corpus_file=path_corpus_processed,
        epochs=model.epochs,
        total_examples=model.corpus_count,
        total_words=model.corpus_total_words
    )
    print('ft train done.')

    print('ft model save done.')
    model.save(path_model_fst)
    print('ft model save done')

    return "success!"


# GloVe
def run_training_glove(path_model_glv_vectors,
                       path_model_glv_vectors_w2v_pattern,
                       path_model_glv):
    print('glv convert start.')
    glove2word2vec(path_model_glv_vectors, path_model_glv_vectors_w2v_pattern)
    print('glv convert end.')

    print('glv training start.')
    model = gensim.models.KeyedVectors.load_word2vec_format(path_model_glv_vectors_w2v_pattern)
    print('glv training done.')

    print('glv model saving.')
    model.save(path_model_glv)
    print('glv model saved.')

    return 'Success!'