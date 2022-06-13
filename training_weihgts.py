import gensim
import smart_open
from app.MyCorpus import MyCorpus
from app.global_parameters import (
    path_corpus_processed,
    path_model_w2v,
    path_model_d2v,
    path_model_glv,
    path_model_fst,
    path_model_glv_vectors,
    path_model_glv_vectors_w2v_pattern,
    dict_weights
)


# word2vec
def run_training_word2vec():
    '''
    1. 读corpus
    2. 训练
    3. save
    4. 乘以weights
    '''

    '''训练模型'''
    model = gensim.models.Word2Vec(corpus_file=path_corpus_processed, size=256, window=5, min_count=1, workers=4)
    model.save(path_model_w2v)

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
def run_training_doc2vec():
    '''Training the Model'''
    corpus_file = list(read_corpus(path_corpus_processed))
    print('Initilize d2v model.')
    model = gensim.models.doc2vec.Doc2Vec(vector_size=256, min_count=1, epochs=40)
    print('Initilize d2v model done.')

    print('Build vocab start.')
    model.build_vocab(corpus_file)
    print('Build vocab done.')

    print('---------------------- multi weights start --------------------------')
    i = 0
    length = str(len(model.wv.vocab))
    for word in model.wv.vocab:
        vec = model.wv[word]
        try:
            weights = dict_weights[word]
        except:
            print('Keyerror!!!!!!!!!!!!')
            weights = 1.0
        model.wv[word] = vec*weights
        print('第 ' + str(i) + '/' + length + ' 个word*weights处理完成.')
        i=i+1

    print('---------------------- multi weights done --------------------------')

    print('Save weighted model.')
    model.save(path_model_d2v)
    print('Save weighted model done.')

    print('Training model start.')
    model.train(corpus_file, total_examples=model.corpus_count, epochs=model.epochs)
    print('Training model done.')

    print('Trained d2v model saving.')
    model.save(path_model_d2v)
    print('Trained d2v model saved.')

    return "success!"


# fasttext
def run_training_fasttext():
    model = gensim.models.fasttext.FastText(size=256)
    model.build_vocab(corpus_file=path_corpus_processed)
    model.train(
        corpus_file=path_corpus_processed,
        epochs=model.epochs,
        total_examples=model.corpus_count,
        total_words=model.corpus_total_words
    )
    model.save(path_model_fst)

    return "success!"


# GloVe
def run_training_glove():
    # gensim.scripts.glove2word2vec(path_model_glv_vectors, path_model_glv_vectors_w2v_pattern)
    model = gensim.models.KeyedVectors.load_word2vec_format(path_model_glv_vectors_w2v_pattern)
    model.save(path_model_glv)
    return 'Success!'