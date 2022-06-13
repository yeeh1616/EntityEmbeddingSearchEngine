import re
import os
import sqlite3
import sys
import gensim
import random
import numpy as np
from app.tools import gen_doc_embedding, vectorize, vectorize_d2v, \
    get_similar_docs, get_similar_docs_d2v, get_ranking_results
from app.training import (
    run_training_word2vec,
    run_training_doc2vec,
    run_training_fasttext,
    run_training_glove)

path_corpus = './static/data/corpus/corpus_10_0000.txt'
path_corpus_processed = './static/data/corpus/corpus_10_0000_processed.txt'

# path_corpus = './static/data/corpus/corpus_10.txt'
# path_corpus_processed = './static/data/corpus/corpus_10_processed.txt'

# # entity frequency weights
# path_vocab = './app/static/data/corpus/vocab_166_0765'

# doc embedding
path_doc_vec_w2v = './static/data/models/word2vec/np_w2v_doc_embedding.txt'
path_doc_vec_fst = './static/data/models/fasttext/np_ft_doc_embedding.txt'
path_doc_vec_glv = './static/data/models/glove/np_glv_doc_embedding.txt'

# model
path_model_w2v = './static/data/models/word2vec/word2vec'
path_model_d2v = './static/data/models/doc2vec/doc2vec'
path_model_fst = './static/data/models/fasttext/fasttext'
path_model_glv = './static/data/models/glove/glove'

path_model_glv_vectors = './static/data/corpus/glove/vectors.txt'
path_model_glv_vectors_w2v_pattern = './static/data/corpus/glove/vectors_w2v_pattern.txt'

stoplist = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]


# corpus预处理
def preprocess_helper(matched):
    ''' entity_filter：
    '0' == title
    '1' == abstract
    '2' == author
    '3' == type
    '4' == subject
    '''

    entity_remove_list = ['title', 'abstract']
    entity_process_list = ['lang', 'author', 'inst', 'pdate', 'issn', 'doi', 'type', 'pmid', 'subject']

    entity = (matched.group('value')).replace('[', '').replace(']', '')
    core = (entity.split(':'))[0]

    if core in entity_remove_list:
        words = entity.replace(core+':', '')

        word_list = re.split('[\s]+', words)

        result = word_list.pop(0)

        for word in word_list:
            if word not in stoplist:
                result = result + ' ' + word

        # for word in stoplist:
        #     result = result.replace(word+' ', '')
        #     result = result.replace(' '+word, '')

        return result

    if core in entity_process_list:
        result = entity.replace(' ', '_')
        return result


# corpus预处理
def preprocess(path_corpus):
    with open(path_corpus) as f_old:
        with open(path_corpus_processed, 'a') as f_new:
            line = f_old.readline()

            i=1
            while line:
                # line_processed = re.sub('\[((?!\[\]).*)\]', preprocess_helper, line)
                line_processed = re.sub('(?P<value>\[([^\[\]]*)\])', preprocess_helper, line)
                # line_processed = re.sub(r'[" "]{2,}', ' ', line_processed)
                line_processed = re.sub(r'[\s]{2,}', '\t', line_processed)
                line_processed = line_processed.replace(':', '_')
                f_new.write(line_processed)

                print('Processed ' + str(i) + ' lines.')
                line = f_old.readline()
                i=i+1

    print('Processed all lines.')


def run_preprocess():
    # 1.
    print('preprocess start.')
    preprocess(path_corpus)
    print('pregrocess done.')

    # 2. 3. 4. 5.
    print('w2v model training start.')
    run_training_word2vec(path_corpus_processed, path_model_w2v)
    print('w2v model training done.')

    print('d2v model training start.')
    run_training_doc2vec(path_corpus_processed, path_model_d2v)
    print('d2v model training done.')

    print('glove model training start.')
    run_training_glove(path_model_glv_vectors,
                       path_model_glv_vectors_w2v_pattern,
                       path_model_glv)
    print('glove model training done.')

    print('ft model training start.')
    run_training_fasttext(path_corpus_processed, path_model_fst)
    print('ft model training done.')


    print('Loading "word2vec word embedding" start......')
    model_w2v = gensim.models.Word2Vec.load(path_model_w2v)
    print('Loading "word2vec word embedding" finish.')

    # print('Loading "doc2vec doc embedding" start......')
    # model_d2v = gensim.models.doc2vec.Doc2Vec.load(path_model_d2v)
    # print('Loading "doc2vec doc embedding" finish.')

    print('Loading "fasttext word embedding" start......')
    model_fst = gensim.models.fasttext.FastText.load(path_model_fst)
    print('Loading "fasttext word embedding" finish.')

    print('Loading "GloVe word embedding" start......')
    model_glv = gensim.models.KeyedVectors.load(path_model_glv)
    print('Loading "GloVe word embedding" finish.')


    # 6. 7. 8.
    print('w2v doc embedding generation start.')
    gen_doc_embedding(path_corpus_processed, path_doc_vec_w2v, model_w2v)
    print('w2v doc embedding generation done.')

    print('fst doc embedding generation start.')
    gen_doc_embedding(path_corpus_processed, path_doc_vec_fst, model_fst)
    print('fst doc embedding generation done.')

    print('glv doc embedding generation start.')
    gen_doc_embedding(path_corpus_processed, path_doc_vec_glv, model_glv)
    print('glv doc embedding generation done.')

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


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except sqlite3.Error as e:
        print(e)

    return conn


def get_corpus_list_100(random_index_list):
    result = []
    db = '../project_database'
    conn = create_connection(db)
    cur = conn.cursor()

    for i in random_index_list:
        cur.execute("SELECT title, abstract FROM corpus WHERE id=" + str(i))
        rows = cur.fetchall()
        result.append(rows[0][0]+rows[0][1])

    return result


def get_title_list_100(random_index_list):
    result = []
    db = '../project_database'
    conn = create_connection(db)
    cur = conn.cursor()

    for i in random_index_list:
        cur.execute("SELECT title FROM corpus WHERE id=" + str(i))
        rows = cur.fetchall()
        result.append(rows[0][0])

    return result


def get_abstract_list_100(random_index_list):
    result = []
    db = '../project_database'
    conn = create_connection(db)
    cur = conn.cursor()

    for i in random_index_list:
        cur.execute("SELECT abstract FROM corpus WHERE id=" + str(i))
        rows = cur.fetchall()
        result.append(rows[0][0])

    return result


def save_txt(path, txt_list):
    with open(path, 'w+') as f:
        for txt in txt_list:
            f.write(str(txt)+'\n')


def load_txt(path, type):
    result = []
    with open(path, 'r') as f:
        line = f.readline()

        while line:
            if type is 1:
                result.append(int(line))
            else:
                result.append(line)
            line = f.readline()
    return result


def extract(bbb):
    result = []

    for i in bbb:
        result.append(i[0]+1)

    return result


def statsitcs():
    # load models
    # run_load_models()
    print('Loading "word2vec word embedding" start......')
    model_w2v = gensim.models.Word2Vec.load(path_model_w2v)
    print('Loading "word2vec word embedding" finish.')

    print('Loading "doc2vec doc embedding" start......')
    model_d2v = gensim.models.doc2vec.Doc2Vec.load(path_model_d2v)
    print('Loading "doc2vec doc embedding" finish.')

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

    while True:
        # 生成100篇文章
        random_index_list = []
        n = 0
        while True:
            if n is 100:
                break
            rnd = random.randint(1, 10000)
            if rnd not in random_index_list:
                random_index_list.append(rnd)
                n += 1

        # 查数据库 生成corpus list

        # random_index_list = load_txt('./static/data/corpus/random_index_list_100.txt', 1)
        # # corpu_list_100 = load_txt('./static/data/corpus/random_corpus_list_100.txt', 2)
        #
        # # corpus_list_100 = get_corpus_list_100(random_index_list)
        # title_list_100 = get_title_list_100(random_index_list)
        # abstract_list_100 = get_abstract_list_100(random_index_list)

        # 存txt,两个文件 id list 和 文章的list
        # save_txt('./static/data/corpus/random_index_list_100.txt', random_index_list)
        # save_txt('./static/data/corpus/random_corpus_list_100.txt', corpus_list_100)
        # save_txt('./static/data/corpus/random_title_list_100.txt', title_list_100)
        # save_txt('./static/data/corpus/random_abstract_list_100.txt', abstract_list_100)

        # random_index_list = load_txt('./static/data/corpus/random_index_list_100.txt', 1)
        # list_100 = load_txt('./static/data/corpus/random_abstract_list_100.txt', 2)

        list_100 = get_abstract_list_100(random_index_list)

        # 用title+abstract搜索
        # 统计title+abstract： id出现在结果集里就加一

        precision_counter_w2v = 0.0
        precision_counter_d2v = 0.0
        precision_counter_fst = 0.0
        precision_counter_glv = 0.0
        precision_counter_rnk = 0.0

        recall_counter_w2v = 0
        recall_counter_d2v = 0
        recall_counter_fst = 0
        recall_counter_glv = 0
        recall_counter_rnk = 0
        doc_num = 10

        m = 0
        for key_words in list_100:

            key_words_vec_w2v = vectorize(key_words, model_w2v)
            key_words_vec_d2v = vectorize_d2v(key_words, model_d2v, stoplist)
            key_words_vec_fst = vectorize(key_words, model_fst)
            key_words_vec_glv = vectorize(key_words, model_glv)

            doc_list_w2v = get_similar_docs(key_words_vec_w2v, doc_vector_list_w2v, doc_num)
            doc_list_d2v = get_similar_docs_d2v(key_words_vec_d2v, model_d2v, doc_num)
            doc_list_fst = get_similar_docs(key_words_vec_fst, doc_vector_list_fst, doc_num)
            doc_list_glv = get_similar_docs(key_words_vec_glv, doc_vector_list_glv, doc_num)
            doc_list_rnk, map_repeat = get_ranking_results([doc_list_w2v, doc_list_d2v, doc_list_fst, doc_list_glv])

            doc_list_w2v = extract(doc_list_w2v)
            doc_list_d2v = extract(doc_list_d2v)
            doc_list_fst = extract(doc_list_fst)
            doc_list_glv = extract(doc_list_glv)
            doc_list_rnk = extract(doc_list_rnk)

            if random_index_list[m] in doc_list_w2v:
                recall_counter_w2v += 1

            if random_index_list[m] in doc_list_d2v:
                recall_counter_d2v += 1

            if random_index_list[m] in doc_list_fst:
                recall_counter_fst += 1

            if random_index_list[m] in doc_list_glv:
                recall_counter_glv += 1

            if random_index_list[m] in doc_list_rnk:
                recall_counter_rnk += 1



            if random_index_list[m] == doc_list_w2v[0]:
                precision_counter_w2v += 1.0
            # elif random_index_list[m] == doc_list_w2v[1]:
            #     precision_counter_w2v += 0.9
            # elif random_index_list[m] == doc_list_w2v[2]:
            #     precision_counter_w2v += 0.8
            # elif random_index_list[m] == doc_list_w2v[3]:
            #     precision_counter_w2v += 0.7
            # elif random_index_list[m] == doc_list_w2v[4]:
            #     precision_counter_w2v += 0.6
            # elif random_index_list[m] == doc_list_w2v[5]:
            #     precision_counter_w2v += 0.5
            # elif random_index_list[m] == doc_list_w2v[6]:
            #     precision_counter_w2v += 0.4
            # elif random_index_list[m] == doc_list_w2v[7]:
            #     precision_counter_w2v += 0.3
            # elif random_index_list[m] == doc_list_w2v[8]:
            #     precision_counter_w2v += 0.2
            # elif random_index_list[m] == doc_list_w2v[9]:
            #     precision_counter_w2v += 0.1

            if random_index_list[m] == doc_list_d2v[0]:
                precision_counter_d2v += 1.0
            # elif random_index_list[m] == doc_list_d2v[1]:
            #     precision_counter_d2v += 0.9
            # elif random_index_list[m] == doc_list_d2v[2]:
            #     precision_counter_d2v += 0.8
            # elif random_index_list[m] == doc_list_d2v[3]:
            #     precision_counter_d2v += 0.7
            # elif random_index_list[m] == doc_list_d2v[4]:
            #     precision_counter_d2v += 0.6
            # elif random_index_list[m] == doc_list_d2v[5]:
            #     precision_counter_d2v += 0.5
            # elif random_index_list[m] == doc_list_d2v[6]:
            #     precision_counter_d2v += 0.4
            # elif random_index_list[m] == doc_list_d2v[7]:
            #     precision_counter_d2v += 0.3
            # elif random_index_list[m] == doc_list_d2v[8]:
            #     precision_counter_d2v += 0.2
            # elif random_index_list[m] == doc_list_d2v[9]:
            #     precision_counter_d2v += 0.1

            if random_index_list[m] == doc_list_fst[0]:
                precision_counter_fst += 1.0
            # elif random_index_list[m] == doc_list_fst[1]:
            #     precision_counter_fst += 0.9
            # elif random_index_list[m] == doc_list_fst[2]:
            #     precision_counter_fst += 0.8
            # elif random_index_list[m] == doc_list_fst[3]:
            #     precision_counter_fst += 0.7
            # elif random_index_list[m] == doc_list_fst[4]:
            #     precision_counter_fst += 0.6
            # elif random_index_list[m] == doc_list_fst[5]:
            #     precision_counter_fst += 0.5
            # elif random_index_list[m] == doc_list_fst[6]:
            #     precision_counter_fst += 0.4
            # elif random_index_list[m] == doc_list_fst[7]:
            #     precision_counter_fst += 0.3
            # elif random_index_list[m] == doc_list_fst[8]:
            #     precision_counter_fst += 0.2
            # elif random_index_list[m] == doc_list_fst[9]:
            #     precision_counter_fst += 0.1

            if random_index_list[m] == doc_list_glv[0]:
                precision_counter_glv += 1.0
            # elif random_index_list[m] == doc_list_glv[1]:
            #     precision_counter_glv += 0.9
            # elif random_index_list[m] == doc_list_glv[2]:
            #     precision_counter_glv += 0.8
            # elif random_index_list[m] == doc_list_glv[3]:
            #     precision_counter_glv += 0.7
            # elif random_index_list[m] == doc_list_glv[4]:
            #     precision_counter_glv += 0.6
            # elif random_index_list[m] == doc_list_glv[5]:
            #     precision_counter_glv += 0.5
            # elif random_index_list[m] == doc_list_glv[6]:
            #     precision_counter_glv += 0.4
            # elif random_index_list[m] == doc_list_glv[7]:
            #     precision_counter_glv += 0.3
            # elif random_index_list[m] == doc_list_glv[8]:
            #     precision_counter_glv += 0.2
            # elif random_index_list[m] == doc_list_glv[9]:
            #     precision_counter_glv += 0.1

            if random_index_list[m] == doc_list_rnk[0]:
                precision_counter_rnk += 1.0
            # elif random_index_list[m] == doc_list_glv[1]:
            #     precision_counter_rnk += 0.9
            # elif random_index_list[m] == doc_list_glv[2]:
            #     precision_counter_rnk += 0.8
            # elif random_index_list[m] == doc_list_glv[3]:
            #     precision_counter_rnk += 0.7
            # elif random_index_list[m] == doc_list_glv[4]:
            #     precision_counter_rnk += 0.6
            # elif random_index_list[m] == doc_list_glv[5]:
            #     precision_counter_rnk += 0.5
            # elif random_index_list[m] == doc_list_glv[6]:
            #     precision_counter_rnk += 0.4
            # elif random_index_list[m] == doc_list_glv[7]:
            #     precision_counter_rnk += 0.3
            # elif random_index_list[m] == doc_list_glv[8]:
            #     precision_counter_rnk += 0.2
            # elif random_index_list[m] == doc_list_glv[9]:
            #     precision_counter_rnk += 0.1


            print('m==' + str(m))
            m = m + 1

        print('end')
        # 用author搜索
        # 用type搜索
        # 用subject搜索
        # 统计用author搜索： id出现在结果集里就加一
        # 统计用type搜索： id出现在结果集里就加一
        # 统计用subject搜索： id出现在结果集里就加一





if __name__ == '__main__':
    print(os.path.abspath('main.py'))
    os.chdir(sys.path[0])
    print(os.path.abspath('main.py'))
    # run_preprocess()

    statsitcs()

    # print('Loading "word2vec word embedding" start......')
    # model_w2v = gensim.models.Word2Vec.load(path_model_w2v)
    # print('Loading "word2vec word embedding" finish.')
    #
    # print('w2v doc embedding generation start.')
    # gen_doc_embedding(path_corpus_processed, path_doc_vec_w2v, model_w2v)
    # print('w2v doc embedding generation done.')