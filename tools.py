import re
import copy
import gensim
import smart_open
import numpy as np
from app.MyDoc import MyDoc
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from app.models import Author, Subject, Corpus, add_corpus


def combine(packed_list, map_repeat):
    result = []
    color_list = ['#00ff69', '#aeffcf', '#008eff', '#94d0ff', '#b400ff', '#df92ff', '#a07d00', '#ecd274', '#f1f400',
                  '#feffa0']
    color_map = {}

    while True:
        if len(packed_list[0]) == 0 and len(packed_list[1]) == 0 and len(packed_list[2]) == 0 and len(
                packed_list[3]) == 0:
            break

        tmp = []
        for l in packed_list:
            if len(l) > 0:
                doc_obj = l.pop(0)
                try:
                    doc_obj.repeat = map_repeat[doc_obj.id]
                except:
                    pass
                if doc_obj.repeat >= 1:
                    # doc_obj.color = "#{:06x}".format(0xFFFFFF & (doc_obj.repeat * 10000))

                    if doc_obj.id not in color_map.keys():
                        color_map[doc_obj.id] = color_list.pop(0)

                    doc_obj.color = color_map[doc_obj.id]
                else:
                    doc_obj.color = "#0xFFFFFF"
                tmp.append(doc_obj)
            else:
                tmp.append([])

        result.append(tmp)

    return result



def check(word_score, doc_list_ranking):
    for i in doc_list_ranking:
        if i[0] is word_score[0]:
            return False
    return True


def get_ranking_results(ranking_list):
    ranking_list_tmp = copy.deepcopy(ranking_list)
    doc_list_ranking = []
    doc_list_credits = []
    map_repeat = {}

    for m in ranking_list_tmp:
        credits = 9
        for n in m:
            try:
                entity = int(n[0])
            except:
                entity = n[0]
            doc_list_ranking.append(entity)

            doc_list_credits.append(credits)

            credits -= 1

    result = []
    while True:
        if len(doc_list_ranking) is 0:
            break

        credits_tmp = doc_list_credits.pop(0)
        ranking_tmp = doc_list_ranking.pop(0)
        index = 0
        remove_list = []
        while True:
            try:
                doc_list_ranking[index]
            except IndexError:
                break

            if ranking_tmp == doc_list_ranking[index]:
                credits_tmp += doc_list_credits[index]

                # 计算搜索结果的结果重复数量
                if index in map_repeat.keys():
                    map_repeat[ranking_tmp] += 1
                else:
                    map_repeat[ranking_tmp] = 1

                remove_list.append(index)
            index+=1

        remove_list.reverse()
        for r in remove_list:
            doc_list_ranking.pop(r)
            doc_list_credits.pop(r)

        result.append((ranking_tmp, credits_tmp))

    result.sort(key=lambda k: k[1], reverse=True)

    return result[0:10], map_repeat


def get_ranking_results_02(ranking_list):
    ranking_list_tmp = copy.deepcopy(ranking_list)
    doc_list_ranking = []

    while True:
        if len(ranking_list_tmp[0]) == 0 and len(ranking_list_tmp[1]) == 0 and len(ranking_list_tmp[2]) == 0 and len(ranking_list_tmp[3]) == 0:
            break

        for l in ranking_list_tmp:
            if len(l) > 0:
                tmp = l.pop(0)
                if check(tmp, doc_list_ranking):
                    doc_list_ranking.append(tmp)
                    if len(doc_list_ranking) is 10:
                        return doc_list_ranking

    return doc_list_ranking


def get_searching_results(key_words,
                          entity_filter,
                          model_w2v,
                          model_d2v,
                          model_fst,
                          model_glv,
                          doc_vector_list_w2v,
                          doc_vector_list_fst,
                          doc_vector_list_glv,
                          stoplist):
    if key_words is '':
        return [], [], [], []

    if key_words is None:
        return [], [], [], []

    doc_num = 10

    if entity_filter == 'article':
        key_words_vec_w2v = vectorize(key_words, model_w2v)
        key_words_vec_d2v = vectorize_d2v(key_words, model_d2v, stoplist)
        key_words_vec_fst = vectorize(key_words, model_fst)
        key_words_vec_glv = vectorize(key_words, model_glv)

        doc_list_w2v = get_similar_docs(key_words_vec_w2v, doc_vector_list_w2v, doc_num)
        doc_list_d2v = get_similar_docs_d2v(key_words_vec_d2v, model_d2v, doc_num)
        doc_list_ftv = get_similar_docs(key_words_vec_fst, doc_vector_list_fst, doc_num)
        doc_list_glv = get_similar_docs(key_words_vec_glv, doc_vector_list_glv, doc_num)
        doc_list_rnk, map_repeat = get_ranking_results([doc_list_w2v, doc_list_d2v, doc_list_ftv, doc_list_glv])

        doc_list_w2v_packed = pack(doc_list_w2v, 1)
        doc_list_d2v_packed = pack(doc_list_d2v, 1)
        doc_list_ftv_packed = pack(doc_list_ftv, 1)
        doc_list_glv_packed = pack(doc_list_glv, 1)
        doc_list_rnk_packed = pack(doc_list_rnk, 1)

        return doc_list_w2v_packed, doc_list_d2v_packed, doc_list_ftv_packed, doc_list_glv_packed, doc_list_rnk_packed, map_repeat
        # return [], doc_list_d2v_packed, [], [], []

    if entity_filter == 'author' or entity_filter == 'terms' or entity_filter == 'subject':
        keywords_processed = key_words.replace(' ', '_')
        prefix = entity_filter + ':'
        prefix02 = entity_filter + '_'

        if entity_filter == 'terms':
            prefix = ''
            prefix02 = ''

        key_words_vec_w2v = vectorize(key_words, model_w2v)
        key_words_vec_fst = vectorize(key_words, model_fst)
        key_words_vec_glv = vectorize(key_words, model_glv)

        word_list_w2v = get_similar_entities(model_w2v, key_words_vec_w2v, prefix, prefix02)
        word_list_fst = get_similar_entities(model_fst, key_words_vec_fst, prefix, prefix02)
        word_list_glv = get_similar_entities(model_glv, key_words_vec_glv, prefix, prefix02)
        word_list_rnk, map_repeat = get_ranking_results([word_list_w2v, [], word_list_fst, word_list_glv])

        word_list_w2v_packed = pack(word_list_w2v, 2)
        word_list_fst_packed = pack(word_list_fst, 2)
        word_list_glv_packed = pack(word_list_glv, 2)
        word_list_rnk_packed = pack(word_list_rnk, 2)

        return word_list_w2v_packed, [], word_list_fst_packed, word_list_glv_packed, word_list_rnk_packed, map_repeat


def get_similar_entities(model, keywords_processed, prefix, prefix02):
    word_embedding_list = []
    try:
        if prefix is '':
            word_list_tmp = model.most_similar(positive=[keywords_processed], topn=10)
            return word_list_tmp

        l = len(model.wv.vectors)
        word_list_tmp = model.most_similar(positive=[keywords_processed], topn=l)

        word_embedding_list = []
        i = 0
        for word_score in word_list_tmp:
            if i >= 10:
                break
            if (prefix in word_score[0]) or (prefix02 in word_score[0]):
                word_embedding_list.append(word_score)
                i = i + 1

        # word_embedding_list = pack(word_embedding_list)
    except:
        pass

    return word_embedding_list


def get_similar_entities_02(model, keywords_processed, prefix):
    word_embedding_list = []
    try:
        l = len(model.wv.vectors)
        word_list_tmp = model.most_similar(positive=[keywords_processed], topn=l)

        word_embedding_list = []
        i = 0
        for word_score in word_list_tmp:
            if i >= 10:
                break
            if prefix in word_score[0]:
                word_embedding_list.append(word_score)
                i = i + 1

        # word_embedding_list = pack(word_embedding_list)
    except:
        pass

    return word_embedding_list

# type: 1==doc_embedding, 2==entity_embedding
def pack(embedding_list, type):
    result = []

    for i in embedding_list:
        if type is 1:
            index = int(i[0])
            bbb = Corpus.get_title_by_id(index+1)
            doc_obj = MyDoc(index, bbb, round(i[1], 2), 'https://www.google.nl/')
        else:
            bbb = i[0]
            doc_obj = MyDoc(i, bbb, round(i[1], 2), 'https://www.google.nl/')

        result.append(doc_obj)

    return result


def get_similar_docs_d2v(keywords_vector, model_d2v, doc_num):
    result = model_d2v.docvecs.most_similar([keywords_vector], topn=doc_num)

    # result = []
    # for item in sims:
    #     document_number = int(item[0]) + 1
    #     score = round(item[1], 2)
    #
    #     doc_obj = MyDoc(document_number, Corpus.get_title_by_id(document_number), score, 'https://www.google.nl/')
    #     result.append(doc_obj)
    return result


def get_similar_docs(keywords_vector, doc_vector_list, doc_num):
    doc_embedding_list = []
    i = 0
    for doc_vec in doc_vector_list:
        score = np.dot(keywords_vector, doc_vec) / (np.linalg.norm(keywords_vector) * np.linalg.norm(doc_vec))
        if np.isnan(np.sum(score)):
            score = 0

        # score_list.append({"doc_id": i, "score": score})
        doc_embedding_list.append((i, score))
        # Sort results by score in desc order

        i = i + 1

    doc_embedding_list.sort(key=lambda k: k[1], reverse=True)

    return doc_embedding_list[0:doc_num]


# def vectorize_d2v_weights(keywords, model):
#     keywords = keywords.lower()
#     keywords_vec = []
#
#     for keyword in keywords:
#         try:
#             vec = model.wv[keyword]
#             weights = dict_weights[keyword]
#             vec = vec * weights
#         except KeyError:
#             vec = model.infer_vector(list(keywords.split()))
#             pass
#
#         keywords_vec.append(vec)
#
#     return np.mean(keywords_vec, axis=0)
#
#
# def vectorize_weights(keywords, model):
#     keywords = keywords.lower()
#     keywords_vec = []
#
#     for keyword in keywords:
#         try:
#             vec = model.wv[keyword]
#             weights = dict_weights[keyword]
#             vec = vec * weights
#             keywords_vec.append(vec)
#         except KeyError:
#             pass
#
#     return np.mean(keywords_vec, axis=0)


def vectorize_d2v(keywords, model, stoplist):
    vector = [word for word in keywords.lower().split() if word not in stoplist]
    return model.infer_vector(vector)


def vectorize(keywords, model):
    keywords = keywords.lower().split(' ')
    keywords_vec = []

    for keyword in keywords:
        try:
            vec = model.wv[keyword]
            keywords_vec.append(vec)
        except KeyError:
            pass

    return np.mean(keywords_vec, axis=0)


def get_corpus_list(path_corpus):
    result = []
    with open(path_corpus, 'r') as f:
        line = f.readline()
        while line:
            entities = re.split('\]\s*\[|\s*\[|\]\s*', line)
            entities.remove('')
            doc_id = entities[0]
            entities.remove(doc_id)

            for entity in entities:
                kv = entity.split(':')
                if 'abstract' is kv[0]:
                    result.append(kv[1])

    return result


# # 生成doc vectors (doc embedding)
# def gen_doc_embedding_weights(path_doc_vec, model):
#     # 生成doc_vectors
#     with open(path_corpus_processed, 'r') as f_corpus:
#         with open(path_doc_vec, 'w+') as f_doc_vectors:
#             line = f_corpus.readline()
#             i = 0
#             while line:
#                 entities = re.split('\s', line)
#                 entities.remove('')
#
#                 # 给value生成向量
#                 doc_vec_list = []
#                 for entity in entities:
#                     try:
#                         vec = model.wv[entity]
#                         try:
#                             weights = dict_weights[entity]
#                         except:
#                             weights = 1.0
#                         model.wv[entity] = vec * weights
#                         vec = vec * weights
#                         doc_vec_list.append(vec)
#                     except KeyError:
#                         pass
#
#                 doc_vec = np.mean(doc_vec_list, axis=0)
#
#                 np.savetxt(f_doc_vectors, [doc_vec])
#                 line = f_corpus.readline()
#                 i=i+1


# 生成doc vectors (doc embedding)
def gen_doc_embedding(path_corpus_processed, path_doc_vec, model):
    # 生成doc_vectors
    with open(path_corpus_processed, 'r') as f_corpus:
        with open(path_doc_vec, 'w+') as f_doc_vectors:
            line = f_corpus.readline()
            i = 0
            while line:
                entities = re.split('\s', line)
                try:
                    entities.remove('')
                except:
                    print('This is a empty!!!')

                # 给value生成向量
                doc_vec_list = []
                for entity in entities:
                    try:
                        vec = model.wv[entity]
                        doc_vec_list.append(vec)
                    except KeyError:
                        print('KeyError: ' + entity)

                print('calaulate doc vector mean value.')
                doc_vec = np.mean(doc_vec_list, axis=0)
                print('calaulate doc vector mean value done.')

                # if 40190 == i:
                #     print(str(i)+':')
                #     print(doc_vec)

                print('save doc vector mean value.')
                np.savetxt(f_doc_vectors, [doc_vec])
                print('save doc vector mean value.')

                line = f_corpus.readline()
                i = i + 1
                print(str(i) + ' docs done.')


# 把corpus文件导入数据库
def load_corpus_to_db(path_corpus):
    with open(path_corpus, 'r') as f_corpus:
        line = f_corpus.readline()
        i = 0
        while line:
            if i == 10:
                break
            entities = re.split('\]\s*\[|\s*\[|\]\s*', line)
            entities.remove('')

            corpus = None
            author_list = []
            subject_list = []

            num = entities[0]
            corpus_id = i
            title = ''
            abstract = ''
            lang = ''
            doi = ''
            type = ''
            pmid = ''

            for entity in entities:
                tmp = entity.split(':')

                if 'title' == tmp[0]:
                    title = tmp[1]
                if 'abstract' == tmp[0]:
                    abstract = tmp[1]
                if 'lang' == tmp[0]:
                    lang = tmp[1]
                if 'doi' == tmp[0]:
                    doi = tmp[1]
                if 'type' == tmp[0]:
                    type = tmp[0]
                if 'pmid' == tmp[0]:
                    pmid = tmp[1]
                if 'author' == tmp[0]:
                    author = Author(num, tmp[1])
                    author_list.append(author)
                if 'subject' == tmp[0]:
                    subject = Subject(num, tmp[1])
                    subject_list.append(subject)

            corpus = Corpus(corpus_id, num, title, abstract, lang, doi, type, pmid)

            add_corpus(corpus, author_list, subject_list)

            line = f_corpus.readline()


# frequency的vocab
# def doc2vec():
#     path_corpus = './static/data/corpus/corpus_10.txt'
#
#     corpus_list = list(read_corpus(path_corpus))
#
#     model = gensim.models.doc2vec.Doc2Vec(vector_size=256, min_count=0, epochs=40)
#     model.build_vocab_from_freq(frequency_dict)
#
#     tmp = model.wv.vectors
#     len_01 = len(tmp)
#     len_02 = len(dict_weights)
#     tmp_02 = list(dict_weights.keys())
#     if len_01 == len_02:
#         i = 0
#         for entity in dict_weights:
#             if i == 10:
#                 break
#             entity_id = tmp_02.index(entity)
#             tmp[entity_id] = tmp[entity_id]*dict_weights[entity]
#             i = i + 1
#     else:
#         return 'failed!'
#
#     model.train(corpus_list, total_examples=len(corpus_list), epochs=model.epochs)
#     model.save(path_model_d2v)
#
#     return "success!"


def read_corpus(fname, tokens_only=False):
    with smart_open.open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            tokens = gensim.utils.simple_preprocess(line)

            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])


def get_dict_entity_list_frequency_list(path_vocab):
    entity_list = []
    frequency_list = []

    with open(path_vocab) as f_vocab:
        line = f_vocab.readline()
        while line:
            tmp_list = line.split('\t')
            entity_list.append(tmp_list[0])
            frequency_list.append(int(tmp_list[1]))

            line = f_vocab.readline()

    return entity_list, frequency_list


def get_sub_dict(span, entity_list, frequency_list):
    i = 0
    result = {}

    while i < span:
        try:
            k = entity_list.pop(0)
            v = frequency_list.pop(0)
            result[k] = v
        except:
            break

        i = i + 1

    return result


def get_stopwords():
    words = stopwords.words('english')

    for w in ['!', ',', '.', '?', '-s', '-ly', '</s>', 's']:
        words.stopwords.add(w)

    return words

    # filtered_words = [word for word in word_list if word not in stopwords.words('english')]


if __name__ == '__main__':
    a, b = get_dict_entity_list_frequency_list()

    print('asdf')
