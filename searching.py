from flask import (
    Blueprint, render_template, request
)
from flask_paginate import Pagination, get_page_parameter
from app.tools import get_searching_results, get_ranking_results, combine
from app.global_parameters import (
    model_w2v,
    model_d2v,
    model_fst,
    model_glv,
    doc_vector_list_w2v,
    doc_vector_list_fst,
    doc_vector_list_glv,
    stoplist)


bp = Blueprint('search', __name__, template_folder='templates')


@bp.route('/test')
def test():
    return 'Search engine started.'


@bp.route('/test2')
def test2():
    return render_template('searching.html')


@bp.route('/')
@bp.route('/homepage', methods=['GET', 'POST'])
def searching():
    search = False
    q = request.args.get('q')
    if q:
        search = True

    key_words = request.args.get("search")

    try:
        entity_filter = request.args.get("entity_filter")
    except:
        pass

    page = request.args.get(get_page_parameter(), type=int, default=1)
    per_page = 1
    offset = (page - 1) * per_page

    if key_words is None:
        doc_list_w2v, doc_list_d2v, doc_list_ftv, doc_list_glv, doc_list_rnk, map_repeat = [], [], [], [], [], {}
    else:
        # key_words = request.args.get("search")
        if key_words.replace(' ', '') is '':
            doc_list_w2v, doc_list_d2v, doc_list_ftv, doc_list_glv, doc_list_rnk, map_repeat = [], [], [], [], [], {}
        else:
            doc_list_w2v, doc_list_d2v, doc_list_ftv, doc_list_glv, doc_list_rnk, map_repeat = get_searching_results(
                key_words,
                entity_filter,
                model_w2v,
                model_d2v,
                model_fst,
                model_glv,
                doc_vector_list_w2v,
                doc_vector_list_fst,
                doc_vector_list_glv,
                stoplist)



    # doc_list_w2v, doc_list_d2v, doc_list_ftv, doc_list_glv, doc_list_rnk = [], [], [], [], []

    c_list = combine([doc_list_w2v, doc_list_d2v, doc_list_ftv, doc_list_glv, doc_list_rnk], map_repeat)

    pagination = Pagination(page=page,
                            total=len(doc_list_w2v),
                            search=search,
                            record_name='all_modules_c',
                            per_page=per_page,
                            show_single_page=True,
                            link='<li><a class="pgn__num" href="{0}">{1}</a></li>')

    pagination.current_page_fmt = '<li><span class="pgn__num current">{0}</span></li>'
    pagination.prev_page_fmt = '<li><a class="pgn__prev" href="{0}">{1}</a></li>'
    pagination.next_page_fmt = '<li><a class="pgn__next" href="{0}">{1}</a></li>'
    pagination.gap_marker_fmt = '<li><span class="pgn__num dots">…</span></li>'
    pagination.link = '<li><a class="pgn__num" href="{0}">{1}</a></li>'
    pagination.link_css_fmt = '<div class="{0}{1}"><ul>'
    pagination.prev_disabled_page_fmt = ''
    pagination.next_disabled_page_fmt = ''

    return render_template('searching.html',
                           c_list=c_list,
                           # doc_list_w2v=doc_list_w2v,
                           # doc_list_d2v=doc_list_d2v,
                           # doc_list_ftv=doc_list_ftv,
                           # doc_list_glv=doc_list_glv,
                           # doc_list_rnk=doc_list_rnk,
                           pagination=pagination,
                           search=key_words)
