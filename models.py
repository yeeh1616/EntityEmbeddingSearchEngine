from app import db


class Corpus(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    num = db.Column(db.Integer)
    title = db.Column(db.String)
    abstract = db.Column(db.String)
    lang = db.Column(db.String)
    doi = db.Column(db.String)
    type = db.Column(db.String)
    pmid = db.Column(db.String)

    def __init__(self, id, num, title, abstract, lang, doi, type, pmid):
        self.id = id
        self.num = num
        self.title = title
        self.abstract = abstract
        self.lang = lang
        self.doi = doi
        self.type = type
        self.pmid = pmid

    def get_title_by_id(id):
        result = db.session.query(Corpus.title).filter(
            Corpus.id == id).first()
        return result

    def get_abstract_by_id(id):
        result = db.session.query(Corpus.abstract).filter(
            Corpus.id == id).first()
        return result


class Author(db.Model):
    corpus_id = db.Column(db.Integer, primary_key=True)
    author = db.Column(db.String)

    def __init__(self, corpus_id, author):
        self.corpus_id = corpus_id
        self.author = author


class Subject(db.Model):
    corpus_id = db.Column(db.Integer, primary_key=True)
    subject = db.Column(db.String)

    def __init__(self, corpus_id, subject):
        self.corpus_id = corpus_id
        self.subject = subject


def add_corpus(corpus, author_list, subject_list):
    db.session.add(corpus)

    for author in author_list:
        db.session.add(author)

    for subject in subject_list:
        db.session.add(subject)

    db.session.commit()
    # try:
    #     db.session.add(corpus)
    #
    #     for author in author_list:
    #         db.session.add(author)
    #
    #     for subject in subject_list:
    #         db.session.add(subject)
    #
    #     db.session.commit()
    # except:
    #     return False

    return True
