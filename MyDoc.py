class MyDoc():
    id = -1
    abstract = ''
    score = 0.0
    url = ''
    repeat = 0
    color = 'white'

    def __init__(self, id, abstract, score, url):
        self.id = id
        self.abstract = abstract
        self.score = score
        self.url = url