from flask import Flask
from flask_sqlalchemy import SQLAlchemy


app = None
db = SQLAlchemy()


def create_app(test_config=None):
    global db
    global app

    app = Flask(__name__, instance_relative_config=True)
    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_pyfile('config_test.py', silent=True)

    db.init_app(app)

    from . import searching
    app.register_blueprint(searching.bp)
    from app.models import db

    return app


if __name__ == '__main__':
    create_app()
