from flask import Flask

from .config import Config
from .extensions import db


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)

    with app.app_context():
        from . import models
        from .seed import seed_demo_data

        db.create_all()
        seed_demo_data()

    from .routes import main_bp

    app.register_blueprint(main_bp)
    return app
