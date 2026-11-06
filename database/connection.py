"""
BioPrint Vision — Database Connection Module
Initializes SQLAlchemy instance and provides helper functions for database management.
"""

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def init_db(app):
    """Bind SQLAlchemy to Flask app and create all tables if needed."""
    db.init_app(app)
    with app.app_context():
        db.create_all()
