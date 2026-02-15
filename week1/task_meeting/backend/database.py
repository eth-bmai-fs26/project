import os
import sqlite3
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv

load_dotenv()

db = SQLAlchemy()

class Employee(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    role = db.Column(db.String(120), nullable=False)

    def __repr__(self):
        return f'<Employee {self.name}>'

def init_db(app):
    db.init_app(app)
    with app.app_context():
        db.create_all()
        
        # Seed data if empty
        if not Employee.query.first():
            employees = [
                Employee(id=0, name="Employee 0", role="Doctor"),
                Employee(id=1, name="Employee 1", role="Nurse"),
                Employee(id=2, name="Employee 2", role="Lab Technician"),
                Employee(id=3, name="Employee 3", role="Pharmacist"),
                Employee(id=4, name="Employee 4", role="Administrator")
            ]
            db.session.bulk_save_objects(employees)
            db.session.commit()
            print("Database initialized and seeded.")

if __name__ == '__main__':
    # Standalone run to init DB
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///employees.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    init_db(app)
