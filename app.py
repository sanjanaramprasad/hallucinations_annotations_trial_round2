from typing import Dict, Tuple, Sequence

from flask import Flask, jsonify, request, render_template, url_for, redirect, send_from_directory, flash
import json 
import requests
import re
import sqlalchemy
from sqlalchemy import text, create_engine, Index, MetaData, Table, select, exists
from sqlalchemy.sql import and_
from nltk.tokenize import sent_tokenize
from flask import session
from random import shuffle
from flask import Flask, render_template, redirect, url_for, request
from flask_login import LoginManager, login_user, current_user, login_required, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
import math
import os
import nltk
nltk.download('punkt_tab')
nltk.download('punkt')


application = Flask(__name__)
application.secret_key = 'super_secret_key' 

db_engine = sqlalchemy.create_engine(
    'postgresql://ufnbklscc76qho:paa07c080a16d99fe343e98dc65914f8a01384355312fcf7cb0135029f6fd2c49'
    '@c229h9evb9laqg.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com:5432/d8i8l39r762feh'
)

metadata = MetaData(bind=db_engine)
metadata.reflect()
print(metadata)
model_summaries = metadata.tables['model_summaries']
label_table = metadata.tables['label_evaluators']
label_evaluators_validated = metadata.tables['label_evaluators_validated']

n_labels_per_doc = 6
n_docs = 5

login_manager = LoginManager()
login_manager.init_app(application)


# ---------------- USER HANDLING ----------------
class User:
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def is_authenticated(self):
        return True

    def is_active(self):
        return True

    def is_anonymous(self):
        return False

    def get_id(self):
        return self.username

_users = None

def get_users():
    global _users
    if _users is None:
        current_count = 4
        _users = {
            "sanjana": User(1, "sanjana", "sanjana"),
            "elisa": User(2, "elisa", "elisa"),
            "pranav": User(3, "pranav", "pranav"),
            "rachel_usher": User(4, "rachel_usher", "rachel_usher"),
            "kathryn_kazanas": User(5, "kathryn_kazanas", "X7xHD")
        }
        # with open('user_ids.json', 'r') as fp:
        #     user_strings = json.load(fp)
        # for uid, username in user_strings.items():
        #     _users[username] = User(len(_users) + 1, username, username)
    return _users

@login_manager.user_loader
def load_user(username):
    return get_users().get(username)


# ---------------- ROUTES ----------------

@login_required
@application.route('/')
def hello():
    return redirect(url_for('login'))

@application.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return next()  # might be better to call url_for('next') but let's keep your logic

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = load_user(username)
        
        if user is not None and user.check_password(password):
            login_user(user)
            return next()
    
    return render_template('login.html')


def get_prefix(id):
        return id.split(':')[0].split('-')[0]

def get_sorted_ids(annotated_ids,
                   ids):
    grouped_ids = {'annotated': {}, 'unannotated': {}}
    
    # Separate unannotated and annotated IDs by prefix (or however you want)
    for id_ in ids:
        prefix = get_prefix(id_)
        category = 'all'
        if id_ in annotated_ids:
            category = 'annotated'
        else:
            category = 'unannotated'
        if category not in grouped_ids:
            grouped_ids[category] = {}
        if prefix not in grouped_ids[category]:
            grouped_ids[category][prefix] = []
        grouped_ids[category][prefix].append(id_)
    
    sorted_ids = []
    for categ, prefix_ids in grouped_ids.items():
        for prefix in sorted(prefix_ids.keys(), key=len, reverse=True):
            sorted_ids.extend(grouped_ids[categ][prefix])
    
    print(sorted_ids)
    return sorted_ids


def __load_all_ids():
    connection = db_engine.connect()
    model_summaries = metadata.tables['model_summaries']
    query = select([model_summaries.c.docid])
    result = connection.execute(query)
    ids = [row[0] for row in result]
    connection.close()
    return ids

def __get_annotated_ids(username,
                        label_table_name = 'label_evaluators_validated'):
    ids = __load_all_ids()
    connection = db_engine.connect()
    label_table = metadata.tables[label_table_name]
    query_annotated_ids = select([label_table.c.docid]).where(label_table.c.user_id == username).distinct()
    result_annotated_ids = connection.execute(query_annotated_ids)
    annotated_ids = {row[0] for row in result_annotated_ids}
    sorted_ids = get_sorted_ids(annotated_ids=annotated_ids,
                   ids = ids)
    connection.close()
    return annotated_ids, sorted_ids

@login_required
@application.route('/next')
def next():
    """
    Renders a page (annotate.html) showing the sorted list of doc IDs
    so the user can pick which to annotate, or automatically choose one.
    """
    
    # Annotated IDs (for this user)
    annotated_ids, sorted_ids = __get_annotated_ids(username=current_user.username)

    # Render a page with a list or auto-redirect to the first unannotated
    return render_template('annotate.html', ids=sorted_ids, annotated_ids=annotated_ids)


def __get_evaluator_annotations(selected_id, username):
    connection = db_engine.connect()

    evaluator_table = metadata.tables['label_evaluators']
    evaluator_annotations = select([evaluator_table]).where(
        and_(
            evaluator_table.c.docid == selected_id,
            
        )
    )

    label_table = metadata.tables['label_evaluators_validated'] 
    validated_annotations = select([label_table]).where(
        and_(
            label_table.c.docid == selected_id,
            label_table.c.user_id == username
        )
    )

    validated_rows = connection.execute(validated_annotations).fetchall()
    print('VALIDATED ANNS', validated_rows)
    if not validated_rows:
        validated_rows = connection.execute(evaluator_annotations).fetchall()

    annotations = []
    for row in validated_rows:
        annotations.append({
            'nonfactual_span': row['nonfactual_span'],
            'error_type': f"highlight-{row['error_type']}" if 'highlight' not in row['error_type'] else row['error_type'],
            'mistake_severity': '',
            'inference_likelihood': row['inference_likelihood'],
            'inference_knowledge': row['inference_knowledge'],
            'inference_severity': row['inference_consequence'],
            'span_uuid': row['span_uuid']
        })
    print('ANNOTATIONS', annotations)
    connection.close()
    return annotations

@login_required
@application.route('/annotate_action', methods=['POST'])
def annotate_action():
    selected_id = request.form['id']
    username = current_user.username
    model_summaries = metadata.tables['model_summaries']
    connection = db_engine.connect()
    
    # 1. Load all doc IDs & find next unannotated (same logic as before)
    all_ids = __load_all_ids()
    total_count = len(all_ids)  # total # of docs in DB
    
    annotated_ids, sorted_ids = __get_annotated_ids(username=username)
    annotated_count = len(set(annotated_ids))  # how many docs the user has annotated
    # Calculate progress (avoid division by zero):
    if total_count > 0:
        progress_percent = round((annotated_count / total_count) * 100)
    else:
        progress_percent = 0

    # 2. Determine next_id (same as before)
    idx = sorted_ids.index(selected_id)
    if idx < len(sorted_ids):
        next_id = sorted_ids[idx + 1]
    else:
        next_id = sorted_ids[0]

    # 3. Fetch the document details for selected_id
    query_doc = select([model_summaries]).where(model_summaries.c.docid == selected_id)
    result_doc = connection.execute(query_doc).fetchone()

    # 4. Existing annotations
    annotations = __get_evaluator_annotations(selected_id = selected_id,
                                              username = username)

    # 5. Format source
    source = result_doc['source']
    summary = result_doc['summary']
    model = result_doc['model']
    origin = result_doc['origin']
    benchmark_dataset_name = result_doc['benchmark_dataset_name']
    source_sentences = sent_tokenize(source)
    source_sentences = [s.capitalize() for s in source_sentences]
    source = "\n".join(source_sentences).strip()

    connection.close()

    return render_template(
        'annotation_detail.html',
        source=source,
        summary=summary,
        model=model,
        origin=origin,
        benchmark_dataset_name=benchmark_dataset_name,
        docid=selected_id,
        annotations=annotations,
        next_id=next_id,
        progress_percent=progress_percent,  # <-- pass progress data here
        annotated_count=annotated_count,    # <-- optional, if you want to display numeric
        total_count=total_count            # <-- optional
    )


@login_required
@application.route('/save_annotations', methods=['POST'])
def save_annotations():
    """
    Saves the current user’s annotations to the DB
    """
    data = request.get_json()
    username = current_user.username
    print("Data received:", data)

    with db_engine.connect() as connection:
        label_table = metadata.tables['label_evaluators_validated']
        query_annotated = sqlalchemy.select([label_table]).where(
            and_(
                label_table.c.docid == data['docid'],
                label_table.c.user_id == username
            )
        )
        existing_annotations = connection.execute(query_annotated).fetchall()
        attempt_number = 1 if existing_annotations else 0
        print('EXISTING ANNOTATIONS', existing_annotations)
        # Delete old entries if they exist
        if existing_annotations:
            delete_stmt = label_table.delete().where(
                and_(
                    label_table.c.docid == data['docid'],
                    label_table.c.user_id == username
                )
            )
            connection.execute(delete_stmt)
            print(f"Deleted existing")

        # Insert new annotations
        if not data['annotations']:
            # If no annotations, insert a row anyway to mark that user annotated
            try:
                stmt = label_table.insert().values(
                    user_id=username,
                    docid=data['docid'],
                    source=data['source'],
                    summary=data['summary'],
                    model=data['model'],
                    benchmark_dataset_name=data['benchmark_dataset_name'],
                    origin=data['origin'],
                    nonfactual_span=None,
                    error_type=None,
                    inference_likelihood=None,
                    inference_knowledge=None,
                    inference_consequence=None,
                    span_uuid=None,
                    attempt_number=1
                )
                connection.execute(stmt)
            except KeyError as e:
                print(f"KeyError: Missing key {e} in annotation")
        else:
            # We have some highlighted spans
            for annotation in data['annotations']:
                print("Annotation received:", annotation)
                try:
                    stmt = label_table.insert().values(
                        user_id=username,
                        docid=data['docid'],
                        source=data['source'],
                        summary=data['summary'],
                        model=data['model'],
                        benchmark_dataset_name=data['benchmark_dataset_name'],
                        origin=data['origin'],
                        nonfactual_span=annotation['nonfactual_span'].strip(),
                        error_type=annotation['error_type'],
                        inference_likelihood=annotation['inference_likelihood'],
                        inference_knowledge=annotation['inference_knowledge'],
                        inference_consequence=annotation['inference_consequence'],
                        span_uuid=annotation['span_uuid'],
                        attempt_number=1
                    )
                    connection.execute(stmt)
                except KeyError as e:
                    print(f"KeyError: Missing key {e} in annotation {annotation}")

    return jsonify({"success": True})


if __name__ == "__main__":
    application.run(debug=True)
