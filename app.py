"""
Main Flask application file.
"""

import os
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import pyrebase
import firebase_admin
from firebase_admin import credentials, firestore
from functools import wraps

# Import Firebase configuration
from firebase_config import config, admin_config

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Initialize Firebase
# Note: In a real application, you would replace the config with your actual Firebase config
firebase = None
auth = None
db = None

# For demo purposes, we'll use a mock implementation
# In a real application, you would uncomment and use the Firebase code below
"""
firebase = pyrebase.initialize_app(config)
auth = firebase.auth()

# Initialize Firebase Admin SDK
try:
    # Check if already initialized
    firebase_admin.get_app()
except ValueError:
    # Initialize the app
    cred = credentials.Certificate(admin_config)
    firebase_admin.initialize_app(cred)

# Get Firestore client
db = firestore.client()
"""

# Mock implementations for demo purposes
class MockAuth:
    def create_user_with_email_and_password(self, email, password):
        # In a real app, this would create a user in Firebase Auth
        return {'localId': f'user_{email.replace("@", "_").replace(".", "_")}'}
    
    def sign_in_with_email_and_password(self, email, password):
        # In a real app, this would verify credentials with Firebase Auth
        return {
            'localId': f'user_{email.replace("@", "_").replace(".", "_")}',
            'idToken': 'mock_token'
        }

class MockFirestore:
    def __init__(self):
        self.data = {}
    
    def collection(self, name):
        if name not in self.data:
            self.data[name] = {}
        return MockCollection(self.data[name])

class MockCollection:
    def __init__(self, data):
        self.data = data
    
    def document(self, doc_id):
        if doc_id not in self.data:
            self.data[doc_id] = {}
        return MockDocument(self.data, doc_id)

class MockDocument:
    def __init__(self, data, doc_id):
        self.data = data
        self.doc_id = doc_id
    
    def set(self, data):
        self.data[self.doc_id] = data
    
    def update(self, data):
        for key, value in data.items():
            parts = key.split('.')
            if len(parts) == 1:
                self.data[self.doc_id][key] = value
            else:
                current = self.data[self.doc_id]
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
    
    def get(self):
        return MockDocSnapshot(self.data.get(self.doc_id, {}))

class MockDocSnapshot:
    def __init__(self, data):
        self.data = data
    
    def to_dict(self):
        return self.data

# Use mock implementations for demo
auth = MockAuth()
db = MockFirestore()

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash('Please log in to access this page', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        try:
            # Create user in Firebase Authentication
            user = auth.create_user_with_email_and_password(email, password)
            
            # Create user document in Firestore
            user_id = user['localId']
            user_data = {
                'email': email,
                'configuration': {
                    'gender': 'None',
                    'age': 'None'
                },
                'conversations': []
            }
            
            db.collection('users').document(user_id).set(user_data)
            
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login'))
        
        except Exception as e:
            flash(f'Error creating account: {str(e)}', 'error')
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        try:
            # Sign in user with Firebase Authentication
            user = auth.sign_in_with_email_and_password(email, password)
            
            # Store user info in session
            session['user'] = {
                'uid': user['localId'],
                'email': email,
                'token': user['idToken']
            }
            
            flash('Logged in successfully!', 'success')
            return redirect(url_for('dashboard'))
        
        except Exception as e:
            flash(f'Error logging in: {str(e)}', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('Logged out successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    user_id = session['user']['uid']
    
    # Get user data from Firestore
    user_doc = db.collection('users').document(user_id).get()
    user_data = user_doc.to_dict()
    
    return render_template('dashboard.html', user=user_data)

@app.route('/configure', methods=['GET', 'POST'])
@login_required
def configure():
    user_id = session['user']['uid']
    
    if request.method == 'POST':
        # Update user configuration in Firestore
        gender = request.form['gender']
        age = request.form['age']
        
        db.collection('users').document(user_id).update({
            'configuration.gender': gender,
            'configuration.age': age
        })
        
        flash('Configuration updated successfully!', 'success')
        return redirect(url_for('dashboard'))
    
    # Get current user configuration
    user_doc = db.collection('users').document(user_id).get()
    user_data = user_doc.to_dict()
    
    return render_template('configure.html', config=user_data['configuration'])

# API routes for AJAX calls
@app.route('/api/user', methods=['GET'])
@login_required
def get_user():
    user_id = session['user']['uid']
    user_doc = db.collection('users').document(user_id).get()
    user_data = user_doc.to_dict()
    return jsonify(user_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=12000, debug=True)