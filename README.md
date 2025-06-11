# Flask Firebase Web Application

A simple web application built with Flask and Firebase that includes user authentication and profile configuration.

## Features

- User authentication (signup, login, logout)
- User profile configuration (gender, age)
- Firebase Firestore database integration
- Responsive design using Bootstrap

## Prerequisites

- Python 3.7+
- Firebase account
- Node.js and npm (for Firebase CLI)

## Setup Instructions

### 1. Clone the repository

```bash
git clone <repository-url>
cd flask_firebase_app
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up Firebase

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Create a new project
3. Set up Firebase Authentication:
   - Enable Email/Password authentication
4. Set up Firestore Database:
   - Create a new Firestore database
   - Start in production mode or test mode

### 4. Configure Firebase in your application

1. In the Firebase console, go to Project Settings
2. Under "Your apps", click the web icon (</>) to add a web app
3. Register your app with a nickname
4. Copy the Firebase configuration object
5. Update the `firebase_config.py` file with your Firebase configuration

### 5. Set up Firebase Admin SDK

1. In the Firebase console, go to Project Settings > Service accounts
2. Click "Generate new private key"
3. Save the JSON file securely
4. Update the `admin_config` in `firebase_config.py` with the contents of the JSON file

### 6. Run the application locally

```bash
python app.py
```

The application will be available at http://localhost:12000

### 7. Deploy to Firebase

#### Install Firebase CLI

```bash
npm install -g firebase-tools
```

#### Login to Firebase

```bash
firebase login
```

#### Initialize Firebase Functions

```bash
mkdir -p functions
cd functions
npm init -y
npm install firebase-admin firebase-functions express
```

#### Create Express app for Firebase Functions

Create a file `functions/index.js`:

```javascript
const functions = require('firebase-functions');
const express = require('express');
const app = express();

// Your Express app configuration here
// This will proxy requests to your Flask app

exports.app = functions.https.onRequest(app);
```

#### Deploy to Firebase

```bash
firebase deploy
```

## Project Structure

```
flask_firebase_app/
├── app.py                  # Main Flask application
├── firebase_config.py      # Firebase configuration
├── requirements.txt        # Python dependencies
├── firebase.json           # Firebase deployment configuration
├── static/
│   └── css/
│       └── style.css       # Custom CSS styles
└── templates/
    ├── base.html           # Base template
    ├── index.html          # Home page
    ├── signup.html         # Signup page
    ├── login.html          # Login page
    ├── dashboard.html      # User dashboard
    └── configure.html      # Configuration page
```

## License

MIT