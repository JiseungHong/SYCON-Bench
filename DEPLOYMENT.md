# Deploying Flask Application to Firebase

This guide will walk you through the process of deploying your Flask application to Firebase Hosting and Cloud Functions.

## Prerequisites

- Firebase account
- Node.js and npm installed
- Firebase CLI installed (`npm install -g firebase-tools`)

## Step 1: Set Up Firebase Project

1. Go to the [Firebase Console](https://console.firebase.google.com/)
2. Create a new project or select an existing one
3. Enable Firebase Authentication (Email/Password)
4. Create a Firestore database

## Step 2: Get Firebase Configuration

1. In the Firebase console, go to Project Settings
2. Under "Your apps", click the web icon (</>) to add a web app
3. Register your app with a nickname
4. Copy the Firebase configuration object
5. Update the `firebase_config.py` file with your Firebase configuration

## Step 3: Set Up Firebase Admin SDK

1. In the Firebase console, go to Project Settings > Service accounts
2. Click "Generate new private key"
3. Save the JSON file securely
4. Update the `admin_config` in `firebase_config.py` with the contents of the JSON file

## Step 4: Set Up Firebase Functions

1. Install Firebase CLI if you haven't already:
   ```bash
   npm install -g firebase-tools
   ```

2. Login to Firebase:
   ```bash
   firebase login
   ```

3. Initialize Firebase in your project directory:
   ```bash
   cd flask_firebase_app
   firebase init
   ```
   
   Select the following options:
   - Hosting
   - Functions
   - Use an existing project (select your Firebase project)
   - For Functions, select JavaScript
   - For Hosting, use "public" as your public directory
   - Configure as a single-page app: No
   - Set up automatic builds and deploys with GitHub: No

4. Create a `functions` directory if it doesn't exist:
   ```bash
   mkdir -p functions
   ```

5. Initialize the Functions package:
   ```bash
   cd functions
   npm init -y
   npm install firebase-admin firebase-functions express
   ```

6. Create an Express app in `functions/index.js`:
   ```javascript
   const functions = require('firebase-functions');
   const express = require('express');
   const { exec } = require('child_process');
   const app = express();

   // Install Python dependencies on function instance
   const installDependencies = () => {
     return new Promise((resolve, reject) => {
       exec('pip install -r ../requirements.txt', (error, stdout, stderr) => {
         if (error) {
           console.error(`Error installing dependencies: ${error}`);
           reject(error);
           return;
         }
         console.log(`stdout: ${stdout}`);
         console.error(`stderr: ${stderr}`);
         resolve();
       });
     });
   };

   // Serve the Flask app
   app.all('*', async (req, res) => {
     try {
       // Install dependencies first
       await installDependencies();
       
       // Import and run the Flask app
       const { spawn } = require('child_process');
       const process = spawn('python', ['../app.py']);
       
       process.stdout.on('data', (data) => {
         console.log(`stdout: ${data}`);
       });
       
       process.stderr.on('data', (data) => {
         console.error(`stderr: ${data}`);
       });
       
       process.on('close', (code) => {
         console.log(`child process exited with code ${code}`);
       });
       
       // Forward the request to the Flask app
       // Note: This is a simplified example. In a real-world scenario,
       // you would use a proper proxy or implement the Flask app's functionality in Express.
       res.send('Flask app is running!');
     } catch (error) {
       console.error('Error:', error);
       res.status(500).send('Internal Server Error');
     }
   });

   exports.app = functions.https.onRequest(app);
   ```

   Note: This is a simplified example. In a real-world scenario, you would either:
   1. Port your Flask app to Express.js
   2. Use a proper proxy to forward requests to your Flask app
   3. Use Cloud Run instead of Cloud Functions to deploy your Flask app directly

## Step 5: Alternative Approach - Deploy to Cloud Run

Cloud Run is a better option for deploying Flask applications. Here's how to do it:

1. Create a `Dockerfile` in your project root:
   ```dockerfile
   FROM python:3.9-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   COPY . .

   ENV PORT=8080

   CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
   ```

2. Create a `.dockerignore` file:
   ```
   .git
   .github
   .gitignore
   Dockerfile
   README.md
   *.pyc
   *.pyo
   *.pyd
   __pycache__
   .pytest_cache
   ```

3. Build and deploy to Cloud Run:
   ```bash
   # Build the container
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/flask-firebase-app

   # Deploy to Cloud Run
   gcloud run deploy flask-firebase-app \
     --image gcr.io/YOUR_PROJECT_ID/flask-firebase-app \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

## Step 6: Update Firebase Hosting Configuration

If you're using Cloud Functions, update your `firebase.json`:

```json
{
  "hosting": {
    "public": "public",
    "ignore": [
      "firebase.json",
      "**/.*",
      "**/node_modules/**"
    ],
    "rewrites": [
      {
        "source": "**",
        "function": "app"
      }
    ]
  },
  "functions": {
    "source": "functions"
  }
}
```

If you're using Cloud Run, update your `firebase.json`:

```json
{
  "hosting": {
    "public": "public",
    "ignore": [
      "firebase.json",
      "**/.*",
      "**/node_modules/**"
    ],
    "rewrites": [
      {
        "source": "**",
        "run": {
          "serviceId": "flask-firebase-app",
          "region": "us-central1"
        }
      }
    ]
  }
}
```

## Step 7: Deploy to Firebase Hosting

```bash
firebase deploy
```

## Step 8: Access Your Deployed Application

After deployment, you can access your application at:
`https://YOUR_PROJECT_ID.web.app`

## Troubleshooting

- **Firebase Functions Timeout**: Firebase Functions have a timeout limit. If your Flask app takes too long to start, consider using Cloud Run instead.
- **Dependencies**: Make sure all your Python dependencies are listed in `requirements.txt`.
- **Environment Variables**: Set environment variables in the Firebase Functions or Cloud Run configuration.
- **Billing**: Make sure your Firebase project is on the Blaze (pay-as-you-go) plan to use Cloud Functions or Cloud Run.

## Additional Resources

- [Firebase Hosting Documentation](https://firebase.google.com/docs/hosting)
- [Firebase Cloud Functions Documentation](https://firebase.google.com/docs/functions)
- [Google Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Flask Documentation](https://flask.palletsprojects.com/)