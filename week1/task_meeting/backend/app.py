import os
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from database import db, init_db, Employee
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app) # Enable CORS for frontend

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///employees.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize DB
init_db(app)

# Load Model
MODEL_PATH = 'backend/models/task_classifier.pkl'
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def extract_tasks_dummy(text):
    """
    Dummy LLM call. 
    If text matches the demo input, return the specific tasks.
    Otherwise, split by sentence.
    """
    default_text_snippet = "John, please update the SQL server today"
    
    if default_text_snippet in text:
        return [
            "Update the SQL server",
            "Focus on the budget for Q4",
            "Call the client",
            "Draft the press release",
            "Check the security logs"
        ]
    
    # Simple fallback: split by periods and filter empty strings
    tasks = [s.strip() for s in text.split('.') if s.strip()]
    return tasks

@app.route('/api/assign_tasks', methods=['POST'])
def assign_tasks():
    data = request.json
    minutes_text = data.get('minutes', '')

    if not minutes_text:
        return jsonify({"error": "No minutes provided"}), 400

    # 1. Extract Tasks (Dummy LLM)
    extracted_tasks_text = extract_tasks_dummy(minutes_text)

    # 2. Classify and Assign
    assignments = []
    
    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    for task_desc in extracted_tasks_text:
        # Predict specialization
        predicted_specialization = model.predict([task_desc])[0]
        
        # Find employee
        employee = Employee.query.filter_by(specialization=predicted_specialization).first()
        
        owner_name = employee.name if employee else "Unassigned"
        
        assignments.append({
            "task": task_desc,
            "owner": owner_name,
            "specialization": predicted_specialization # Optional debug info
        })

    return jsonify(assignments)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
