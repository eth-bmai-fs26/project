import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from database import db, init_db, Employee
from dotenv import load_dotenv
import dataset  # Assuming dataset.py is in the same directory

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app) # Enable CORS for frontend

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///employees.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize DB
init_db(app)

@app.route('/api/meetings', methods=['GET'])
def get_meetings():
    """Returns a list of available meeting IDs."""
    # Assuming meetings 0-99 are available based on dataset
    # We can just return a range or check the dataset
    # For now, let's return a list of IDs present in the dataset
    # dataset.py should have a function for this ideally, or we can iterate
    # Since dataset.py load_transcripts returns a dict, we can get keys
    try:
        meeting_ids = dataset.get_all_meeting_ids() # We need to implement this in dataset.py or just use keys
        # If dataset.py doesn't have get_all_meeting_ids, we can implement it or just do:
        # transcripts = dataset.load_transcripts()
        # meeting_ids = sorted(list(transcripts.keys()))
        return jsonify(meeting_ids)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/meetings/<int:meeting_id>', methods=['GET'])
def get_meeting(meeting_id):
    """Returns the transcript for a given meeting ID."""
    transcript = dataset.get_meeting_transcript(meeting_id)
    if not transcript:
        return jsonify({"error": "Meeting not found"}), 404
    return jsonify({"meeting_id": meeting_id, "transcript": transcript})

@app.route('/api/assign_tasks', methods=['POST'])
def assign_tasks():
    """
    Returns tasks for a given meeting ID from the dataset.
    Expects JSON: {"meeting_id": 123}
    """
    data = request.json
    meeting_id = data.get('meeting_id')

    if meeting_id is None:
        return jsonify({"error": "No meeting_id provided"}), 400

    try:
        meeting_id = int(meeting_id)
    except ValueError:
        return jsonify({"error": "Invalid meeting_id"}), 400

    # Get tasks from dataset
    tasks_data = dataset.get_meeting_tasks(meeting_id)
    
    # Format for frontend
    assignments = []
    for task in tasks_data:
        # Map employee_id to name if needed, or just return ID
        # User said "employees are identified by their id (from 0 to 4)"
        # But frontend might want names for display or we can send both
        # We can query DB for name/role if we want to be fancy
        
        emp_id = task['employee_id']
        emp = Employee.query.get(emp_id) # Should match seeded IDs 0-4
        owner_name = emp.name if emp else f"Employee {emp_id}"
        role = emp.role if emp else "Unknown"

        assignments.append({
            "task": task['description'],
            "owner_id": emp_id,
            "owner_name": owner_name,
            "role": role,
            "embedding": task['embedding']
        })

    return jsonify(assignments)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)