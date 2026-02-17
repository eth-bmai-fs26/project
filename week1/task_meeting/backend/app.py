import joblib
import numpy as np
import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from dotenv import load_dotenv  

# --- 1. SETUP & PATHS ---
app = Flask(__name__)
CORS(app)

# Get the directory where this script (app.py) is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'task_classifier.pkl')

# --- 2. AUTO-TRAIN CHECK ---
# Check if the model exists. If not, train it immediately.
if not os.path.exists(MODEL_PATH):
    print(f"⚠️ Model not found at {MODEL_PATH}")
    print("🚀 Starting automatic training sequence...")
    
    try:
        # Import the train function dynamically
        # This assumes train_model.py is in the same folder as app.py
        import train_model 
        train_model.train()
        print("✅ Automatic training complete.")
    except ImportError:
        print("❌ Error: Could not import 'train_model.py'. Make sure it is in the backend folder.")
    except Exception as e:
        print(f"❌ Error during training: {e}")

# --- 3. LOAD RESOURCES ---
print("⏳ Initializing AI Backend...")

# A. Load Embedder
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# B. Load Classifier (Now guaranteed to exist if training succeeded)
try:
    task_model = joblib.load(MODEL_PATH)
    print("✅ Classifier loaded successfully!")
except Exception as e:
    print(f"⚠️ CRITICAL: Could not load model even after training check. Error: {e}")
    task_model = None

# C. OpenAI Client
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("❌ API Key not found! Make sure you have a .env file.")

client = OpenAI(
    api_key=api_key, 
    base_url="https://litellm.sph-prod.ethz.ch/v1"
)

# --- 4. API ROUTES ---
@app.route('/api/assign_tasks', methods=['POST'])
def assign_tasks():
    data = request.json
    transcript = data.get('transcript', '')

    if not transcript:
        return jsonify([])

    # STEP A: Extract Tasks using LLM
    try:
        # Improved Prompt Engineering:
        # 1. Persona: Acts as a Technical Project Manager.
        # 2. Constraints: Forces "Action Verb + Object" format.
        # 3. Filtering: Explicitly ignores chit-chat and status updates.
        prompt = f"""
        You are an expert Technical Project Manager. 
        Analyze the following meeting transcript and extract a list of specific, actionable tasks.
        
        Rules:
        1. Each task must start with a strong action verb (e.g., "Fix", "Update", "Analyze", "Deploy").
        2. Keep tasks concise (under 10 words).
        3. Ignore pleasantries, general discussion, and past status updates ("I finished X").
        4. Focus only on future actions required by the team.
        
        Transcript: 
        "{transcript}"
        
        Output format:
        Return ONLY a raw JSON list of strings. Do not use Markdown code blocks.
        Example: ["Refactor the login API", "Update the SQL schema", "Schedule a meeting with HR"]
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[{"role": "user", "content": prompt}],
            temperature=0  # Lower temperature = more deterministic/focused
        )
        
        content = response.choices[0].message.content.strip()
        
        # Clean potential markdown formatting just in case
        if "```" in content: 
            content = content.replace("```json", "").replace("```", "")
        
        import json
        extracted_tasks = json.loads(content)
        
    except Exception as e:
        print(f"LLM Error: {e}")
        return jsonify([])
    except Exception as e:
        print(f"LLM Error: {e}")
        return jsonify([])

    # STEP B: Predict Owner (SVM)
    results = []
    if task_model and extracted_tasks:
        embeddings = embedder.encode(extracted_tasks)
        predictions = task_model.predict(embeddings)
        
        for task_text, owner_id in zip(extracted_tasks, predictions):
            results.append({
                "task": task_text,
                "owner_id": int(owner_id)
            })
    else:
        # Fallback if model is broken
        results = [{"task": t, "owner_id": -1} for t in extracted_tasks]

    return jsonify(results)

if __name__ == '__main__':
    app.run(port=5001, debug=True)