import pandas as pd
import numpy as np
import joblib
import os
from sklearn.svm import LinearSVC  # Using Linear SVM as requested
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Going up one level to find 'data', then down into 'backend/models'
DATA_PATH = os.path.join(BASE_DIR, '../data/employee_tasks_hybrid.pkl')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'task_classifier.pkl')

# Ensure models directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

def train():
    print(f"📂 Loading data from: {DATA_PATH}")
    
    # 1. Load Data
    try:
        df = pd.read_pickle(DATA_PATH)
    except FileNotFoundError:
        print(f"❌ Error: Could not find dataset at {DATA_PATH}")
        return

    # 2. Prepare Features (X) and Target (y)
    # Stack the list of embeddings into a proper 2D numpy matrix
    print("⚙️ Processing features...")
    X = np.vstack(df['task_embedding'].values)
    y = df['employee_id']

    # 3. Split Data (Optional, but good for validation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Train the SVM Model
    print("🧠 Training Support Vector Machine (SVM)...")
    # We use LinearSVC as it is faster and standard for high-dim text vectors
    svm_model = LinearSVC(random_state=42, dual='auto')
    svm_model.fit(X_train, y_train)

    # 5. Validate
    accuracy = svm_model.score(X_test, y_test)
    print(f"✅ Training Complete! Model Accuracy: {accuracy:.2%}")

    # 6. Save the Model
    joblib.dump(svm_model, MODEL_PATH)
    print(f"💾 Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    train()