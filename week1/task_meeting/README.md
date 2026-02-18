# Meeting Task Assigner - BMAI Project 1

This project is a web-based application designed to automatically extract and assign tasks from meeting minutes. It demonstrates a full-stack integration of a simple HTML frontend with a Python Flask backend. The application leverages a large language model (LLM) that takes meeting transcripts as input and generates a list of tasks to be assigned. A high-performance machine learning model then uses task embeddings to predict the most suitable employee for each task.

It demonstrates a **Hybrid AI approach**:
1.  **LLM (GPT-4o-mini)**: Converts unstructured meeting text into structured task lists.
2.  **Classic ML (SVM + BERT Embeddings)**: Assigns tasks to employees based on historical data.

## 📂 Project Structure

```text
week1/task_meeting/
├── backend/
│   ├── app.py                      # Main Flask application (API + Auto-Train Logic)
│   ├── train_model.py               # Script to train the SVM classifier; run this only if `task_classifier.pkl` does not exist
│   └── models/
│       └── task_classifier.pkl      # Saved SVM model (auto-generated after training)
├── data/
│   ├── employee_tasks_hybrid.pkl    # Dataset with embeddings for training
│   └── final_transcripts_meetings.csv  # Source of meeting IDs and transcripts for task extraction
├── task_assigner_app.html           # Frontend interface (Bootstrap + JS)
├── requirements.txt                 # Python dependencies for the project
└── README.md                        # Project documentation
```

## Architecture

1.  **Frontend (`task_assigner_app.html`)**:
    *   A clean Bootstrap-based UI allowing users to select preset meeting IDs or paste custom transcripts.
    *   Uses JavaScript `fetch` API to communicate with the Flask backend.
    *   Dynamically renders the task assignment table with checkmarks for the five employee IDs.

2.  **Backend (`backend/app.py`)**:
    *   **Framework**: Flask.
    *   **API**: Exposes a POST endpoint `/api/assign_tasks`.
    *   **Logic**:
        *   **Task Extraction**: Uses an LLM Agent (GPT-4o-mini) to analyze unstructured text and extract concise, future-facing actionable tasks.
        *   **Classification**: Converts extracted tasks into high-dimensional vectors (embeddings) and uses a trained Linear SVM classifier to predict the most suitable employee_id.
        *   **Assignment**: Automatically assigns tasks to specific IDs (0: Backend, 1: Support, 2: Data Sci, 3: HR, 4: Marketing).

3.  **Data Layer**:
    *   **Training Dataset (`employee_tasks_hybrid.pkl`)**: A pickle file containing historical task descriptions and their pre-computed embeddings used to train the classifier.
    *   **Meeting Repository (final_transcripts_meetings.csv)**: A dataset used to populate the frontend dropdown with preset meeting IDs and transcripts for demonstration.

4.  **Machine Learning (`backend/train_model.py`)**:
    *   **Algorithm**: Linear Support Vector Machine (LinearSVC).
    *   **Training Process**: Checks for the existence of task_classifier.pkl; if missing, it stacks embeddings from the hybrid dataset into a 2D matrix and fits the SVM model to the employee labels.
    *   **Output**: A ML model saved as `backend/models/task_classifier.pkl` that returns the employee_id for any given task embedding.

## Prerequisites

*   **OS**: Linux (recommended) / macOS / Windows
*   **Python**: Version 3.12 or higher recommended.
    
    Windows: Download the installer from https://www.python.org/ (Check the box "Add Python to PATH" during installation).

    macOS: Use the installer from https://www.python.org/ or install via Homebrew: ```bash brew install python@3.12```
  
    Linux: Usually pre-installed. Update using sudo apt install python3.12.
*   **Environment Manager**: `pyenv` (as configured in this project) and pyenv-virtualenv.
*   **API Key**: An OpenAI-compatible API key (configured in a .env file).

## Setup & Installation

### 1. Python Environment Setup

We recommend using a dedicated virtual environment to avoid dependency conflicts.

```bash
# Create the virtual environment (if not already done)
pyenv virtualenv 3.12.0 bmai-project

# Activate the project environment
pyenv activate bmai-project
```

### 2. Install Dependencies
Install all required packages at once using the provided requirements file:

```bash
pip install -r requirements.txt
```

### 3. Key Libraries Included

*   `openai`: For the LLM-based task extraction (GPT-4o-mini).
*   `sentence-transformers`: To generate BERT-based embeddings for the SVM classifier.
*   `scikit-learn`: Powering the Linear SVM model.
*   `flask & flask-cors`: To serve the API and allow frontend communication.
*   `pandas & joblib`: For dataset handling and model serialization.

### 4. Train the Classification Model

The system is designed to handle model initialization automatically. When you start the backend via 
```bash
python backend/app.py
```
The script checks if the trained model already exists.

If `backend/models/task_classifier.pkl` is missing, app.py will automatically trigger the training process.

## Running the Application

### 1. Start the Backend Server

Start the Flask development server. It will listen on `http://localhost:5001`.

```bash
python backend/app.py
```

*Keep this terminal window open.*

### 2. Run the Frontend

Open the `task_assigner_app.html` file in your web browser. In your file explorer, double-click the `task_assigner_app.html` file. If your computer asks which program to use, select your preferred web browser. Alternatively you can run:

```bash
xdg-open task_assigner_app.html  # Linux
# OR open open task_assigner_app.html # macOS
```

### 3. Using the Application

Once the backend is running and you have opened the `task_assigner_app.html` in your browser, follow these steps:

1.  **Select or Input a Transcript**:
    *   Option A: Click the "Choose a Meeting ID" dropdown and select an ID from our company dataset. The corresponding transcript will instantly appear in the text area below.
    *   Option B: Clear the text area and paste your own custom meeting transcript directly into the box.
2.  **Trigger the AI**: Click the "🚀 Extract & Assign Tasks" button.
3.  **Processing**: You will see a "Model is predicting..." overlay. During this time, the frontend sends the text to the Flask backend, where the LLM extracts tasks and the SVM predicts the correct owners.
4.  **Review Results**: The table will dynamically update. Each row represents a task extracted by the AI, and a green checkmark (✓) will appear under the employee ID (0–4) assigned by the machine learning model.

## API Endpoint Reference

**POST** `/api/assign_tasks`

*   **Request Body**:
    ```json
    {
      "transcript": "We need to fix the backend API and update the HR handbook."
    }
    ```

*   **Response**:
    ```json
    [
      {
        "task": "Fix the backend API",
        "owner_id": 0
      },
      {
        "task": "Update the HR handbook",
        "owner_id": 3
      }
    ]
    ```

## Troubleshooting

*   **Port 5001 in use**: Our app runs on port 5001 to avoid conflicts with AirPlay or default Flask settings. If it fails to start, check the process using 
  ```bash
  lsof -i :5001 (Mac/Linux)
  ``` 
  or 
  ```bash
  netstat -ano | findstr :5001 (Windows)
  ```
  and kill it.
*   **CORS Errors**: If the frontend doesn't work, ensure `flask-cors` is installed and `CORS(app)` is active in `app.py`.
*   **Model not found**: Ensure `employee_tasks_hybrid.pkl` is located in the `data/` folder. If the auto-training fails, try running `python backend/train_model.py` manually to see the specific error message.
*   **ModuleNotFoundError**: No module named 'sentence_transformers': Run `pip install sentence-transformers`.