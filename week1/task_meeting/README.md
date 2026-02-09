# Meeting Task Assigner - BMAI Project 1

This project is a web-based application designed to automatically extract and assign tasks from meeting minutes. It demonstrates a full-stack integration of a simple HTML frontend with a Python Flask backend, utilizing a machine learning model for task classification and a SQLite database for employee management.

## Project Structure

```
task_meeting/
├── backend/
│   ├── app.py                 # Main Flask application entry point
│   ├── database.py            # Database setup and employee seeding script
│   ├── train_model.py         # Script to train and save the task classification model
│   ├── .env                   # Environment variables (Database URL, etc.)
│   └── models/
│       └── task_classifier.pkl # Saved Scikit-learn model (generated)
├── task_assigner_app.html     # Frontend interface
├── project_task_meeting.ipynb # Jupyter Notebook for exploration
└── README.md                  # Project documentation
```

## Architecture

1.  **Frontend (`task_assigner_app.html`)**:
    *   A clean Bootstrap-based UI where users can paste meeting transcripts.
    *   Uses JavaScript `fetch` API to communicate with the Flask backend.
    *   Dynamically renders the task assignment table based on the JSON response.

2.  **Backend (`backend/app.py`)**:
    *   **Framework**: Flask.
    *   **API**: Exposes a POST endpoint `/api/assign_tasks`.
    *   **Logic**:
        *   **Task Extraction**: Currently uses a dummy extractor (`extract_tasks_dummy`) that splits text by sentences or detects a specific demo transcript. In a production scenario, this would be replaced by an LLM call.
        *   **Classification**: Uses a trained Naive Bayes classifier (`task_classifier.pkl`) to predict the "Specialization" required for each extracted task.
        *   **Assignment**: Queries the SQLite database to find an employee whose specialization matches the predicted one.

3.  **Database (`employees.db`)**:
    *   **Type**: SQLite (via SQLAlchemy).
    *   **Schema**: `Employee` table with `id`, `name`, and `specialization`.
    *   **Seed Data**: Includes employees like John (DB Admin), Sarah (Finance), etc.

4.  **Machine Learning (`backend/train_model.py`)**:
    *   **Algorithm**: TF-IDF Vectorizer + Multinomial Naive Bayes.
    *   **Training Data**: A synthetic dataset mapping task descriptions to job titles.
    *   **Output**: A serialized pipeline saved as `backend/models/task_classifier.pkl`.

## Prerequisites

*   **OS**: Linux (recommended) / macOS / Windows
*   **Python**: Version 3.12 or higher recommended.
*   **Environment Manager**: `pyenv` (as configured in this project).

## Setup & Installation

### 1. Python Environment Setup

Ensure you are using the correct Python environment using `pyenv`.

```bash
# Activate the project environment
pyenv activate bmai-project
```

### 2. Install Dependencies

Install the required Python packages:

```bash
pip install flask flask-sqlalchemy flask-cors python-dotenv joblib scikit-learn
```

*   `flask`: Web framework.
*   `flask-sqlalchemy`: ORM for database interactions.
*   `flask-cors`: Handles Cross-Origin Resource Sharing for the frontend.
*   `python-dotenv`: Loads configuration from `.env` files.
*   `scikit-learn`: Machine learning library.
*   `joblib`: For saving/loading the trained model.

### 3. Initialize the Database

Run the database script to create `employees.db` and seed it with initial data.

```bash
python backend/database.py
```
*Output: "Database initialized and seeded."*

### 4. Train the Classification Model

Run the training script to generate the task classifier.

```bash
python backend/train_model.py
```
*Output: "Training model... Model saved to backend/models/task_classifier.pkl"*

## Running the Application

### 1. Start the Backend Server

Start the Flask development server. It will listen on `http://localhost:5000`.

```bash
python backend/app.py
```

*Keep this terminal window open.*

### 2. Run the Frontend

Open the `task_assigner_app.html` file in your web browser. You can simply double-click the file or run:

```bash
xdg-open task_assigner_app.html  # Linux
# OR open open task_assigner_app.html # macOS
```

### 3. Usage

1.  In the web interface, you will see a text area pre-filled with a sample meeting transcript.
2.  Click the **"Extract & Assign Tasks"** button.
3.  The frontend sends the text to the backend.
4.  The backend extracts tasks, classifies them, finds the right employee, and returns the results.
5.  The table updates to show the task and the assigned employee (marked with a checkmark).

## API Endpoint Reference

**POST** `/api/assign_tasks`

*   **Request Body**:
    ```json
    {
      "minutes": "John, please update the SQL server..."
    }
    ```

*   **Response**:
    ```json
    [
      {
        "task": "Update the SQL server",
        "owner": "John",
        "specialization": "Database Admin"
      },
      ...
    ]
    ```

## Troubleshooting

*   **Port 5000 in use**: If Flask fails to start, check if another process is using port 5000 (`lsof -i :5000`) and kill it.
*   **CORS Errors**: If the frontend doesn't work, ensure `flask-cors` is installed and `CORS(app)` is active in `app.py`.
*   **Model not found**: Ensure you ran `python backend/train_model.py` before starting the app.
