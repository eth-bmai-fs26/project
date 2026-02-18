import pandas as pd
import os

# Paths to datasets
TRANSCRIPTS_PATH = os.path.join(os.path.dirname(__file__), '../data/employee_tasks_meeting_id_transcripts.pkl')
TASKS_PATH = os.path.join(os.path.dirname(__file__), '../data/employee_tasks_meeting_id.pkl')

def load_transcripts():
    """
    Returns a dictionary: {meeting_id (int): transcript (str)}
    """
    if not os.path.exists(TRANSCRIPTS_PATH):
        print(f"Warning: Transcripts file not found at {TRANSCRIPTS_PATH}")
        return {}

    df = pd.read_pickle(TRANSCRIPTS_PATH)
    # The transcript file has multiple rows per meeting, but the transcription is the same
    transcripts_df = df[['meeting_id', 'transcription']].drop_duplicates()
    return dict(zip(transcripts_df['meeting_id'].astype(int), transcripts_df['transcription']))

def load_tasks():
    """
    Returns a dictionary: {meeting_id (int): [list of task dicts]}
    Task dict: {
        'description': str,
        'embedding': str, # kept as string for now
        'employee_id': int (0-4),
        'role': str
    }
    """
    if not os.path.exists(TASKS_PATH):
        print(f"Warning: Tasks file not found at {TASKS_PATH}")
        return {}

    df = pd.read_pickle(TASKS_PATH)
    tasks = {}
    
    for _, row in df.iterrows():
        m_id = int(row['meeting_id'])
        if m_id not in tasks:
            tasks[m_id] = []
        
        tasks[m_id].append({
            'description': row['task_description'],
            'embedding': row['task_embedding'],
            'employee_id': int(row['employee_id']),
            'role': row['role']
        })
    return tasks

# Global cache (simple in-memory)
_TRANSCRIPTS = None
_TASKS = None

def get_meeting_transcript(meeting_id):
    global _TRANSCRIPTS
    if _TRANSCRIPTS is None:
        _TRANSCRIPTS = load_transcripts()
    return _TRANSCRIPTS.get(meeting_id, "Meeting not found.")

def get_meeting_tasks(meeting_id):
    global _TASKS
    if _TASKS is None:
        _TASKS = load_tasks()
    return _TASKS.get(meeting_id, [])

def get_all_meeting_ids():
    global _TRANSCRIPTS
    if _TRANSCRIPTS is None:
        _TRANSCRIPTS = load_transcripts()
    return sorted(list(_TRANSCRIPTS.keys()))
