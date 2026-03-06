# Warehouse Inspector

A web app for the CAS TA Week 2 project. A U-Net model segments intact regions of reclaimed floor tile batches. Batches with **> 40% intact coverage** are approved for reuse.

The app has two modes:
- **Challenge Mode** — look at a tile image, decide APPROVE or REJECT, then see the segmentation mask and find out if you matched the algorithm.
- **Explorer Mode** — browse the full dataset with filters for verdict and coverage percentage.

---

## Project Structure

```
project1/
├── app.py                  # Flask server (API + static files)
├── requirements.txt
├── templates/
│   └── index.html          # Single HTML page
├── static/
│   ├── style.css
│   └── app.js              # Vanilla JS — no build step
├── dataset_new/
│   ├── ground_truth.csv
│   ├── original/           # Source tile images
│   └── segmented/          # U-Net output masks
└── game_app.py             # Original Streamlit version (kept for reference)
```

**No npm, no React, no build step.** The only dependency is Python.

---

## Requirements

- **Python** 3.9 or newer

---

## Setup & Running

### macOS / Linux

```bash
# 1. Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
python3 app.py
```

Open `http://localhost:5000` in your browser.

---

### Windows (Command Prompt)

```bat
:: 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

:: 2. Install dependencies
pip install -r requirements.txt

:: 3. Start the server
python app.py
```

Open `http://localhost:5000` in your browser.

---

### Windows (PowerShell)

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
python app.py
```

> **Note:** If you get a script execution error in PowerShell, run:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`
> then try activating again.

Open `http://localhost:5000` in your browser.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serves the app |
| `GET` | `/api/tiles` | Returns all tile records as JSON |
| `GET` | `/images/original/<filename>` | Serves the original tile image |
| `GET` | `/images/segmented/<filename>` | Serves the U-Net segmentation mask |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'flask'`**
Make sure your virtual environment is activated before running `python app.py`.

**`python` is not recognised (Windows)**
Try `py` instead of `python`. If neither works, install Python from [python.org](https://python.org) and make sure to check "Add Python to PATH" during installation.

**Port 5000 already in use**
Change the port at the bottom of `app.py`:
```python
app.run(debug=True, port=5001)   # or any free port
```

**macOS: port 5000 is taken by AirPlay Receiver**
Either disable AirPlay Receiver in System Settings > General > AirDrop & Handoff, or change the port as shown above.
