# Tool Router — Travel Booking AI

An interactive web demo for the **Week 1 Tool Picker** case study.
Type any travel request — single or multi-step — and a trained MLP classifier routes it to the correct booking agent(s): ✈️ flights, 🏨 hotels, or 🚗 car rentals.
Multi-step queries (e.g. *"Book flights to Athens, a nice hotel there, and a van to move around"*) are split into individual steps and presented as a guided progress flow across listing pages.

---

## Project structure

```
tool-picker/
├── backend/
│   ├── app.py              # Flask application (routes + /plan endpoint)
│   ├── train_model.py      # MLP definition + train / load helpers
│   ├── models/             # Saved checkpoint (auto-created on first run)
│   └── templates/
│       ├── tool_router_app.html  # Main frontend (input + routing)
│       ├── flights.html          # Flight listing page
│       ├── hotels.html           # Hotel listing page
│       └── cars.html             # Car rental listing page
├── data/
│   ├── train.csv           # Pre-computed 384-dim sentence embeddings
│   ├── val.csv
│   └── test.csv
├── lib/
│   ├── dataset.py          # PyTorch Dataset + DataLoader helpers
│   ├── metrics.py          # accuracy, confusion_matrix, plot helpers
│   ├── utils.py            # set_seed
│   └── visualization.py
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1 — Create a virtual environment (Python 3.12 recommended)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### 3 — Run the app

```bash
python backend/app.py
```

On first launch the app checks for `backend/models/tool_router.pt`.
If the file is not found it trains the MLP automatically (~30 s on CPU) and saves the checkpoint before starting the server.

Open **http://localhost:5001** in your browser.

### 4 — (Optional) Pre-train manually

```bash
# Train and save the checkpoint without starting the server
python backend/train_model.py

# Force a fresh retrain even if a checkpoint already exists
python backend/train_model.py --force
```

---

## How it works

| Step | What happens |
|------|-------------|
| **Embedding** | The user's sentence is encoded by [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) into a 384-dimensional vector |
| **Planning** | `/plan` splits compound queries on commas / "and" / semicolons, classifies each segment, and deduplicates by label |
| **Classification** | A two-layer MLP (`Linear → ReLU → Dropout → Linear`) maps each embedding to one of three classes |
| **Routing** | The user is redirected to the first listing page; for multi-step plans, a progress bar guides them through each step sequentially |

### Model architecture

```
Input (384) → Linear(384→128) → ReLU → Dropout(0.1) → Linear(128→3) → Softmax
```

Trained for 20 epochs with Adam (lr=1e-3) on ~4,600 labelled travel sentences,
reaching **~97–98 % test accuracy**.

### Classes

| Label | Tool |
|-------|------|
| `FLIGHT_BOOKER` | Books flights and airline tickets |
| `HOTEL_BOOKER` | Reserves hotels and accommodations |
| `CAR_RENTAL_BOOKER` | Arranges car rentals and ground transport |

---

## API

### `POST /plan` _(primary endpoint)_

Accepts a free-text travel request — single or multi-step — and returns an ordered list of steps.

**Request body**
```json
{ "sentence": "Book flights to Athens, a nice hotel there, and a van to move around." }
```

**Response**
```json
{
  "steps": [
    { "label": "FLIGHT_BOOKER",     "query": "Book flights to Athens",       "icon": "✈️", "color": "#3b82f6", "path": "/flights" },
    { "label": "HOTEL_BOOKER",      "query": "a nice hotel there",           "icon": "🏨", "color": "#10b981", "path": "/hotels"  },
    { "label": "CAR_RENTAL_BOOKER", "query": "a van to move around",         "icon": "🚗", "color": "#f59e0b", "path": "/cars"    }
  ]
}
```

Single-intent queries return a `steps` array with one element and redirect directly to the relevant listing page (no progress bar shown).

---

### `POST /predict` _(low-level)_

Classifies a single sentence and returns raw probabilities.

**Request body**
```json
{ "sentence": "I need a hotel near the airport for two nights." }
```

**Response**
```json
{
  "prediction":   "HOTEL_BOOKER",
  "icon":         "🏨",
  "color":        "#10b981",
  "description":  "Reserves hotels and accommodations",
  "confidence":   97.3,
  "probabilities": {
    "FLIGHT_BOOKER":     1.2,
    "HOTEL_BOOKER":      97.3,
    "CAR_RENTAL_BOOKER": 1.5
  }
}
```

---

### `GET /flights` · `GET /hotels` · `GET /cars`

Listing pages. Accept the following query parameters when coming from a multi-step plan:

| Parameter | Description |
|-----------|-------------|
| `q` | The original query segment for this step |
| `plan` | Comma-separated ordered list of labels, e.g. `FLIGHT_BOOKER,HOTEL_BOOKER,CAR_RENTAL_BOOKER` |
| `qs` | Pipe-separated query segments matching each label |
| `step` | Zero-based index of the current step |

### `GET /health`

Returns `{"status": "ok"}` when the server is running.
