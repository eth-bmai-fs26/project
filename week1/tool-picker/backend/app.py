"""
app.py — Flask backend for the Tool Router demo.

Endpoints
---------
GET  /           → serves the HTML frontend
POST /predict    → accepts {"sentence": "..."}, returns routing prediction + probabilities
GET  /health     → quick liveness check
GET  /flights    → flight listing page  (?q=<original query>)
GET  /hotels     → hotel listing page   (?q=<original query>)
GET  /cars       → car rental listing   (?q=<original query>)
"""

import os
import re
import sys

import torch
from flask import Flask, jsonify, render_template, request
from sentence_transformers import SentenceTransformer

# Allow imports from lib/ and from train_model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train_model import LABEL_NAMES, load_model  # noqa: E402

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
)

# ---------------------------------------------------------------------------
# Load models at startup (train if checkpoint missing)
# ---------------------------------------------------------------------------
print("[app] Loading sentence encoder …")
_encoder = SentenceTransformer("all-MiniLM-L6-v2")

print("[app] Loading ToolRouter classifier …")
_model, _device = load_model()

# ---------------------------------------------------------------------------
# Label metadata shown in the UI
# ---------------------------------------------------------------------------
LABEL_META = {
    "FLIGHT_BOOKER": {
        "icon": "✈️",
        "color": "#3b82f6",
        "description": "Books flights and airline tickets",
    },
    "HOTEL_BOOKER": {
        "icon": "🏨",
        "color": "#10b981",
        "description": "Reserves hotels and accommodations",
    },
    "CAR_RENTAL_BOOKER": {
        "icon": "🚗",
        "color": "#f59e0b",
        "description": "Arranges car rentals and ground transport",
    },
}


# ---------------------------------------------------------------------------
# Mock listing data (hardcoded for demo purposes)
# ---------------------------------------------------------------------------
MOCK_DATA = {
    "flights": [
        {
            "airline": "Swiss Air",       "code": "LX 204",  "flag": "🇨🇭",
            "from_code": "ZRH", "from_city": "Zurich",
            "to_code":   "CDG", "to_city":   "Paris",
            "dep": "08:30", "arr": "10:45", "duration": "2h 15m", "stops": "Non-stop",
            "cabin": "Economy", "baggage": "23 kg", "currency": "CHF", "price": 289,
            "badge": "Best value",
            "tags": ["Meal included"],
        },
        {
            "airline": "Lufthansa",       "code": "LH 1289", "flag": "🇩🇪",
            "from_code": "ZRH", "from_city": "Zurich",
            "to_code":   "CDG", "to_city":   "Paris",
            "dep": "07:15", "arr": "11:50", "duration": "4h 35m", "stops": "1 stop (FRA)",
            "cabin": "Economy", "baggage": "23 kg", "currency": "CHF", "price": 187,
            "badge": "Cheapest",
            "tags": [],
        },
        {
            "airline": "EasyJet",         "code": "U2 834",  "flag": "🟠",
            "from_code": "GVA", "from_city": "Geneva",
            "to_code":   "ORY", "to_city":   "Paris Orly",
            "dep": "11:00", "arr": "12:55", "duration": "1h 55m", "stops": "Non-stop",
            "cabin": "Basic", "baggage": "Carry-on only", "currency": "CHF", "price": 94,
            "badge": None,
            "tags": [],
        },
        {
            "airline": "Air France",      "code": "AF 1726", "flag": "🇫🇷",
            "from_code": "ZRH", "from_city": "Zurich",
            "to_code":   "CDG", "to_city":   "Paris",
            "dep": "14:30", "arr": "16:45", "duration": "2h 15m", "stops": "Non-stop",
            "cabin": "Economy", "baggage": "23 kg", "currency": "CHF", "price": 251,
            "badge": None,
            "tags": ["Meal included"],
        },
        {
            "airline": "British Airways", "code": "BA 578",  "flag": "🇬🇧",
            "from_code": "ZRH", "from_city": "Zurich",
            "to_code":   "CDG", "to_city":   "Paris",
            "dep": "09:45", "arr": "14:30", "duration": "4h 45m", "stops": "1 stop (LHR)",
            "cabin": "Economy Plus", "baggage": "23 kg", "currency": "CHF", "price": 318,
            "badge": None,
            "tags": ["Extra legroom"],
        },
        {
            "airline": "Turkish Airlines","code": "TK 1847", "flag": "🇹🇷",
            "from_code": "ZRH", "from_city": "Zurich",
            "to_code":   "CDG", "to_city":   "Paris",
            "dep": "06:00", "arr": "14:20", "duration": "8h 20m", "stops": "1 stop (IST)",
            "cabin": "Economy", "baggage": "23 kg", "currency": "CHF", "price": 165,
            "badge": None,
            "tags": ["Meal included"],
        },
    ],
    "hotels": [
        {
            "name": "Hôtel de Crillon", "stars": 5,
            "location": "Place de la Concorde, 8th arr.",
            "rating": 4.9, "reviews": 2341,
            "per_night": 820, "currency": "CHF",
            "badge": "Luxury pick",
            "amenities": ["Pool", "Spa", "Fine dining", "Concierge", "Valet"],
            "desc": "Iconic 18th-century palace steps from the Champs-Élysées.",
        },
        {
            "name": "Hôtel Lutetia", "stars": 5,
            "location": "Saint-Germain-des-Prés, 6th arr.",
            "rating": 4.8, "reviews": 1872,
            "per_night": 490, "currency": "CHF",
            "badge": None,
            "amenities": ["Spa", "Jazz bar", "Restaurant", "Gym"],
            "desc": "Art Deco masterpiece in the heart of the Left Bank.",
        },
        {
            "name": "Le Marais Boutique Hotel", "stars": 4,
            "location": "Le Marais, 4th arr.",
            "rating": 4.6, "reviews": 938,
            "per_night": 210, "currency": "CHF",
            "badge": "Best rated 4★",
            "amenities": ["Breakfast", "Wi-Fi", "Bar", "Terrace"],
            "desc": "Charming boutique hotel in Paris's trendiest neighbourhood.",
        },
        {
            "name": "Aparthotel Adagio", "stars": 4,
            "location": "Montparnasse, 14th arr.",
            "rating": 4.4, "reviews": 654,
            "per_night": 175, "currency": "CHF",
            "badge": None,
            "amenities": ["Kitchen", "Laundry", "Wi-Fi", "Gym"],
            "desc": "Fully equipped apartments, perfect for longer stays.",
        },
        {
            "name": "Ibis Paris Bastille", "stars": 3,
            "location": "Bastille, 11th arr.",
            "rating": 4.1, "reviews": 3210,
            "per_night": 105, "currency": "CHF",
            "badge": "Great location",
            "amenities": ["Wi-Fi", "24h Reception", "Bar"],
            "desc": "Reliable value hotel with easy metro access.",
        },
        {
            "name": "Generator Paris", "stars": 2,
            "location": "Canal Saint-Martin, 10th arr.",
            "rating": 4.3, "reviews": 5621,
            "per_night": 45, "currency": "CHF",
            "badge": "Budget pick",
            "amenities": ["Wi-Fi", "Bar", "Social lounge"],
            "desc": "Design hostel with private rooms in a vibrant neighbourhood.",
        },
    ],
    "cars": [
        {
            "model": "Toyota Yaris",    "category": "Mini",       "company": "Hertz",        "emoji": "🚗",
            "transmission": "Automatic", "seats": 5, "fuel": "Petrol",
            "per_day": 35, "currency": "CHF", "badge": "Best value",
            "features": ["Free cancellation", "Airport pickup", "Unlimited mileage"],
        },
        {
            "model": "Volkswagen Golf", "category": "Compact",    "company": "Europcar",     "emoji": "🚙",
            "transmission": "Automatic", "seats": 5, "fuel": "Petrol",
            "per_day": 48, "currency": "CHF", "badge": None,
            "features": ["Free cancellation", "Airport pickup", "GPS included"],
        },
        {
            "model": "Peugeot 308",     "category": "Compact",    "company": "Sixt",         "emoji": "🚘",
            "transmission": "Manual",    "seats": 5, "fuel": "Diesel",
            "per_day": 44, "currency": "CHF", "badge": None,
            "features": ["Airport pickup", "Child seat available"],
        },
        {
            "model": "Renault Zoé",     "category": "Electric",   "company": "Eco Mobility", "emoji": "⚡",
            "transmission": "Automatic", "seats": 5, "fuel": "Electric",
            "per_day": 55, "currency": "CHF", "badge": "Eco choice",
            "features": ["Free cancellation", "Charging cable", "Zero emissions"],
        },
        {
            "model": "Mercedes C-Class","category": "Mid-size",   "company": "Avis",         "emoji": "🏎️",
            "transmission": "Automatic", "seats": 5, "fuel": "Petrol",
            "per_day": 89, "currency": "CHF", "badge": "Premium",
            "features": ["Free cancellation", "Airport pickup", "Premium audio"],
        },
        {
            "model": "Renault Kangoo",  "category": "Minivan",    "company": "Budget",       "emoji": "🚐",
            "transmission": "Manual",    "seats": 7, "fuel": "Diesel",
            "per_day": 72, "currency": "CHF", "badge": None,
            "features": ["Airport pickup", "Large boot", "Child seats available"],
        },
    ],
}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("tool_router_app.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/flights")
def flights():
    return render_template("flights.html", items=MOCK_DATA["flights"], query=request.args.get("q", ""))


@app.route("/hotels")
def hotels():
    return render_template("hotels.html", items=MOCK_DATA["hotels"], query=request.args.get("q", ""))


@app.route("/cars")
def cars():
    return render_template("cars.html", items=MOCK_DATA["cars"], query=request.args.get("q", ""))


@app.route("/plan", methods=["POST"])
def plan():
    data = request.get_json(silent=True) or {}
    sentence = data.get("sentence", "").strip()
    if not sentence:
        return jsonify({"error": "No sentence provided"}), 400

    # Split compound query on commas / "and" / semicolons
    parts = re.split(r",\s*(?:and\s+)?|\s+and\s+|;\s*", sentence, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 3]
    if not parts:
        parts = [sentence]

    PATHS = {
        "FLIGHT_BOOKER":     "/flights",
        "HOTEL_BOOKER":      "/hotels",
        "CAR_RENTAL_BOOKER": "/cars",
    }

    steps = []
    seen = set()
    for part in parts:
        emb = _encoder.encode([part])
        x = torch.tensor(emb, dtype=torch.float32).to(_device)
        with torch.no_grad():
            pred = int(_model(x).argmax(1).item())
        label = LABEL_NAMES[pred]
        if label not in seen:
            seen.add(label)
            meta = LABEL_META[label]
            steps.append({
                "label": label,
                "query": part,
                "icon":  meta["icon"],
                "color": meta["color"],
                "path":  PATHS[label],
            })

    return jsonify({"steps": steps})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    sentence = data.get("sentence", "").strip()

    if not sentence:
        return jsonify({"error": "No sentence provided"}), 400

    # Encode with the sentence transformer
    embedding = _encoder.encode([sentence])
    x = torch.tensor(embedding, dtype=torch.float32).to(_device)

    with torch.no_grad():
        logits = _model(x)
        probs = torch.softmax(logits, dim=1)[0]

    pred_idx = int(probs.argmax().item())
    pred_label = LABEL_NAMES[pred_idx]
    meta = LABEL_META[pred_label]

    probabilities = {
        LABEL_NAMES[i]: round(float(probs[i].item()) * 100, 1)
        for i in range(len(LABEL_NAMES))
    }

    return jsonify(
        {
            "prediction": pred_label,
            "icon": meta["icon"],
            "color": meta["color"],
            "description": meta["description"],
            "confidence": round(float(probs[pred_idx].item()) * 100, 1),
            "probabilities": probabilities,
        }
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5001)
