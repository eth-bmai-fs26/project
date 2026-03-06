from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, send_from_directory

BASE = Path(__file__).parent / "dataset_new"
ORIG_DIR = BASE / "original"
SEG_DIR = BASE / "segmented"
CSV_PATH = BASE / "ground_truth.csv"

app = Flask(__name__)

df = pd.read_csv(CSV_PATH)
df["coverage_pct"] = df["coverage_pct"].astype(float)
TILES = df.to_dict(orient="records")
for i, tile in enumerate(TILES):
    tile["index"] = i


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/tiles")
def get_tiles():
    return jsonify(TILES)


@app.route("/images/original/<filename>")
def serve_original(filename):
    return send_from_directory(str(ORIG_DIR), filename)


@app.route("/images/segmented/<filename>")
def serve_segmented(filename):
    stem = Path(filename).stem
    return send_from_directory(str(SEG_DIR), f"{stem}-segmented.jpg")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
