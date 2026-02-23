import os
import json
import base64
import requests
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory, Response, stream_with_context
from dotenv import load_dotenv
import openai
import pandas as pd

load_dotenv() #read .env and load API key

app = Flask(__name__) #create web server

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY      = os.environ.get("OPENAI_API_KEY", "your_key_here")
BASE_URL     = "https://litellm.sph-prod.ethz.ch/v1"
DATA_PATH    = "Data/fashion_data_2018_2022.xls"
OUTPUT_DIR   = "output"
HISTORY_DIR  = "history"

os.environ["OPENAI_API_KEY"] = API_KEY

# Load dataset once at startup
try:
    df = pd.read_excel(DATA_PATH) #load database as FASHION_DATA
    FASHION_DATA = df.to_string()
except Exception as e:
    print(f"Warning: Could not load fashion data: {e}")
    FASHION_DATA = "No fashion data available."

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_client():
    return openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)


def save_history(kind: str, entry: dict):
    """
    kind: 'articles' | 'images' | 'text'
    Appends entry to history/{kind}.json
    """
    os.makedirs(HISTORY_DIR, exist_ok=True)
    path = os.path.join(HISTORY_DIR, f"{kind}.json")
    history = []
    if os.path.exists(path):
        with open(path) as f:
            history = json.load(f)
    entry["timestamp"] = datetime.now().isoformat()
    history.append(entry)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def load_history(kind: str) -> list:
    path = os.path.join(HISTORY_DIR, f"{kind}.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def get_fashion_inspiration() -> str:
    return FASHION_DATA


def generate_image(title: str, style: str) -> str:
    """Returns saved filename."""
    client = get_client()
    prompt = f"Generate {style} for a fashion article about: {title}"
    resp = client.images.generate(
        model="azure/dall-e-3",
        prompt=prompt,
        size="1024x1024",
        n=1,
    )
    image_data = resp.data[0]
    raw = image_data.url if image_data.url else image_data.b64_json

    # Save to disk
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"image_{ts}_{style[:20].replace(' ','_')}.png"
    path = os.path.join(OUTPUT_DIR, filename)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if raw.startswith("http"):
        r = requests.get(raw, timeout=30)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    else:
        with open(path, "wb") as f:
            f.write(base64.b64decode(raw))

    # Log to image history
    save_history("images", {"title": title, "style": style, "filename": filename})
    return filename


def generate_article(query: str) -> str:
    client = get_client()
    resp = client.chat.completions.create(
        model="anthropic/claude-sonnet-4-5",
        messages=[{"role": "user", "content": f"Write a fashion article about: {query}"}],
    )
    article = resp.choices[0].message.content
    save_history("text", {"query": query, "article_preview": article[:300]})
    return article


def build_html_prompt(title, article_query, trend_data, article_text,
                      image_filenames, chat_history_text) -> str:
    images_list = "\n".join(f"  - /output/{f}" for f in image_filenames)
    return f"""You are a fashion expert, editor, and web designer.
Produce a COMPLETE, beautiful, standalone HTML page combining the article and ALL images.

ARTICLE TITLE: {title}

AVAILABLE IMAGES (use these exact paths in <img> src):
{images_list}

RULES:
• Return a COMPLETE HTML document (<!DOCTYPE html> … </html>)
• Embedded CSS in a <style> tag — modern, elegant, magazine-like
• Use "{title}" as <h1>
• Use ALL image paths above in <img src="…">
• Responsive layout, no external CDN/fonts
• Return ONLY the raw HTML, no markdown fences

DRAFT ARTICLE:
{article_text}

ARTICLE TOPIC: {article_query}

TREND DATA:
{trend_data}

PRIOR SESSIONS (for continuity):
{chat_history_text}
"""


def validate_html(html: str, image_filenames: list) -> tuple[bool, list]:
    errors = []
    # Strip markdown fences if present
    s = html.strip()
    if s.startswith("```"):
        lines = s.split("\n")[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        html = "\n".join(lines)

    if "<!DOCTYPE html>" not in html and "<html" not in html:
        errors.append("missing <!DOCTYPE html>")
    if "</html>" not in html:
        errors.append("missing </html>")
    if "<body" not in html:
        errors.append("missing <body>")
    if "<style" not in html:
        errors.append("missing <style>")
    for f in image_filenames:
        if f not in html:
            errors.append(f"image '{f}' not referenced")

    return (len(errors) == 0, errors, html)


def emergency_fallback_html(title: str, article_text: str, image_filenames: list) -> str:
    """Assembles a basic but valid HTML page from raw parts."""
    imgs = "\n".join(
        f'<img src="/output/{f}" alt="Fashion image" style="max-width:100%;margin:1rem 0;">'
        for f in image_filenames
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  body {{ font-family: Georgia, serif; max-width: 800px; margin: 40px auto; padding: 0 20px; color: #333; }}
  h1 {{ font-size: 2rem; border-bottom: 2px solid #ccc; padding-bottom: 0.5rem; }}
  p {{ line-height: 1.8; }}
</style>
</head>
<body>
<h1>{title}</h1>
{imgs}
<div>{article_text.replace(chr(10), '<br>')}</div>
</body>
</html>"""


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(title: str, image_styles: list, article_query: str,
                 progress_callback=None):
    """
    Full generation pipeline.
    progress_callback(step, total, message) — optional, for SSE streaming.
    """
    def emit(step, total, msg):
        if progress_callback:
            progress_callback(step, total, msg)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total = 2 + len(image_styles) + 2  # trend + history + images + article + html

    step = 1
    emit(step, total, "Fetching fashion trend data…")
    trend_data = get_fashion_inspiration()

    step += 1
    emit(step, total, "Loading prior session history…")
    prior_articles = load_history("articles")
    prior_text     = load_history("text")
    chat_history_text = "\n".join(
        f"[{e['timestamp']}] {e.get('title','?')}: {e.get('article_preview','')}"
        for e in (prior_articles + prior_text)[-5:]  # last 5 entries
    ) or "No prior sessions."

    image_filenames = []
    for i, style in enumerate(image_styles, 1):
        step += 1
        emit(step, total, f"Generating image {i}/{len(image_styles)}: {style}…")
        try:
            fname = generate_image(title, style)
            image_filenames.append(fname)
        except Exception as e:
            emit(step, total, f"⚠️ Image {i} failed: {e}")

    step += 1
    emit(step, total, "Writing article…")
    article_text = generate_article(article_query)

    step += 1
    emit(step, total, "Assembling final HTML page…")
    prompt = build_html_prompt(title, article_query, trend_data,
                               article_text, image_filenames, chat_history_text)

    client  = get_client()
    MAX_RETRIES = 3
    final_html  = ""
    last_errors = []

    for attempt in range(1, MAX_RETRIES + 1):
        emit(step, total, f"HTML generation attempt {attempt}/{MAX_RETRIES}…")
        raw = client.chat.completions.create(
            model="anthropic/claude-sonnet-4-5",
            messages=[{"role": "user", "content": prompt}],
        ).choices[0].message.content

        valid, errors, cleaned = validate_html(raw, image_filenames)
        if valid:
            final_html = cleaned
            emit(step, total, "✅ HTML validated successfully.")
            break
        last_errors = errors
        emit(step, total, f"⚠️ Validation failed ({', '.join(errors)}), retrying…")
    else:
        # Emergency fallback
        emit(step, total, "🚨 Max retries reached — using emergency fallback HTML.")
        final_html = emergency_fallback_html(title, article_text, image_filenames)

    # Save article output
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_filename = f"article_{ts}.html"
    html_path = os.path.join(OUTPUT_DIR, html_filename)
    with open(html_path, "w") as f:
        f.write(final_html)

    # Log to article history
    save_history("articles", {
        "title": title,
        "article_query": article_query,
        "image_styles": image_styles,
        "html_filename": html_filename,
        "image_filenames": image_filenames,
        "article_preview": article_text[:300],
    })

    emit(step, total, f"✅ Done! Article saved as {html_filename}")
    return {
        "html": final_html,
        "html_filename": html_filename,
        "image_filenames": image_filenames,
        "article_text": article_text,
    }


# ── Flask Routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("fashion_magazine.html")


@app.route("/generate", methods=["POST"])
def generate():
    data          = request.json
    title         = data.get("title", "").strip()
    image_styles  = [s.strip() for s in data.get("image_styles", []) if s.strip()]
    article_query = data.get("article_query", "").strip()

    if not title or not article_query:
        return jsonify({"error": "Title and article query are required."}), 400

    def stream():
        result = {}

        def progress_callback(step, total, message):
            # Send a progress event
            pct = int((step / total) * 100)
            payload = json.dumps({"type": "progress", "step": step, "total": total,
                                  "pct": pct, "message": message})
            yield f"data: {payload}\n\n"

        try:
            # We need to collect yields from the callback — use a queue trick
            import queue, threading

            q = queue.Queue()

            def cb(step, total, message):
                pct = int((step / total) * 100)
                q.put(json.dumps({"type": "progress", "step": step, "total": total,
                                  "pct": pct, "message": message}))

            def run():
                try:
                    r = run_pipeline(title, image_styles, article_query,
                                     progress_callback=cb)
                    q.put(json.dumps({"type": "done",
                                      "html_filename": r["html_filename"],
                                      "images": r["image_filenames"]}))
                except Exception as ex:
                    q.put(json.dumps({"type": "error", "message": str(ex)}))
                finally:
                    q.put(None)  # sentinel

            thread = threading.Thread(target=run, daemon=True)
            thread.start()

            while True:
                item = q.get()
                if item is None:
                    break
                yield f"data: {item}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return Response(stream_with_context(stream()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})



@app.route("/article/<filename>")
def view_article(filename):
    html_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(html_path):
        return render_template("error.html",
                               error_message=f"Article file '{filename}' not found."), 404
    with open(html_path, encoding="utf-8") as f:
        article_html = f.read()
    # Extract title from filename (article_20250222_143012.html → article_20250222_143012)
    title = filename.replace(".html", "").replace("_", " ").title()
    return render_template("article.html", title=title, article_html=article_html)


@app.route("/delete", methods=["POST"])
def delete_history():
    import shutil
    deleted = []
    for kind in ["articles", "images", "text"]:
        path = os.path.join(HISTORY_DIR, f"{kind}.json")
        if os.path.exists(path):
            os.remove(path)
            deleted.append(f"{kind}.json")
    return jsonify({"success": True, "deleted": deleted})


@app.route("/history/<kind>")
def get_history(kind):
    if kind not in ("articles", "images", "text"):
        return jsonify({"error": "Invalid history type"}), 400
    return jsonify(load_history(kind))


@app.route("/output/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000)