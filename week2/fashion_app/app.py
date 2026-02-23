import os
import json
import base64
import requests
import queue
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory, Response, stream_with_context
from dotenv import load_dotenv
import openai
import pandas as pd

load_dotenv()

app = Flask(__name__)
from flask_cors import CORS
CORS(app)

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY  = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Please create a .env file with your key.")

BASE_URL     = "https://litellm.sph-prod.ethz.ch/v1"
DATA_PATH    = "Data/fashion_data_2018_2022.xls"
OUTPUT_DIR   = "output"
HISTORY_DIR  = "history"

# Load dataset once at startup
try:
    df = pd.read_excel(DATA_PATH)
    FASHION_DATA = df.to_string()
except Exception as e:
    print(f"Warning: Could not load fashion data: {e}")
    FASHION_DATA = "No fashion data available."

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_client():
    return openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)


def save_history(kind: str, entry: dict):
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
    """Generates one image via DALL-E 3, saves to disk, returns filename."""
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


def parse_image_feedback(feedback: str, image_filenames: list) -> list[int]:
    """
    Uses the LLM to parse which images (by 1-based index) the user wants regenerated.
    Returns a list of 1-based indices, e.g. [2] or [1, 3] or [] for text-only feedback.
    """
    if not image_filenames:
        return []

    client = get_client()
    prompt = f"""You are a feedback parser for a fashion article generator.
The article has {len(image_filenames)} image(s), numbered 1 to {len(image_filenames)}.
The image filenames are: {image_filenames}

User feedback: "{feedback}"

Your job: identify which image numbers the user wants regenerated.
Be generous in your interpretation — if the user describes wanting a different image subject,
style, or content (even implicitly), treat that as wanting to regenerate that image.

Examples of implicit image change requests:
- "Instead of laptops I want a student holding one" → [1] (they want image 1 changed)
- "The photo should show Paris streets instead" → [1]
- "Make the second photo more dramatic" → [2]
- "Change the image to show evening wear" → [1] (if only 1 image)
- "Regenerate image 2 with bolder colors" → [2]

Only return [] if the user EXPLICITLY says to keep images as they are,
or if the feedback is purely about text with zero mention of visuals/photos/images.

Respond ONLY with a JSON array of integers, e.g. [1] or [1, 3] or [].
No explanation, no markdown, just the array."""

    resp = client.chat.completions.create(
        model="anthropic/claude-sonnet-4-5",
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.choices[0].message.content.strip()
    try:
        indices = json.loads(raw)
        # Validate — only return valid 1-based indices
        result = [i for i in indices if isinstance(i, int) and 1 <= i <= len(image_filenames)]
        print(f"[parse_image_feedback] feedback='{feedback}' → indices={result}")
        return result
    except Exception:
        print(f"[parse_image_feedback] Failed to parse: {raw}")
        return []  # if parsing fails, assume text-only feedback



def extract_image_style_from_feedback(feedback: str, image_index: int,
                                       original_style: str) -> str:
    """
    Extracts the new image style description from the user feedback for a specific image.
    Falls back to original style if nothing specific is found.
    """
    client = get_client()
    prompt = f"""You are extracting an image generation prompt from user feedback.

The user wants to change image {image_index}.
Original image style: "{original_style}"
User feedback: "{feedback}"

Extract a detailed image generation prompt for the NEW image the user wants.
- Use the user's description as the main subject
- Keep it suitable for DALL-E 3
- Make it fashion-focused and visually compelling
- If the user says "remove people", describe the scene without people
- If the user says "add a student", describe a fashionable student

Respond ONLY with the image generation prompt, no explanation, no quotes."""

    resp = client.chat.completions.create(
        model="anthropic/claude-sonnet-4-5",
        messages=[{"role": "user", "content": prompt}],
    )
    new_style = resp.choices[0].message.content.strip()
    print(f"[extract_image_style] image {image_index}: '{original_style}' → '{new_style}'")
    return new_style

def build_fresh_prompt(title, article_query, trend_data, article_text,
                       image_filenames, chat_history_text) -> str:
    """Prompt for generating a brand new article from scratch."""
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


def build_feedback_prompt(existing_html: str, feedback: str,
                          updated_image_map: dict) -> str:
    """
    Prompt for applying targeted feedback to an existing article.
    updated_image_map: {old_filename: new_filename} for regenerated images.
    """
    image_replacements = ""
    if updated_image_map:
        image_replacements = "IMAGE REPLACEMENTS (update these src paths in the HTML):\n"
        for old, new in updated_image_map.items():
            image_replacements += f"  - Replace /output/{old} with /output/{new}\n"

    return f"""You are a fashion editor making precise targeted edits to an existing HTML article.

USER FEEDBACK:
{feedback}

{image_replacements}

RULES:
• Apply ONLY the changes requested in the feedback
• Keep ALL other text, styling, layout, and images exactly as they are
• If image replacements are listed above, update ONLY those src paths
• Return the COMPLETE updated HTML document
• Return ONLY the raw HTML, no markdown fences, no explanation

EXISTING HTML TO EDIT:
{existing_html}
"""


def validate_html(html: str, image_filenames: list) -> tuple[bool, list, str]:
    errors = []
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



def fix_image_paths(html: str, image_filenames: list) -> str:
    """
    Post-process the LLM HTML to ensure all image filenames
    are correctly referenced as /output/<filename>.
    The LLM sometimes uses just the filename or a wrong path.
    """
    for filename in image_filenames:
        # Replace any of these wrong variants with the correct path
        wrong_variants = [
            f'src="{filename}"',
            f"src='{filename}'",
            f'src="output/{filename}"',
            f"src='output/{filename}'",
            f'src="./output/{filename}"',
            f"src='./output/{filename}'",
            f'src="../output/{filename}"',
            f"src='../output/{filename}'",
        ]
        correct = f'src="/output/{filename}"'
        for wrong in wrong_variants:
            html = html.replace(wrong, correct)
    return html

# ── Fresh Pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(title: str, image_styles: list, article_query: str,
                 progress_callback=None):
    """Full generation pipeline — generates everything from scratch."""
    def emit(step, total, msg):
        if progress_callback:
            progress_callback(step, total, msg)
        print(f"[{step}/{total}] {msg}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total = 2 + len(image_styles) + 2

    step = 1
    emit(step, total, "Fetching fashion trend data…")
    trend_data = get_fashion_inspiration()

    step += 1
    emit(step, total, "Loading prior session history…")
    prior_articles = load_history("articles")
    prior_text     = load_history("text")
    chat_history_text = "\n".join(
        f"[{e['timestamp']}] {e.get('title','?')}: {e.get('article_preview','')}"
        for e in (prior_articles + prior_text)[-5:]
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
    prompt = build_fresh_prompt(title, article_query, trend_data,
                                article_text, image_filenames, chat_history_text)

    client = get_client()
    MAX_RETRIES = 3
    final_html = ""

    for attempt in range(1, MAX_RETRIES + 1):
        emit(step, total, f"HTML generation attempt {attempt}/{MAX_RETRIES}…")
        raw = client.chat.completions.create(
            model="anthropic/claude-sonnet-4-5",
            messages=[{"role": "user", "content": prompt}],
        ).choices[0].message.content

        fixed = fix_image_paths(raw, image_filenames)
        valid, errors, cleaned = validate_html(fixed, image_filenames)
        if valid:
            final_html = cleaned
            emit(step, total, "✅ HTML validated successfully.")
            break
        emit(step, total, f"⚠️ Validation failed ({', '.join(errors)}), retrying…")
    else:
        emit(step, total, "🚨 Max retries reached — using emergency fallback HTML.")
        final_html = emergency_fallback_html(title, article_text, image_filenames)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_filename = f"article_{ts}.html"
    with open(os.path.join(OUTPUT_DIR, html_filename), "w") as f:
        f.write(final_html)

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


# ── Feedback Pipeline ──────────────────────────────────────────────────────────

def run_feedback_pipeline(feedback: str, current_filename: str,
                          current_image_filenames: list, title: str,
                          image_styles: list, progress_callback=None):
    """
    Feedback pipeline — applies targeted edits to existing article.
    Only regenerates images explicitly mentioned in feedback.
    """
    def emit(step, total, msg):
        if progress_callback:
            progress_callback(step, total, msg)
        print(f"[{step}/{total}] {msg}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1 — Parse which images need regenerating
    total = 4  # parse + load + (optional images) + apply feedback
    step = 1
    emit(step, total, "Analysing feedback…")
    indices_to_regenerate = parse_image_feedback(feedback, current_image_filenames)

    # Step 2 — Load existing HTML
    step += 1
    emit(step, total, "Loading existing article…")
    html_path = os.path.join(OUTPUT_DIR, current_filename)
    with open(html_path, encoding="utf-8") as f:
        existing_html = f.read()

    # Step 3 — Regenerate only the requested images
    updated_image_map = {}  # {old_filename: new_filename}
    final_image_filenames = list(current_image_filenames)

    if indices_to_regenerate:
        total += len(indices_to_regenerate)
        for idx in indices_to_regenerate:
            step += 1
            original_style = image_styles[idx - 1] if idx <= len(image_styles) else f"image {idx}"
            emit(step, total, f"Extracting new style for image {idx}…")
            new_style = extract_image_style_from_feedback(feedback, idx, original_style)
            emit(step, total, f"Regenerating image {idx}: {new_style[:40]}…")
            try:
                old_filename = current_image_filenames[idx - 1]
                new_filename = generate_image(title, new_style)
                updated_image_map[old_filename] = new_filename
                final_image_filenames[idx - 1] = new_filename
            except Exception as e:
                emit(step, total, f"⚠️ Image {idx} regeneration failed: {e}")
    else:
        emit(step, total, "No images to regenerate — text-only edit…")

    # Step 4 — Apply feedback to HTML via LLM
    step += 1
    emit(step, total, "Applying feedback to article…")
    prompt = build_feedback_prompt(existing_html, feedback, updated_image_map)

    client = get_client()
    MAX_RETRIES = 3
    final_html = ""

    for attempt in range(1, MAX_RETRIES + 1):
        emit(step, total, f"Feedback application attempt {attempt}/{MAX_RETRIES}…")
        raw = client.chat.completions.create(
            model="anthropic/claude-sonnet-4-5",
            messages=[{"role": "user", "content": prompt}],
        ).choices[0].message.content

        valid, errors, cleaned = validate_html(raw, final_image_filenames)
        if valid:
            final_html = cleaned
            emit(step, total, "✅ HTML validated successfully.")
            break
        emit(step, total, f"⚠️ Validation failed ({', '.join(errors)}), retrying…")
    else:
        emit(step, total, "🚨 Max retries reached — keeping original article.")
        final_html = existing_html  # fallback: return original unchanged

    # Save updated article
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_filename = f"article_{ts}_revised.html"
    with open(os.path.join(OUTPUT_DIR, html_filename), "w") as f:
        f.write(final_html)

    save_history("articles", {
        "title": title,
        "article_query": f"[FEEDBACK] {feedback}",
        "image_styles": image_styles,
        "html_filename": html_filename,
        "image_filenames": final_image_filenames,
        "article_preview": feedback[:300],
        "revised_from": current_filename,
    })

    emit(step, total, f"✅ Feedback applied! Saved as {html_filename}")
    return {
        "html": final_html,
        "html_filename": html_filename,
        "image_filenames": final_image_filenames,
    }


# ── SSE Stream Helper ──────────────────────────────────────────────────────────

def stream_pipeline(pipeline_fn):
    """Generic SSE streamer — runs any pipeline function in a thread and streams progress."""
    q = queue.Queue()

    def cb(step, total, message):
        pct = int((step / total) * 100)
        q.put(json.dumps({"type": "progress", "step": step, "total": total,
                          "pct": pct, "message": message}))

    def run():
        try:
            result = pipeline_fn(cb)
            q.put(json.dumps({"type": "done",
                              "html_filename": result["html_filename"],
                              "images": result["image_filenames"]}))
        except Exception as ex:
            q.put(json.dumps({"type": "error", "message": str(ex)}))
        finally:
            q.put(None)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    def generate():
        while True:
            item = q.get()
            if item is None:
                break
            yield f"data: {item}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


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

    return stream_pipeline(
        lambda cb: run_pipeline(title, image_styles, article_query, progress_callback=cb)
    )


@app.route("/feedback", methods=["POST"])
def feedback():
    data                   = request.json
    feedback_text          = data.get("feedback", "").strip()
    current_filename       = data.get("current_filename", "").strip()
    current_image_filenames = data.get("current_image_filenames", [])
    title                  = data.get("title", "").strip()
    image_styles           = data.get("image_styles", [])

    if not feedback_text or not current_filename:
        return jsonify({"error": "Feedback and current article filename are required."}), 400

    html_path = os.path.join(OUTPUT_DIR, current_filename)
    if not os.path.exists(html_path):
        return jsonify({"error": f"Article '{current_filename}' not found."}), 404

    return stream_pipeline(
        lambda cb: run_feedback_pipeline(
            feedback_text, current_filename, current_image_filenames,
            title, image_styles, progress_callback=cb
        )
    )



@app.route("/article-html/<filename>")
def article_html_raw(filename):
    html_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(html_path):
        return "Article not found", 404
    with open(html_path, encoding="utf-8") as f:
        return f.read(), 200, {"Content-Type": "text/html"}

@app.route("/article/<filename>")
def view_article(filename):
    html_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(html_path):
        return render_template("error.html",
                               error_message=f"Article file '{filename}' not found."), 404
    with open(html_path, encoding="utf-8") as f:
        article_html = f.read()
    title = filename.replace(".html", "").replace("_", " ").title()
    return render_template("article.html", title=title, article_html=article_html)


@app.route("/delete", methods=["POST"])
def delete_history():
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
    app.run(debug=True, port=5001)