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
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
API_KEY     = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Please create a .env file with your key.")

BASE_URL    = "https://litellm.sph-prod.ethz.ch/v1"
DATA_PATH   = os.path.join(BASE_DIR, "Data", "fashion_data_2018_2022.xls")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output")
HISTORY_DIR = os.path.join(BASE_DIR, "history")

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
    """Appends entry to history/{kind}.json — we save but never load for generation."""
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
    """Used only for the history panel in the UI — never for generation."""
    path = os.path.join(HISTORY_DIR, f"{kind}.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


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
    # DALL-E returns either a URL or base64 — handle both
    raw = image_data.url if image_data.url else image_data.b64_json

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"image_{ts}_{style[:20].replace(' ','_')}.png"
    path = os.path.join(OUTPUT_DIR, filename)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if raw.startswith("http"):
        # URL → download image bytes and save to disk
        r = requests.get(raw, timeout=30)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    else:
        # Base64 → decode and save to disk
        with open(path, "wb") as f:
            f.write(base64.b64decode(raw))

    save_history("images", {"title": title, "style": style, "filename": filename})
    return filename


def generate_article(query: str) -> str:
    """Generates article text via Claude Sonnet."""
    client = get_client()
    resp = client.chat.completions.create(
        model="anthropic/claude-sonnet-4-5",
        messages=[{"role": "user", "content": f"Write a fashion article about: {query}"}],
    )
    article = resp.choices[0].message.content
    save_history("text", {"query": query, "article_preview": article[:300]})
    return article


def should_regenerate_image(feedback: str) -> bool:
    """
    Uses LLM to decide if the user actually wants the image changed.
    Returns False if feedback is empty, positive-only, or says keep as is.
    Returns True if the user wants any visual change.
    """
    if not feedback or not feedback.strip():
        return False

    client = get_client()
    prompt = f"""You are deciding whether a user wants an image regenerated.

User feedback for this image: "{feedback}"

Return "yes" if the user wants ANY visual change to this image.
Return "no" if:
- The input is empty
- The user says they like it / it's fine / keep it / no change
- The feedback is purely a compliment with no change request

Respond ONLY with "yes" or "no"."""

    resp = client.chat.completions.create(
        model="anthropic/claude-sonnet-4-5",
        messages=[{"role": "user", "content": prompt}],
    )
    answer = resp.choices[0].message.content.strip().lower()
    print(f"[should_regenerate_image] feedback='{feedback}' → {answer}")
    return answer == "yes"


def extract_image_style_from_feedback(feedback: str, original_style: str) -> str:
    """
    Extracts a new DALL-E prompt from the user's image feedback.
    Combines what the user wants changed with what should stay the same.
    """
    client = get_client()
    prompt = f"""You are extracting an image generation prompt from user feedback.

Original image style: "{original_style}"
User feedback: "{feedback}"

Write a detailed DALL-E 3 prompt for the NEW image.
- Incorporate the user's requested changes
- Keep elements from the original style that the user did not mention changing
- Make it fashion-focused and visually compelling
- Be specific and descriptive

Respond ONLY with the image generation prompt, no explanation, no quotes."""

    resp = client.chat.completions.create(
        model="anthropic/claude-sonnet-4-5",
        messages=[{"role": "user", "content": prompt}],
    )
    new_style = resp.choices[0].message.content.strip()
    print(f"[extract_image_style] '{original_style}' → '{new_style}'")
    return new_style


def build_fresh_prompt(title: str, article_query: str, trend_data: str,
                       article_text: str, image_filenames: list) -> str:
    """Prompt for generating a brand new article HTML from scratch."""
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

FASHION TREND DATA (use as inspiration and grounding):
{trend_data}
"""


def build_feedback_prompt(existing_html: str, text_feedback: str,
                          updated_image_map: dict) -> str:
    """
    Prompt for applying targeted feedback to an existing article.
    updated_image_map: {old_filename: new_filename} for regenerated images.
    """
    image_replacements = ""
    if updated_image_map:
        image_replacements = "IMAGE PATH REPLACEMENTS (update these src attributes):\n"
        for old, new in updated_image_map.items():
            image_replacements += f"  - Replace /output/{old} with /output/{new}\n"

    text_section = ""
    if text_feedback and text_feedback.strip():
        text_section = f"TEXT CHANGES REQUESTED:\n{text_feedback}\n"

    return f"""You are a fashion editor making precise targeted edits to an existing HTML article.

{text_section}
{image_replacements}

RULES:
• Apply ONLY the changes listed above
• Keep ALL other text, styling, layout, and images exactly as they are
• Return the COMPLETE updated HTML document
• Return ONLY the raw HTML, no markdown fences, no explanation

EXISTING HTML TO EDIT:
{existing_html}
"""


def validate_html(html: str, image_filenames: list) -> tuple[bool, list, str]:
    """
    Validates generated HTML: structure, tag balancing, CSS braces, image references.
    Returns (is_valid, errors, cleaned_html).
    """
    import re
    from html.parser import HTMLParser

    errors = []

    # 1. Strip markdown fences if LLM wrapped output
    s = html.strip()
    if s.startswith("```"):
        lines = s.split("\n")[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        html = "\n".join(lines)

    # 2. Basic structural checks
    required = {
        "<!DOCTYPE html>": "missing <!DOCTYPE html> declaration",
        "<html":           "missing <html> tag",
        "</html>":         "missing closing </html> tag",
        "<body":           "missing <body> tag",
        "</body>":         "missing closing </body> tag",
        "<style":          "missing <style> tag (no CSS)",
    }
    for token, message in required.items():
        if token not in html:
            errors.append(message)

    # 3. Image reference checks
    for f in image_filenames:
        if f not in html:
            errors.append(f"image '{f}' not referenced in HTML")

    # 4. Tag balancing and attribute validation
    VOID_ELEMENTS = {
        "area", "base", "br", "col", "embed", "hr", "img",
        "input", "link", "meta", "param", "source", "track", "wbr",
    }
    UNIQUE_TAGS = {"html", "head", "body", "title"}

    class HTMLValidator(HTMLParser):
        def __init__(self):
            super().__init__()
            self.stack = []
            self.tag_counts = {}
            self.parse_errors = []

        def handle_starttag(self, tag, attrs):
            attr_dict = dict(attrs)
            self.tag_counts[tag] = self.tag_counts.get(tag, 0) + 1
            if tag in UNIQUE_TAGS and self.tag_counts[tag] > 1:
                self.parse_errors.append(f"duplicate <{tag}> tag found")
            if tag == "img":
                if "src" not in attr_dict or not attr_dict.get("src", "").strip():
                    self.parse_errors.append("an <img> tag is missing a valid src attribute")
            if tag not in VOID_ELEMENTS:
                self.stack.append(tag)

        def handle_endtag(self, tag):
            if tag in VOID_ELEMENTS:
                return
            if not self.stack:
                self.parse_errors.append(f"unexpected closing tag </{tag}> (stack is empty)")
                return
            if self.stack[-1] == tag:
                self.stack.pop()
            else:
                if tag in self.stack:
                    while self.stack and self.stack[-1] != tag:
                        skipped = self.stack.pop()
                        self.parse_errors.append(f"unclosed tag <{skipped}> before </{tag}>")
                    if self.stack:
                        self.stack.pop()
                else:
                    self.parse_errors.append(f"unexpected closing tag </{tag}> (no matching open tag)")

        def close(self):
            super().close()
            for tag in self.stack:
                self.parse_errors.append(f"unclosed tag <{tag}> at end of document")

    try:
        v = HTMLValidator()
        v.feed(html)
        v.close()
        errors.extend(v.parse_errors)
    except Exception as exc:
        errors.append(f"HTML parsing exception: {exc}")

    # 5. CSS brace balance check
    style_blocks = re.findall(r"<style[^>]*>(.*?)</style>", html, re.DOTALL | re.IGNORECASE)
    if style_blocks:
        css = "\n".join(style_blocks)
        if css.count("{") != css.count("}"):
            errors.append(f"CSS brace mismatch: {css.count('{')}' opening vs {css.count('}')}' closing")

    return (len(errors) == 0, errors, html)




def patch_truncated_html(html: str) -> str:
    """
    If the LLM output was truncated and is missing closing tags,
    append them so validation has a chance to pass.
    """
    html = html.strip()
    if "</li>" not in html and "<li" in html:
        html += "\n</li>"
    if "</ul>" not in html and "<ul" in html:
        html += "\n</ul>"
    if "</ol>" not in html and "<ol" in html:
        html += "\n</ol>"
    if "</p>" not in html.split("<body")[-1] and "<p" in html.split("<body")[-1]:
        html += "\n</p>"
    if "</article>" not in html and "<article" in html:
        html += "\n</article>"
    if "</section>" not in html and "<section" in html:
        html += "\n</section>"
    opens  = html.count("<div")
    closes = html.count("</div>")
    html  += "\n</div>" * max(0, opens - closes)
    if "</main>" not in html and "<main" in html:
        html += "\n</main>"
    if "</body>" not in html:
        html += "\n</body>"
    if "</html>" not in html:
        html += "\n</html>"
    return html

def safe_html_pipeline(generate_fn, image_filenames: list,
                       emit=None, max_attempts: int = 3) -> tuple[str, bool]:
    """
    Tries to generate and validate HTML up to max_attempts times.
    On each retry, passes previous errors back to the LLM so it can fix them.
    Falls back to emergency_fallback_html if all attempts fail.

    generate_fn(attempt, previous_errors) → raw HTML string
    """
    last_errors = []
    last_html   = ""

    for attempt in range(1, max_attempts + 1):
        if emit:
            emit(f"HTML generation attempt {attempt}/{max_attempts}…")
        raw = generate_fn(attempt, last_errors)
        fixed = fix_image_paths(raw, image_filenames)
        # If output was truncated, close any missing structural tags
        fixed = patch_truncated_html(fixed)
        valid, errors, cleaned = validate_html(fixed, image_filenames)

        if valid:
            if emit:
                emit("✅ HTML validated successfully.")
            return cleaned, True

        last_errors = errors
        last_html   = cleaned
        if emit:
            emit(f"⚠️ Validation failed ({len(errors)} errors), retrying with corrections…")
        print(f"[Attempt {attempt}/{max_attempts}] Errors: {errors}")

    # All attempts exhausted
    if emit:
        emit("🚨 Max retries reached — using emergency fallback HTML.")
    return last_html, False

def fix_image_paths(html: str, image_filenames: list) -> str:
    """
    Corrects image src paths in LLM-generated HTML.
    LLMs sometimes use wrong path variants — this normalises them all to /output/<filename>.
    """
    for filename in image_filenames:
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


def emergency_fallback_html(title: str, article_text: str, image_filenames: list) -> str:
    """Assembles a basic valid HTML page if all LLM retries fail."""
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


# ── Fresh Pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(title: str, image_styles: list, article_query: str,
                 progress_callback=None):
    """
    Full generation pipeline — generates everything from scratch.
    Uses only the fashion dataset as grounding, no prior session history.
    """
    def emit(step, total, msg):
        if progress_callback:
            progress_callback(step, total, msg)
        print(f"[{step}/{total}] {msg}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total = 1 + len(image_styles) + 2  # trend data + images + article + html

    step = 1
    emit(step, total, "Fetching fashion trend data…")
    trend_data = FASHION_DATA

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
    base_prompt = build_fresh_prompt(title, article_query, trend_data,
                                     article_text, image_filenames)
    client = get_client()

    def generate_fn(attempt, previous_errors):
        # On retries, feed the previous errors back to the LLM so it can fix them
        prompt = base_prompt
        if previous_errors:
            prompt += f"""

IMPORTANT — Your previous attempt had these HTML errors. Please fix all of them:
{chr(10).join(f"  - {e}" for e in previous_errors)}
"""
        return client.chat.completions.create(
            model="anthropic/claude-sonnet-4-5",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
        ).choices[0].message.content

    final_html, succeeded = safe_html_pipeline(
        generate_fn=generate_fn,
        image_filenames=image_filenames,
        emit=lambda msg: emit(step, total, msg),
    )
    if not succeeded:
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
        "html_filename": html_filename,
        "image_filenames": image_filenames,
        "article_text": article_text,
    }


# ── Feedback Pipeline ──────────────────────────────────────────────────────────

def run_feedback_pipeline(text_feedback: str, image_feedbacks: dict,
                          current_filename: str, current_image_filenames: list,
                          title: str, image_styles: list,
                          progress_callback=None):
    """
    Feedback pipeline — applies targeted edits to an existing article.
    image_feedbacks: {1: "feedback for image 1", 2: "feedback for image 2", ...}
    text_feedback: free text for article text changes.
    Only regenerates images where feedback requests a change.
    """
    def emit(step, total, msg):
        if progress_callback:
            progress_callback(step, total, msg)
        print(f"[{step}/{total}] {msg}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total = 2 + len(image_feedbacks) + 1  # load + images + apply feedback

    step = 1
    emit(step, total, "Loading existing article…")
    html_path = os.path.join(OUTPUT_DIR, current_filename)
    with open(html_path, encoding="utf-8") as f:
        existing_html = f.read()

    # Regenerate only images where user actually requested a change
    updated_image_map = {}
    final_image_filenames = list(current_image_filenames)

    for idx, img_feedback in image_feedbacks.items():
        step += 1
        idx = int(idx)
        emit(step, total, f"Checking image {idx} feedback…")

        if not should_regenerate_image(img_feedback):
            emit(step, total, f"↩ Image {idx} — no change requested, keeping original.")
            continue

        original_style = image_styles[idx - 1] if idx <= len(image_styles) else f"image {idx}"
        new_style = extract_image_style_from_feedback(img_feedback, original_style)
        emit(step, total, f"Regenerating image {idx}: {new_style[:50]}…")
        try:
            old_filename = current_image_filenames[idx - 1]
            new_filename = generate_image(title, new_style)
            updated_image_map[old_filename] = new_filename
            final_image_filenames[idx - 1] = new_filename
        except Exception as e:
            emit(step, total, f"⚠️ Image {idx} regeneration failed: {e}")

    # Apply text feedback + image replacements to existing HTML
    step += 1
    emit(step, total, "Applying feedback to article…")
    base_prompt = build_feedback_prompt(existing_html, text_feedback, updated_image_map)
    client = get_client()

    def generate_fn(attempt, previous_errors):
        prompt = base_prompt
        if previous_errors:
            prompt += f"""

IMPORTANT — Your previous attempt had these HTML errors. Please fix all of them:
{chr(10).join(f"  - {e}" for e in previous_errors)}
"""
        return client.chat.completions.create(
            model="anthropic/claude-sonnet-4-5",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
        ).choices[0].message.content

    final_html, succeeded = safe_html_pipeline(
        generate_fn=generate_fn,
        image_filenames=final_image_filenames,
        emit=lambda msg: emit(step, total, msg),
    )
    if not succeeded:
        emit(step, total, "🚨 Max retries reached — keeping original article.")
        final_html = existing_html

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_filename = f"article_{ts}_revised.html"
    with open(os.path.join(OUTPUT_DIR, html_filename), "w") as f:
        f.write(final_html)

    save_history("articles", {
        "title": title,
        "article_query": f"[FEEDBACK] {text_feedback}",
        "image_styles": image_styles,
        "html_filename": html_filename,
        "image_filenames": final_image_filenames,
        "article_preview": text_feedback[:300],
        "revised_from": current_filename,
    })

    emit(step, total, f"✅ Feedback applied! Saved as {html_filename}")
    return {
        "html_filename": html_filename,
        "image_filenames": final_image_filenames,
    }


# ── SSE Stream Helper ──────────────────────────────────────────────────────────

def stream_pipeline(pipeline_fn):
    """Generic SSE streamer — runs pipeline in background thread, streams progress to browser."""
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
            q.put(None)  # sentinel — tells stream to stop

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
    data                    = request.json
    text_feedback           = data.get("text_feedback", "").strip()
    image_feedbacks         = data.get("image_feedbacks", {})   # {index: feedback_text}
    current_filename        = data.get("current_filename", "").strip()
    current_image_filenames = data.get("current_image_filenames", [])
    title                   = data.get("title", "").strip()
    image_styles            = data.get("image_styles", [])

    if not current_filename:
        return jsonify({"error": "Current article filename is required."}), 400

    html_path = os.path.join(OUTPUT_DIR, current_filename)
    if not os.path.exists(html_path):
        return jsonify({"error": f"Article '{current_filename}' not found."}), 404

    return stream_pipeline(
        lambda cb: run_feedback_pipeline(
            text_feedback, image_feedbacks, current_filename,
            current_image_filenames, title, image_styles,
            progress_callback=cb
        )
    )


@app.route("/article-html/<filename>")
def article_html_raw(filename):
    """Returns raw HTML for iframe rendering — bypasses template to avoid path issues."""
    html_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(html_path):
        return "Article not found", 404
    with open(html_path, encoding="utf-8") as f:
        return f.read(), 200, {"Content-Type": "text/html"}


@app.route("/article/<filename>")
def view_article(filename):
    """Renders article in article.html template wrapper."""
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
    """Deletes all history JSON files."""
    deleted = []
    for kind in ["articles", "images", "text"]:
        path = os.path.join(HISTORY_DIR, f"{kind}.json")
        if os.path.exists(path):
            os.remove(path)
            deleted.append(f"{kind}.json")
    return jsonify({"success": True, "deleted": deleted})


@app.route("/history/<kind>")
def get_history(kind):
    """Returns history JSON for the UI history panel."""
    if kind not in ("articles", "images", "text"):
        return jsonify({"error": "Invalid history type"}), 400
    return jsonify(load_history(kind))


@app.route("/output/<path:filename>")
def serve_output(filename):
    """Serves generated files (images and HTML) to the browser."""
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, port=5001)