"""
app_colab.py — Fashion Magazine Generator for Google Colab

Same functions as app.py with two differences:
  1. generate_image() uses the DDPM model instead of DALL-E
  2. register_all() replaces Flask routes — displays the app via IPython.display
"""

import os
import json
import base64
import io
import queue
import re
import numpy as np
import requests
from datetime import datetime
from html.parser import HTMLParser
import openai
import pandas as pd

BASE_URL = "https://litellm.sph-prod.ethz.ch/v1"

# Module-level state (set by register_all)
_state = {"generate_fn": None, "api_key": None}
FASHION_DATA = "No fashion trend data available."

# ── Config helpers ────────────────────────────────────────────────────────────

def _load_fashion_data():
    paths = [
        "/content/fashion_data_2018_2022.xls",
        "/content/Data/fashion_data_2018_2022.xls",
        "/content/drive/MyDrive/BMAI/week2/notebook/Data/fashion_data_2018_2022.xls",
    ]
    for path in paths:
        try:
            df = pd.read_excel(path)
            print(f"[app_colab] Loaded fashion data from {path}")
            return df.to_string()
        except Exception:
            continue
    print("[app_colab] Warning: could not load fashion data.")
    return "No fashion trend data available."


def get_client():
    return openai.OpenAI(api_key=_state["api_key"], base_url=BASE_URL)


# ── History — in-memory (same interface as app.py) ───────────────────────────

_history_store = {"articles": [], "images": [], "text": []}


def save_history(kind: str, entry: dict):
    entry["timestamp"] = datetime.now().isoformat()
    _history_store[kind].append(entry)


def load_history(kind: str) -> list:
    return list(_history_store.get(kind, []))


# ── Image generation — ONLY function that differs from app.py ─────────────────

def tensor_to_b64(tensor_chw):
    """Convert a (C, H, W) tensor in [0,1] to a base64-encoded PNG string."""
    from PIL import Image
    img_np = tensor_chw.cpu().numpy()
    if img_np.shape[0] == 1:
        pil = Image.fromarray((img_np.squeeze(0) * 255).astype(np.uint8), mode='L').convert('RGB')
    else:
        pil = Image.fromarray((np.transpose(img_np, (1, 2, 0)) * 255).astype(np.uint8), mode='RGB')
    pil = pil.resize((512, 512), Image.NEAREST)
    buf = io.BytesIO()
    pil.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def generate_image(title: str, style: str) -> str:
    """
    Generates one image using the trained DDPM model.
    Returns base64-encoded PNG string (app.py returns a filename).
    style can be a digit "0"-"9" (direct class label from the UI dropdown)
    or a text string (class label derived via hash).
    """
    class_label = int(style.strip()) % 10 if style.strip().isdigit() else abs(hash(style)) % 10
    b64 = _state["generate_fn"](class_label=class_label)
    save_history("images", {"title": title, "style": style})
    return b64


# ── LLM helpers — identical to app.py ────────────────────────────────────────

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
    return answer == "yes"


def extract_image_style_from_feedback(feedback: str, original_style: str) -> str:
    """
    Extracts a new style description from user feedback.
    Used to pick a different DDPM class label via hash.
    """
    client = get_client()
    prompt = f"""You are extracting an image generation prompt from user feedback.

Original image style: "{original_style}"
User feedback: "{feedback}"

Write a detailed style description for the NEW image.
- Incorporate the user's requested changes
- Keep elements from the original style that the user did not mention changing
- Make it fashion-focused and visually compelling

Respond ONLY with the style description, no explanation, no quotes."""

    resp = client.chat.completions.create(
        model="anthropic/claude-sonnet-4-5",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


# ── HTML building — same structure as app.py, IMAGE_N instead of file paths ───

def build_fresh_prompt(title: str, article_query: str, trend_data: str,
                       article_text: str, images_b64: list) -> str:
    """Prompt for generating a brand new article HTML from scratch."""
    img_tags = "\n".join(
        f'  <img src="IMAGE_{i+1}" alt="fashion image {i+1}" style="max-width:100%;display:block;margin:1.5rem auto;">'
        for i in range(len(images_b64))
    )
    return f"""You are a fashion expert, editor, and web designer.
Produce a COMPLETE, beautiful, standalone HTML page combining the article and ALL images.

ARTICLE TITLE: {title}

CRITICAL — copy these EXACT <img> tags into your HTML, do NOT change the src attribute:
{img_tags}

RULES:
\u2022 Return a COMPLETE HTML document (<!DOCTYPE html> \u2026 </html>)
\u2022 Embedded CSS in a <style> tag \u2014 modern, elegant, magazine-like
\u2022 Use "{title}" as <h1>
\u2022 The src of every image MUST be exactly IMAGE_1, IMAGE_2, etc. \u2014 nothing else
\u2022 Responsive layout, no external CDN/fonts
\u2022 Return ONLY the raw HTML, no markdown fences

DRAFT ARTICLE:
{article_text}

ARTICLE TOPIC: {article_query}

FASHION TREND DATA (use as inspiration and grounding):
{trend_data}
"""


def build_feedback_prompt(existing_article: str, text_feedback: str) -> str:
    """Prompt for applying text feedback to the article."""
    return f"""You are a fashion editor making precise targeted edits to an existing article.

TEXT CHANGES REQUESTED:
{text_feedback}

RULES:
\u2022 Apply ONLY the changes listed above
\u2022 Keep ALL other content exactly as it is
\u2022 Return ONLY the updated article text, no HTML, no explanation

EXISTING ARTICLE:
{existing_article}
"""


def _embed_images(html: str, images_b64: list) -> str:
    """Replace IMAGE_N placeholders with base64 data URIs."""
    for i, b64 in enumerate(images_b64):
        data_uri = f"data:image/png;base64,{b64}"
        html = html.replace(f'src="IMAGE_{i+1}"', f'src="{data_uri}"')
        html = html.replace(f"src='IMAGE_{i+1}'", f"src='{data_uri}'")
        html = html.replace(f"IMAGE_{i+1}", data_uri)
    return html


def _html_to_b64(html: str) -> str:
    return base64.b64encode(html.encode("utf-8")).decode("ascii")


# ── Validation — identical to app.py, adapted for IMAGE_N placeholders ────────

def validate_html(html: str, num_images: int) -> tuple:
    """
    Validates generated HTML: structure, tag balancing, CSS braces, image placeholders.
    Returns (is_valid, errors, cleaned_html).
    """
    errors = []

    s = html.strip()
    if s.startswith("```"):
        lines = s.split("\n")[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        html = "\n".join(lines)

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

    for i in range(1, num_images + 1):
        if f'src="IMAGE_{i}"' not in html and f"src='IMAGE_{i}'" not in html:
            errors.append(f'IMAGE_{i} not found as img src (must be src="IMAGE_{i}")')

    VOID_ELEMENTS = {"area","base","br","col","embed","hr","img","input","link","meta","param","source","track","wbr"}
    UNIQUE_TAGS   = {"html","head","body","title"}

    class HTMLValidator(HTMLParser):
        def __init__(self):
            super().__init__()
            self.stack, self.tag_counts, self.parse_errors = [], {}, []

        def handle_starttag(self, tag, attrs):
            self.tag_counts[tag] = self.tag_counts.get(tag, 0) + 1
            if tag in UNIQUE_TAGS and self.tag_counts[tag] > 1:
                self.parse_errors.append(f"duplicate <{tag}> tag found")
            if tag not in VOID_ELEMENTS:
                self.stack.append(tag)

        def handle_endtag(self, tag):
            if tag in VOID_ELEMENTS:
                return
            if not self.stack:
                self.parse_errors.append(f"unexpected closing tag </{tag}>")
                return
            if self.stack[-1] == tag:
                self.stack.pop()
            else:
                if tag in self.stack:
                    while self.stack and self.stack[-1] != tag:
                        self.parse_errors.append(f"unclosed tag <{self.stack.pop()}> before </{tag}>")
                    if self.stack:
                        self.stack.pop()
                else:
                    self.parse_errors.append(f"unexpected closing tag </{tag}>")

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

    style_blocks = re.findall(r"<style[^>]*>(.*?)</style>", html, re.DOTALL | re.IGNORECASE)
    if style_blocks:
        css = "\n".join(style_blocks)
        if css.count("{") != css.count("}"):
            errors.append(f"CSS brace mismatch: {css.count('{')} opening vs {css.count('}')} closing")

    return (len(errors) == 0, errors, html)


def patch_truncated_html(html: str) -> str:
    """If the LLM output was truncated, append missing closing tags."""
    html = html.strip()
    for tag in ["li", "ul", "ol", "article", "section", "main"]:
        if f"</{tag}>" not in html and f"<{tag}" in html:
            html += f"\n</{tag}>"
    if "</p>" not in html.split("<body")[-1] and "<p" in html.split("<body")[-1]:
        html += "\n</p>"
    opens  = html.count("<div")
    closes = html.count("</div>")
    html  += "\n</div>" * max(0, opens - closes)
    if "</body>" not in html:
        html += "\n</body>"
    if "</html>" not in html:
        html += "\n</html>"
    return html


def safe_html_pipeline(generate_fn, num_images: int,
                       emit=None, max_attempts: int = 3) -> tuple:
    """
    Tries to generate and validate HTML up to max_attempts times.
    generate_fn(attempt, previous_errors) → raw HTML string
    """
    last_errors = []
    last_html   = ""

    for attempt in range(1, max_attempts + 1):
        if emit:
            emit(f"HTML generation attempt {attempt}/{max_attempts}…")
        raw   = generate_fn(attempt, last_errors)
        fixed = patch_truncated_html(raw)
        valid, errors, cleaned = validate_html(fixed, num_images)

        if valid:
            if emit:
                emit("✅ HTML validated successfully.")
            return cleaned, True

        last_errors = errors
        last_html   = cleaned
        if emit:
            emit(f"⚠️ Validation failed ({len(errors)} errors), retrying…")

    if emit:
        emit("🚨 Max retries reached — using emergency fallback HTML.")
    return last_html, False


def emergency_fallback_html(title: str, article_text: str, num_images: int) -> str:
    """Assembles a basic valid HTML page if all LLM retries fail."""
    imgs = "\n".join(
        f'<img src="IMAGE_{i+1}" alt="Fashion image" style="max-width:100%;margin:1rem 0;">'
        for i in range(num_images)
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>{title}</title>
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


# ── Fresh Pipeline — same structure as app.py ─────────────────────────────────

def run_pipeline(title: str, image_styles: list, article_query: str,
                 progress_callback=None):
    """Full generation pipeline — same structure as app.py."""
    def emit(step, total, msg):
        if progress_callback:
            progress_callback(step, total, msg)

    total = 1 + len(image_styles) + 2
    step  = 1

    emit(step, total, "Fetching fashion trend data…")
    trend_data = FASHION_DATA

    images_b64 = []
    for i, style in enumerate(image_styles, 1):
        step += 1
        emit(step, total, f"Generating image {i}/{len(image_styles)}: {style}…")
        try:
            images_b64.append(generate_image(title, style))
        except Exception as e:
            emit(step, total, f"⚠️ Image {i} failed: {e}")

    step += 1
    emit(step, total, "Writing article…")
    article_text = generate_article(article_query)

    step += 1
    emit(step, total, "Assembling final HTML page…")
    base_prompt = build_fresh_prompt(title, article_query, trend_data, article_text, images_b64)
    client      = get_client()

    def gen_fn(attempt, previous_errors):
        prompt = base_prompt
        if previous_errors:
            prompt += "\n\nIMPORTANT — Fix these HTML errors from your previous attempt:\n"
            prompt += "\n".join(f"  - {e}" for e in previous_errors)
        return client.chat.completions.create(
            model="anthropic/claude-sonnet-4-5",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
        ).choices[0].message.content

    raw_html, succeeded = safe_html_pipeline(gen_fn, len(images_b64),
                                             emit=lambda msg: emit(step, total, msg))
    if not succeeded:
        raw_html = emergency_fallback_html(title, article_text, len(images_b64))

    # Do NOT embed images — JS will inject them directly into the DOM
    save_history("articles", {
        "title":           title,
        "article_query":   article_query,
        "image_styles":    image_styles,
        "article_preview": article_text[:300],
    })

    emit(step, total, "✅ Done!")
    return {
        "html_b64":      _html_to_b64(raw_html),
        "article_text":  article_text,
        "images_b64":    images_b64,
        "image_styles":  image_styles,
        "num_images":    len(images_b64),
    }


# ── Feedback Pipeline — same structure as app.py ──────────────────────────────

def run_feedback_pipeline(text_feedback: str, image_feedbacks: dict,
                          current_state: dict, title: str,
                          progress_callback=None):
    """Feedback pipeline — same structure as app.py."""
    def emit(step, total, msg):
        if progress_callback:
            progress_callback(step, total, msg)

    article_text = current_state["article_text"]
    images_b64   = list(current_state["images_b64"])
    image_styles = list(current_state["image_styles"])

    total = 2 + len(image_feedbacks) + 1
    step  = 1

    emit(step, total, "Loading current article state…")

    for idx_str, img_feedback in image_feedbacks.items():
        step += 1
        idx = int(idx_str)
        emit(step, total, f"Checking image {idx} feedback…")

        if not should_regenerate_image(img_feedback):
            emit(step, total, f"↩ Image {idx} — no change requested, keeping original.")
            continue

        original_style = image_styles[idx - 1] if idx <= len(image_styles) else f"image {idx}"
        new_style      = extract_image_style_from_feedback(img_feedback, original_style)
        emit(step, total, f"Regenerating image {idx}…")
        try:
            images_b64[idx - 1]   = generate_image(title, new_style)
            image_styles[idx - 1] = new_style
        except Exception as e:
            emit(step, total, f"⚠️ Image {idx} regeneration failed: {e}")

    step += 1
    emit(step, total, "Applying text feedback and rebuilding article…")

    if text_feedback and text_feedback.strip():
        article_text = get_client().chat.completions.create(
            model="anthropic/claude-sonnet-4-5",
            messages=[{"role": "user", "content": build_feedback_prompt(article_text, text_feedback)}],
        ).choices[0].message.content

    base_prompt = build_fresh_prompt(title, "fashion article", FASHION_DATA, article_text, images_b64)
    client      = get_client()

    def gen_fn(attempt, previous_errors):
        prompt = base_prompt
        if previous_errors:
            prompt += "\n\nIMPORTANT — Fix these HTML errors from your previous attempt:\n"
            prompt += "\n".join(f"  - {e}" for e in previous_errors)
        return client.chat.completions.create(
            model="anthropic/claude-sonnet-4-5",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
        ).choices[0].message.content

    raw_html, succeeded = safe_html_pipeline(gen_fn, len(images_b64),
                                             emit=lambda msg: emit(step, total, msg))
    if not succeeded:
        raw_html = emergency_fallback_html(title, article_text, len(images_b64))

    # Do NOT embed images — JS will inject them directly into the DOM
    save_history("articles", {
        "title":           f"[FEEDBACK] {title}",
        "article_query":   f"[FEEDBACK] {text_feedback}",
        "image_styles":    image_styles,
        "article_preview": text_feedback[:300],
    })

    emit(step, total, "✅ Feedback applied!")
    return {
        "html_b64":     _html_to_b64(raw_html),
        "article_text": article_text,
        "images_b64":   images_b64,
        "image_styles": image_styles,
        "num_images":   len(images_b64),
    }


# ── register_all — replaces Flask routes ─────────────────────────────────────

def register_all(generate_fn, api_key, html_url=None,
                 branch="week2/fashion-magazine-app", repo="eth-bmai-fs26/project"):
    """
    Wire up the Colab web app.
    Registers Python callbacks and displays the UI HTML inline via IPython.display.
    No Flask server or tunnel needed.
    """
    global FASHION_DATA
    _state["generate_fn"] = generate_fn
    _state["api_key"]     = api_key
    FASHION_DATA          = _load_fashion_data()

    for store in _history_store.values():
        store.clear()

    from google.colab import output
    import IPython.display

    _current_state = {}

    # ── Callbacks (replace Flask routes) ──────────────────────────────────────

    def colabGenerate(title, image_styles_json, article_query, api_key_override=""):
        if api_key_override:
            _state["api_key"] = api_key_override
        image_styles = json.loads(image_styles_json) if isinstance(image_styles_json, str) else image_styles_json
        if not image_styles:
            image_styles = ["fashion editorial"]
        result = run_pipeline(title, image_styles, article_query)
        _current_state.update({
            "title":        title,
            "article_text": result["article_text"],
            "images_b64":   result["images_b64"],
            "image_styles": result["image_styles"],
        })
        return json.dumps({"html_b64": result["html_b64"], "images_b64": result["images_b64"], "num_images": result["num_images"]})

    def colabFeedback(text_feedback, image_feedbacks_json, title, api_key_override=""):
        if api_key_override:
            _state["api_key"] = api_key_override
        if not _current_state:
            return json.dumps({"error": "No article generated yet."})
        image_feedbacks = json.loads(image_feedbacks_json) if isinstance(image_feedbacks_json, str) else image_feedbacks_json
        result = run_feedback_pipeline(text_feedback, image_feedbacks, _current_state, title)
        _current_state.update({
            "article_text": result["article_text"],
            "images_b64":   result["images_b64"],
            "image_styles": result["image_styles"],
        })
        return json.dumps({"html_b64": result["html_b64"], "images_b64": result["images_b64"], "num_images": result["num_images"]})

    def colabGetHistory():
        return json.dumps(load_history("articles")[-10:])

    def colabDeleteHistory():
        for store in _history_store.values():
            store.clear()
        _current_state.clear()
        return json.dumps({"success": True, "deleted": ["history"]})

    output.register_callback('colabGenerate',      colabGenerate)
    output.register_callback('colabFeedback',      colabFeedback)
    output.register_callback('colabGetHistory',    colabGetHistory)
    output.register_callback('colabDeleteHistory', colabDeleteHistory)

    # ── Display the UI HTML inline — no server needed ─────────────────────────
    if html_url is None:
        html_url = (
            f"https://raw.githubusercontent.com/{repo}"
            f"/refs/heads/{branch}/visualization/ddpm_app_colab.html"
        )
    try:
        response = requests.get(html_url)
        response.raise_for_status()
        IPython.display.display(IPython.display.HTML(response.text))
    except Exception as e:
        print(f"[app_colab] Could not fetch UI from {html_url}: {e}")
        print("Tip: IPython.display.display(IPython.display.HTML(open('ddpm_app_colab.html').read()))")
