"""
app_colab.py — Fashion Magazine Generator for Google Colab

LLM helpers + register_all() to wire up the web UI.
Image generation is handled by the caller (notebook) via generate_fn,
which uses the DDPM score_model trained earlier in the notebook.
"""

import base64
import io
import json
import numpy as np
import requests
from datetime import datetime
import openai

BASE_URL = "https://litellm.sph-prod.ethz.ch/v1"

# ── Image helpers ──────────────────────────────────────────────────────────────

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

# ── LLM helpers ────────────────────────────────────────────────────────────────

def _get_client(api_key):
    return openai.OpenAI(api_key=api_key, base_url=BASE_URL)

def _call_llm(prompt, api_key, max_tokens=4096):
    client = _get_client(api_key)
    resp = client.chat.completions.create(
        model="anthropic/claude-sonnet-4-5",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content

def _should_regenerate(feedback, api_key):
    if not feedback or not feedback.strip():
        return False
    resp = _call_llm(
        f'Does this feedback request a visual image change? "{feedback}". Reply ONLY "yes" or "no".',
        api_key, max_tokens=5
    )
    return resp.strip().lower().startswith("yes")

# ── HTML building helpers ──────────────────────────────────────────────────────

def _build_fresh_prompt(title, article_query, trend_data, article_text, num_images):
    img_list = "\n".join(f"  - IMAGE_{i+1}" for i in range(num_images))
    return f"""You are a fashion expert, editor, and web designer.
Produce a COMPLETE, beautiful, standalone HTML page combining the article and ALL images.

ARTICLE TITLE: {title}

AVAILABLE IMAGES (use these EXACT strings as <img> src values):
{img_list}

RULES:
\u2022 Return a COMPLETE HTML document (<!DOCTYPE html> \u2026 </html>)
\u2022 Embedded CSS in a <style> tag \u2014 modern, elegant, magazine-like
\u2022 Use "{title}" as <h1>
\u2022 Place ALL image src values listed above exactly as shown (e.g. src="IMAGE_1")
\u2022 Responsive layout, no external CDN or font URLs
\u2022 Return ONLY the raw HTML, no markdown fences, no explanation

DRAFT ARTICLE:
{article_text}

ARTICLE TOPIC: {article_query}

FASHION TREND DATA (use as inspiration):
{trend_data}
"""

def _build_feedback_prompt(article_text, text_feedback):
    return f"""You are a fashion editor. Modify the following article based on the feedback below.
Return ONLY the updated article text, no HTML, no explanation.

FEEDBACK: {text_feedback}

ORIGINAL ARTICLE:
{article_text}
"""

def _clean_html(html):
    html = html.strip()
    if html.startswith("```"):
        lines = html.split("\n")[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        html = "\n".join(lines)
    return html

def _embed_images(html, images_b64):
    for i, b64 in enumerate(images_b64):
        data_uri = f"data:image/png;base64,{b64}"
        html = html.replace(f'src="IMAGE_{i+1}"', f'src="{data_uri}"')
        html = html.replace(f"src='IMAGE_{i+1}'", f"src='{data_uri}'")
        html = html.replace(f"IMAGE_{i+1}", data_uri)
    return html

def _html_to_b64(html):
    return base64.b64encode(html.encode("utf-8")).decode("ascii")

# ── Fashion data loader ────────────────────────────────────────────────────────

def _load_fashion_data():
    paths = [
        "/content/fashion_data_2018_2022.xls",          # downloaded by curl in section 7
        "/content/drive/MyDrive/BMAI/week2/notebook/Data/fashion_data_2018_2022.xls",
    ]
    for path in paths:
        try:
            import pandas as pd
            df = pd.read_excel(path)
            print(f"[app_colab] Loaded fashion data from {path}")
            return df.to_string()
        except Exception:
            continue
    print("[app_colab] Warning: could not load fashion data.")
    return "No fashion trend data available."

# ── register_all ───────────────────────────────────────────────────────────────

def register_all(generate_fn, api_key, html_url=None, branch="week2/fashion-magazine-app", repo="eth-bmai-fs26/project"):
    """
    Wire up the Colab web app.

    Parameters
    ----------
    generate_fn : callable
        generate_fn(class_label: int) -> str
        Takes a FashionMNIST class label (0-9) and returns a base64 PNG string.
        This function must be defined in the notebook because it needs score_model.
    api_key : str
        LiteLLM / OpenAI API key for LLM calls.
    html_url : str, optional
        Direct raw URL to ddpm_app_colab.html. Overrides branch.
    branch : str
        GitHub branch to fetch ddpm_app_colab.html from if html_url is not given.
    """
    from google.colab import output
    import IPython

    fashion_data  = _load_fashion_data()
    history       = []
    current_state = {}

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def colabGenerate(title, image_styles_json, article_query, api_key_override=""):
        key          = api_key_override or api_key
        image_styles = json.loads(image_styles_json) if isinstance(image_styles_json, str) else image_styles_json
        if not image_styles:
            image_styles = ["fashion editorial"]

        # Generate one DDPM image per style — class label cycles through 0-9
        images_b64   = [generate_fn(class_label=i % 10) for i in range(len(image_styles))]
        article_text = _call_llm(f"Write a fashion article about: {article_query}", key)
        prompt       = _build_fresh_prompt(title, article_query, fashion_data, article_text, len(images_b64))
        raw_html     = _call_llm(prompt, key)
        html         = _embed_images(_clean_html(raw_html), images_b64)

        current_state.update({
            "title":        title,
            "article_text": article_text,
            "images_b64":   images_b64,
            "image_styles": image_styles,
        })
        history.append({
            "title":         title,
            "article_query": article_query,
            "timestamp":     datetime.now().isoformat(),
        })
        return json.dumps({"html_b64": _html_to_b64(html), "num_images": len(images_b64)})

    def colabFeedback(text_feedback, image_feedbacks_json, title, api_key_override=""):
        key = api_key_override or api_key
        if not current_state:
            return json.dumps({"error": "No article generated yet."})

        image_feedbacks = json.loads(image_feedbacks_json) if isinstance(image_feedbacks_json, str) else image_feedbacks_json
        article_text    = current_state["article_text"]
        images_b64      = list(current_state["images_b64"])

        if text_feedback and text_feedback.strip():
            article_text = _call_llm(_build_feedback_prompt(article_text, text_feedback), key)

        for idx_str, feedback in image_feedbacks.items():
            idx = int(idx_str)
            if 1 <= idx <= len(images_b64) and _should_regenerate(feedback, key):
                # Pick a different class label so the regenerated image is visibly different
                images_b64[idx - 1] = generate_fn(class_label=(idx * 3 + 2) % 10)

        prompt   = _build_fresh_prompt(title, "fashion article", fashion_data, article_text, len(images_b64))
        raw_html = _call_llm(prompt, key)
        html     = _embed_images(_clean_html(raw_html), images_b64)

        current_state["article_text"] = article_text
        current_state["images_b64"]   = images_b64
        history.append({
            "title":         f"[FEEDBACK] {title}",
            "article_query": f"[FEEDBACK] {text_feedback}",
            "timestamp":     datetime.now().isoformat(),
        })
        return json.dumps({"html_b64": _html_to_b64(html), "num_images": len(images_b64)})

    def colabGetHistory():
        return json.dumps(history[-10:])

    def colabDeleteHistory():
        history.clear()
        current_state.clear()
        return json.dumps({"success": True, "deleted": ["history"]})

    # Register all Python callbacks so the HTML can call them via invokeFunction
    output.register_callback('colabGenerate',      colabGenerate)
    output.register_callback('colabFeedback',      colabFeedback)
    output.register_callback('colabGetHistory',    colabGetHistory)
    output.register_callback('colabDeleteHistory', colabDeleteHistory)

    # ── Fetch and display the HTML UI ──────────────────────────────────────────
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
        print(f"[app_colab] Could not fetch HTML from {html_url}: {e}")
        print("Tip: display the HTML manually with:")
        print("  IPython.display.display(IPython.display.HTML(open('ddpm_app_colab.html').read()))")
