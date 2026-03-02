"""
prompts.py
──────────
Pure string-building functions for every LLM prompt in the pipeline.
No side effects — safe to import anywhere and easy to iterate on independently.
"""


def build_refinement_prompt(article_query: str, image_styles: list[str]) -> str:
    styles_block = "\n".join(f"IMAGE_{i+1}: {s}" for i, s in enumerate(image_styles))
    return f"""You are refining user inputs for a fashion magazine generator.
All inputs may contain filler words, instructions, or vague descriptions.
Your job is to return clean, optimised versions for each component.

CONTEXT: This is for a fashion magazine app used at ETH Zurich (a Swiss university).
"ETH" always refers to ETH Zurich university — NEVER to Ethereum or cryptocurrency.

ARTICLE QUERY: {article_query}

IMAGE INPUTS:
{styles_block}

Return EXACTLY this format (one line each, no extra text):
INTENT: <3-6 keywords, core fashion topic only, no filler>
ARTICLE: <refined article query, editorial tone, specific>
IMAGE_1: <refined DALL-E prompt — visual description, fashion photography language>
IMAGE_2: <refined DALL-E prompt — if exists>
...and so on for each image

Rules:
- INTENT:   strip all filler ("I want", "generate", "please", "fashion magazine about"), max 8 words
- ARTICLE:  keep the full topic, make it editorial and specific, max 30 words
- IMAGE_X:  keep it SHORT and visual — max 15 words
            if input is already descriptive, keep it nearly as-is
            only add: photography style or lighting if missing
            do NOT write full sentences or paragraphs
"""


def build_article_prompt(query: str) -> str:
    return f"Write a fashion article about: {query}"


def build_fresh_html_prompt(title: str, article_query: str, relevant_fashion_data: str,
                             article_text: str, image_filenames: list[str]) -> str:
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

RELEVANT FASHION DATA (most similar products/trends retrieved for this topic):
{relevant_fashion_data}
"""


def build_feedback_html_prompt(existing_html: str, text_feedback: str,
                                updated_image_map: dict) -> str:
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


def inject_validation_errors(base_prompt: str, errors: list[str]) -> str:
    """Appends a list of HTML validation errors to an existing prompt for retry attempts."""
    if not errors:
        return base_prompt
    error_lines = "\n".join(f"  - {e}" for e in errors)
    return base_prompt + f"\n\nIMPORTANT — Fix these HTML errors from the previous attempt:\n{error_lines}"