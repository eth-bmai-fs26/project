 # imports
import os
from openai import OpenAI
import openai
import pandas as pd
from openai import OpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage
import json
import requests
import base64

# pip install langchain langchain-core
# pip install xlrd

## Set variables
api_key =#yourkey  
os.environ["OPENAI_API_KEY"] = api_key
base_url = "https://litellm.sph-prod.ethz.ch/v1"


# Load the dataset once at module level
file_path = "Data/fashion_data_2018_2022.xls"
df = pd.read_excel(file_path)
CHAT_HISTORY_FILE = "chat_history.html"


def save_chat_history(user_input: str, ai_html: str):
    """Appends a conversation entry to the chat history HTML file."""
    from datetime import datetime

    if not os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "w") as f:
            f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Chat History</title>
    <style>
        body { font-family: Georgia, serif; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #333; }
        h1 { text-align: center; }
        .entry { border-bottom: 2px solid #eee; padding: 20px 0; }
        .timestamp { color: #999; font-size: 0.85em; }
        .user-query { background: #f0f4ff; padding: 10px 15px; border-radius: 8px; margin: 10px 0; }
        .ai-response { margin-top: 10px; }
    </style>
</head>
<body>
<h1>Chat History</h1>
""")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(CHAT_HISTORY_FILE, "a") as f:
        f.write(f'<div class="entry">\n')
        f.write(f'  <p class="timestamp">{timestamp}</p>\n')
        f.write(f'  <div class="user-query"><strong>User:</strong> {user_input}</div>\n')
        f.write(f'  <div class="ai-response">{ai_html}</div>\n')
        f.write(f'</div>\n')


### Tools
@tool
def get_fashion_inspiration() -> str:
    """
    Retrieves fashion inspiration and trends from historical fashion data (2018–2022).
    Use this tool to generate fashion proposals grounded in real trend data.
    """
    return df.to_string()

@tool
def get_image_model(user_query: str, style: str = "") -> str:
    """
    Generates an image from a text prompt with a given style.
    Returns the URL or base64 of the generated image.
    """
    client = openai.OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=base_url,
    )
    prompt = f"Generate {style} for a fashion article about: {user_query}"
    image_response = client.images.generate(
        model="azure/dall-e-3",
        prompt=prompt,
        size="1024x1024",
        n=1,
    )

    image_data = image_response.data[0]
    return image_data.url if image_data.url else image_data.b64_json

@tool
def get_article_model(user_query: str) -> str:
    """
    Generates a fashion article from a text prompt using an LLM.
    Returns the generated article text.
    """
    prompt = f"Write a fashion article about: {user_query}"
    client = openai.OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=base_url,
    )
    response = client.chat.completions.create(
        model="anthropic/claude-sonnet-4-5",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response.choices[0].message.content

@tool
def chat_history_tool() -> str:
    """
    Retrieves the chat history.
    Use this tool to get the chat history.
    """
    if not os.path.exists(CHAT_HISTORY_FILE):
        return "No previous conversations."
    with open(CHAT_HISTORY_FILE, "r") as f:
        return f.read()




def build_tool_prompt(title_query: str, article_query: str, tool_result: str, tool_chat_history: str, tool_article: str, image_paths: list[str] = None) -> str:
    """Builds a prompt that incorporates tool output into the LLM request."""
    if not image_paths:
        image_paths = ["image_1.png"]

    images_list = "\n".join(f"  - {p}" for p in image_paths)

    return f"""You are a fashion expert, editor, and web designer.
You are given a draft article, trend data, and {len(image_paths)} generated image(s).

Your job: produce a COMPLETE, beautiful, standalone HTML page that combines the article and ALL the images.
Place the images in the most visually appealing positions — hero banner, inline between sections, side by side, as a gallery, sidebar, or any creative layout you think works best for a fashion magazine style page.
Use ALL the images provided, each one at least once.

ARTICLE TITLE:
{title_query}

AVAILABLE IMAGES (relative paths):
{images_list}

RULES:
- Return a COMPLETE HTML document (<!DOCTYPE html> through </html>)
- Include embedded CSS in a <style> tag — make it modern, elegant, magazine-like
- Use "{title_query}" as the main title (<h1>) of the page
- Use the image paths listed above in <img> src attributes
- Make the page responsive and visually stunning
- Do NOT use any external resources (CDN, fonts URLs, etc.) — everything must be self-contained
- Return ONLY the HTML code, no markdown fences, no explanation

DRAFT ARTICLE:
{tool_article}

ARTICLE TOPIC:
{article_query}

TREND DATA:
{tool_result}

CHAT HISTORY:
{tool_chat_history}"""


def llm_chat(prompt: str) -> str:
    """
    Chat with the LLM.
    Use this function to chat with the LLM.
    """
    client = openai.OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],  
    base_url=base_url,
    )
    completion = client.chat.completions.create(
        model="anthropic/claude-sonnet-4-5",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
    return completion.choices[0].message.content


OUTPUT_DIR = "output"


def validate_html(html: str, image_paths: list[str]) -> tuple[bool, list[str]]:
    """
    Validates the generated HTML.
    Returns (is_valid, list_of_errors).
    """
    from html.parser import HTMLParser

    errors = []

    # Strip markdown fences if the LLM wrapped the output
    stripped = html.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        lines = lines[1:]  # remove opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        html = "\n".join(lines)

    if "<!DOCTYPE html>" not in html and "<html" not in html:
        errors.append("missing <!DOCTYPE html> or <html> tag")

    if "</html>" not in html:
        errors.append("missing closing </html> tag")

    if "<body" not in html:
        errors.append("missing <body> tag")

    if "<style" not in html:
        errors.append("missing <style> tag (no CSS)")

    for img in image_paths:
        if img not in html:
            errors.append(f"image '{img}' not referenced in HTML")

    # Check that HTML is parseable
    class HTMLValidator(HTMLParser):
        def __init__(self):
            super().__init__()
            self.parse_error = False
        def handle_starttag(self, tag, attrs):
            pass
        def handle_endtag(self, tag):
            pass

    try:
        validator = HTMLValidator()
        validator.feed(html)
    except Exception:
        errors.append("HTML parsing error")

    return (len(errors) == 0, errors)


def save_image(image_data: str, path: str):
    """Save an image from a URL or base64 string to disk."""
    if image_data.startswith("http"):
        resp = requests.get(image_data, timeout=30)
        resp.raise_for_status()
        with open(path, "wb") as f:
            f.write(resp.content)
    else:
        img_bytes = base64.b64decode(image_data)
        with open(path, "wb") as f:
            f.write(img_bytes)


def llm_chat_with_tools(title_query: str, image_styles: list[str], article_query: str):
    """
    Enhanced llm_chat that uses the fashion inspiration tool before calling the LLM.

    Args:
        title_query: The article title / general topic.
        image_styles: List of style descriptions for each image to generate.
        article_query: What the article should be about in detail.
    """
    num_images = len(image_styles)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_steps = 2 + num_images + 1  # trend + history + N images + article

    step = 1
    print(f"[{step}/{total_steps}] Fetching fashion trend data...")
    tool_result = get_fashion_inspiration.invoke({})
    with open(os.path.join(OUTPUT_DIR, "fashion_data.txt"), "w") as f:
        f.write(tool_result)

    step += 1
    print(f"[{step}/{total_steps}] Fetching chat history...")
    tool_chat_history = chat_history_tool.invoke({})

    image_paths = []
    for i in range(1, num_images + 1):
        step += 1
        style = image_styles[i - 1]
        print(f"[{step}/{total_steps}] Generating image {i}/{num_images} ({style})...")
        tool_image = get_image_model.invoke({"user_query": title_query, "style": style})
        filename = f"image_{i}.png"
        save_image(tool_image, os.path.join(OUTPUT_DIR, filename))
        image_paths.append(filename)

    step += 1
    print(f"[{step}/{total_steps}] Generating article...")
    tool_article = get_article_model.invoke(article_query)
    with open(os.path.join(OUTPUT_DIR, "article.txt"), "w") as f:
        f.write(tool_article)

    # LLM builds the full HTML page placing all images (with retry on bad output)
    enriched_prompt = build_tool_prompt(title_query, article_query, tool_result, tool_chat_history, tool_article, image_paths=image_paths)

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        print(f"Generating final HTML page (attempt {attempt}/{max_retries})...")
        final_html = llm_chat(enriched_prompt)
        is_valid, errors = validate_html(final_html, image_paths)
        if is_valid:
            print("HTML validation passed.")
            break
        print(f"HTML validation failed: {', '.join(errors)}")
        if attempt < max_retries:
            print("Retrying...")
    else:
        print("Warning: max retries reached, saving best attempt.")

    html_path = os.path.join(OUTPUT_DIR, "final_article.html")
    with open(html_path, "w") as f:
        f.write(final_html)

    save_chat_history(title_query, final_html)

    images_list = "\n".join(f"  - {p}" for p in image_paths)
    print(f"\nAll outputs saved to '{OUTPUT_DIR}/' folder:")
    print(f"  - fashion_data.txt     (trend data)")
    print(f"  - chat_history.html    (conversation history)")
    print(f"{images_list}")
    print(f"  - article.txt          (generated article draft)")
    print(f"  - final_article.html   (final page with {num_images} image(s) + article)")
    print(f"\nOpen final_article.html in your browser to see the result!")

    return final_html


def clear_history():
    """Delete chat_history.html."""
    if os.path.exists(CHAT_HISTORY_FILE):
        os.remove(CHAT_HISTORY_FILE)
        print("Chat history deleted.")
    else:
        print("No chat history found.")


if __name__ == "__main__":
    print("=== Fashion Article Generator ===")
    print("  1. Generate a new article")
    print("  2. Delete chat history")
    print()

    choice = input("Choose an option (1-2): ").strip()

    if choice == "1":
        title_query = input("What is the title of the article? ")
        num_images = int(input("How many images do you want to generate? "))
        image_styles = []
        for i in range(1, num_images + 1):
            style = input(f"  Describe the style for image {i}: ")
            image_styles.append(style)
        article_query = input("What should the article be about? ")
        result = llm_chat_with_tools(title_query, image_styles, article_query)
        print("\n--- Final Result ---")
        print(result)
    elif choice == "2":
        clear_history()
    else:
        print("Invalid option.")