# Fashion Magazine Generator — Flask App

## Setup

### 1. Create your `.env` file
In the `fashion_app/` folder create a file called `.env`:
```
OPENAI_API_KEY=your_litellm_key_here
```

### 2. Create virtual environment & install dependencies

**macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the app
```bash
python3 app.py     # macOS
python app.py      # Windows
```

Open http://127.0.0.1:5001

---

## How It Works

1. Fill in **Article Title**, **Image Styles**, and **Article Description**
2. Press **▶ Generate** — live progress streams in the sidebar
3. Article renders in the preview panel on the right
4. Press **Generate** again to refine — the last 5 sessions feed into the next generation for continuity
5. Press **🗑 Delete History** to start completely fresh

---

## Project Structure

```
fashion_app/
├── app.py                  ← Flask app + full AI pipeline
├── templates/
│   ├── fashion_magazine.html ← Main UI
│   ├── article.html        ← Article wrapper
│   └── error.html          ← Error page
├── static/                 ← Static assets
├── Data/                   ← Place your .xls dataset here
├── output/                 ← Generated articles + images (auto-created)
├── history/                ← Session logs (auto-created)
├── requirements.txt
├── .env                    ← Your API key (never commit this!)
└── .gitignore
```

---

## Key Design Decisions

- **Separate history logs** — `articles.json`, `images.json`, `text.json` tracked independently
- **Last 5 sessions** injected into each prompt for continuity (kept at 5 to avoid token limits)
- **3 retries + emergency fallback** — if HTML generation fails, a basic valid page is assembled from raw parts
- **SSE streaming** — live progress updates streamed to browser during generation
- **LangChain removed** — replaced with plain Python functions since the pipeline is fixed and predetermined