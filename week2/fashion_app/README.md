# Fashion Magazine Generator — Flask App

## Setup

### 1. Create your `.env` file
In the `fashion_app/` folder create a file called `.env`:
```
OPENAI_API_KEY=your_litellm_key_here
```

### 2. Create virtual environment & install dependencies
We have provided you with two scripts one ending in ".sh" for macOS and one ending in ".ps1" for Windows. These scripts ensure you have python installed and create a virtual enviorment with all the necesarry libraries installed.

**macOS:**
Make the script executable.
```bash
chmod +x setup.sh
```

Run the setup
```bash
./setup.sh
```

**Windows:**
This script is made to be run in a powershell terminal.
Make the script executable.
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
Run the setup
```powershell
.\setup.ps1
```

### 3. Activate virtual environment

**macOS:**
```bash
source venv/bin/activate
```

**Windows:**
```powershell
.\venv\Scripts\activate
```

You can tell that your terminal is in the virtual environment by seeing a "(venv)" in front of the current file in the terminal. It is essential to have your virtual environment active otherwise the app will not be able to find the necessary libraries. 

### 4. Run the app
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