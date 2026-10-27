# DermaScan AI

A local Flask prototype for skincare ingredient-label OCR, cautious ingredient review, product discovery, and budget routine construction.

## Requirements
- Python 3.10+
- Tesseract OCR installed (Windows default path is configured in `ocr_engine.py`)
- Optional: local SearXNG at `http://localhost:8080`
- Optional: `GROQ_API_KEY` for GroqCloud model analysis

## Setup (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:GROQ_API_KEY="your-key-here"
python app.py
```
Open http://127.0.0.1:5000

Without SearXNG, the app uses clearly labeled mock product/search leads. Without Groq, ingredient analysis uses conservative local rules. Prices are illustrative estimates, not live quotes.

## Production notes
Use a production WSGI server, HTTPS, a persistent secret key, CSRF protection, rate limiting, upload scanning, and a cleanup policy for uploaded images before public deployment. Do not expose this development server directly to the internet.
