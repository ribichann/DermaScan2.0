"""Flask web app for DermaScan AI."""
from __future__ import annotations

import os
import secrets
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, abort, flash, redirect, render_template, request, session, url_for
from werkzeug.utils import secure_filename

from advisor import analyze_ingredients, build_routine, discover_products
from ocr_engine import extract_text

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
# Keep the product catalog server-side; Flask cookie sessions have a small size limit.
CATALOG_CACHE: dict[str, list[dict]] = {}

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY") or secrets.token_hex(32)
app.config.update(
    UPLOAD_FOLDER=str(UPLOAD_DIR),
    MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def current_profile() -> tuple[str, list[str]]:
    skin_type = session.get("skin_type", "Oily")
    concerns = session.get("concerns", ["Excess Oil"])
    if not isinstance(concerns, list):
        concerns = []
    return skin_type, concerns


@app.after_request
def security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; img-src 'self' data:")
    return response


@app.errorhandler(413)
def too_large(_error):
    flash("The image is too large. Please upload an image under 8 MB.", "error")
    return redirect(url_for("index"))


@app.route("/", methods=["GET"])
def index():
    skin_type, concerns = current_profile()
    return render_template("profile.html", skin_type=skin_type, concerns=concerns)


@app.route("/profile", methods=["POST"])
def profile():
    skin_type = request.form.get("skin_type", "Oily").strip()
    allowed_skin_types = {"Oily", "Dry", "Combination", "Sensitive", "Normal"}
    if skin_type not in allowed_skin_types:
        skin_type = "Oily"
    allowed_concerns = {"Acne", "Excess Oil", "Hyperpigmentation", "Barrier Repair", "Anti-Aging"}
    concerns = [c for c in request.form.getlist("concerns") if c in allowed_concerns]
    session["skin_type"] = skin_type
    session["concerns"] = concerns
    destination = request.form.get("destination", "scan")
    if destination == "discover":
        return redirect(url_for("discover"))
    return redirect(url_for("scan_page"))


@app.route("/scan", methods=["GET", "POST"])
def scan_page():
    skin_type, concerns = current_profile()
    if request.method == "GET":
        return render_template("scan.html", skin_type=skin_type, concerns=concerns)
    uploaded = request.files.get("label_image")
    if not uploaded or not uploaded.filename:
        flash("Choose an ingredient-label image first.", "error")
        return redirect(url_for("scan_page"))
    if not allowed_file(uploaded.filename):
        flash("Use a PNG, JPG, JPEG, or WEBP image.", "error")
        return redirect(url_for("scan_page"))
    safe_name = secure_filename(uploaded.filename)
    extension = safe_name.rsplit(".", 1)[1].lower()
    filename = f"{secrets.token_hex(12)}.{extension}"
    target = UPLOAD_DIR / filename
    uploaded.save(target)
    try:
        ocr_text = extract_text(str(target))
        result = analyze_ingredients(ocr_text, skin_type, concerns)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        target.unlink(missing_ok=True)
        flash(str(exc), "error")
        return redirect(url_for("scan_page"))
    except Exception:
        target.unlink(missing_ok=True)
        app.logger.exception("Label scan failed")
        flash("The scan could not be completed. Check that Tesseract OCR is installed and try again.", "error")
        return redirect(url_for("scan_page"))
    session["last_scan"] = result
    session["last_scan_filename"] = filename
    return render_template("result.html", result=result, skin_type=skin_type, concerns=concerns)


@app.route("/discover", methods=["GET"])
def discover():
    skin_type, concerns = current_profile()
    products = discover_products(skin_type, concerns)
    catalog_key = session.get("catalog_key")
    if not catalog_key:
        catalog_key = secrets.token_urlsafe(18)
        session["catalog_key"] = catalog_key
    CATALOG_CACHE[catalog_key] = products
    return render_template("discover.html", products=products, skin_type=skin_type, concerns=concerns)


@app.route("/routine", methods=["GET", "POST"])
def routine():
    skin_type, _concerns = current_profile()
    catalog_key = session.get("catalog_key")
    products = CATALOG_CACHE.get(catalog_key) if catalog_key else None
    if not isinstance(products, list) or not products:
        products = discover_products(skin_type, _concerns)
        catalog_key = secrets.token_urlsafe(18)
        session["catalog_key"] = catalog_key
        CATALOG_CACHE[catalog_key] = products
    if request.method == "POST":
        try:
            budget = int(request.form.get("budget", "1500"))
        except (TypeError, ValueError):
            budget = 1500
        budget = max(200, min(budget, 50000))
    else:
        budget = 1500
    result = build_routine(products, budget, skin_type)
    return render_template("routine.html", routine=result, budget=budget)


if __name__ == "__main__":
    # For production, run behind a WSGI server and set FLASK_SECRET_KEY in the environment.
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")), debug=False)
