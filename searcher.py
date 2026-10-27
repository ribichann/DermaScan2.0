"""SearXNG search client with sanitized results and an offline Indian-market fallback."""
from __future__ import annotations

import html
import re
from typing import Any
from urllib.parse import quote_plus, urlparse

import requests

SEARXNG_URL = "http://localhost:8080/search"
USER_AGENT = "DermaScanAI/1.0 (+local skincare research app)"
TIMEOUT_SECONDS = 7

MOCK_RESULTS = [
    {
        "title": "Minimalist 2% Salicylic Acid Cleanser",
        "snippet": "A salicylic-acid cleanser commonly listed for oily and acne-prone skin. Check the current label and price.",
        "url": "https://www.google.com/search?q=Minimalist+2%25+Salicylic+Acid+Cleanser",
    },
    {
        "title": "Cetaphil Gentle Skin Cleanser",
        "snippet": "A gentle cleanser option for dry or sensitive-feeling skin. Verify the current ingredient list.",
        "url": "https://www.google.com/search?q=Cetaphil+Gentle+Skin+Cleanser+India",
    },
    {
        "title": "The Ordinary Glycolic Acid 7% Exfoliating Toner",
        "snippet": "An exfoliating toner; can irritate sensitive skin and should not be layered casually with retinoids.",
        "url": "https://www.google.com/search?q=The+Ordinary+Glycolic+Acid+7%25+Toner+India",
    },
    {
        "title": "Minimalist 5% Niacinamide Serum",
        "snippet": "Niacinamide serum option often marketed for uneven tone and oil balance. Verify the exact formulation.",
        "url": "https://www.google.com/search?q=Minimalist+5%25+Niacinamide+Serum",
    },
    {
        "title": "Deconstruct 10% Vitamin C Serum",
        "snippet": "Vitamin C serum marketed for dullness and uneven tone. Patch test and check current packaging.",
        "url": "https://www.google.com/search?q=Deconstruct+Vitamin+C+Serum+India",
    },
    {
        "title": "CeraVe Moisturising Cream",
        "snippet": "A ceramide-containing moisturizer option. May feel rich for some oily skin types.",
        "url": "https://www.google.com/search?q=CeraVe+Moisturising+Cream+India",
    },
    {
        "title": "Pond's Super Light Gel",
        "snippet": "A lightweight gel moisturizer commonly available in India; check fragrance and individual tolerance.",
        "url": "https://www.google.com/search?q=Ponds+Super+Light+Gel+India",
    },
    {
        "title": "Re'equil Ultra Matte Dry Touch Sunscreen SPF 50",
        "snippet": "A matte-finish sunscreen option. Confirm current SPF, UVA claims, and directions on the product label.",
        "url": "https://www.google.com/search?q=Reequil+Ultra+Matte+Dry+Touch+Sunscreen+SPF+50",
    },
    {
        "title": "La Roche-Posay Anthelios SPF 50+",
        "snippet": "A higher-priced sunscreen line; local availability and exact variant can differ.",
        "url": "https://www.google.com/search?q=La+Roche+Posay+Anthelios+SPF+50+India",
    },
    {
        "title": "COSRX Advanced Snail 96 Mucin Essence",
        "snippet": "Hydrating essence; not a required step and should be judged by the full ingredient list and tolerance.",
        "url": "https://www.google.com/search?q=COSRX+Advanced+Snail+96+Mucin+Essence+India",
    },
    {
        "title": "Bioderma Atoderm Intensive Baume",
        "snippet": "A richer barrier-support product often used for dry skin; may feel heavy for some oily skin types.",
        "url": "https://www.google.com/search?q=Bioderma+Atoderm+Intensive+Baume+India",
    },
    {
        "title": "The Ordinary Retinol 0.2% in Squalane",
        "snippet": "A retinoid product that can irritate; avoid during pregnancy and do not begin without understanding precautions.",
        "url": "https://www.google.com/search?q=The+Ordinary+Retinol+0.2%25+Squalane+India",
    },
]


def _clean(value: Any, limit: int = 500) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _safe_url(value: Any) -> str:
    url = _clean(value, 1000)
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return url


def search(query: str, limit: int = 8) -> list[dict[str, str]]:
    """Search the configured SearXNG JSON endpoint. Falls back to local mock results."""
    query = _clean(query, 300)
    if not query:
        return []
    try:
        response = requests.get(
            SEARXNG_URL,
            params={"q": query, "format": "json", "language": "en-IN", "safesearch": 1},
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        results = []
        for item in payload.get("results", []):
            title = _clean(item.get("title"), 180)
            snippet = _clean(item.get("content") or item.get("snippet"), 500)
            url = _safe_url(item.get("url"))
            if title and url:
                results.append({"title": title, "snippet": snippet, "url": url})
            if len(results) >= max(1, min(limit, 20)):
                break
        if results:
            return results
    except (requests.RequestException, ValueError, TypeError):
        pass

    # Offline fallback: relevant mock results first, then remaining products.
    terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    ranked = []
    for item in MOCK_RESULTS:
        haystack = (item["title"] + " " + item["snippet"]).lower()
        score = sum(1 for term in terms if len(term) > 2 and term in haystack)
        ranked.append((score, item))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    selected = [dict(item) for score, item in ranked if score > 0][:limit]
    if not selected:
        selected = [dict(item) for item in MOCK_RESULTS[:limit]]
    return selected


def search_evidence(ingredient: str) -> list[dict[str, str]]:
    """Search for ingredient-specific clinical/safety context; results are leads, not proof."""
    ingredient = _clean(ingredient, 100)
    return search(f"{ingredient} topical skincare safety irritation comedogenic evidence", limit=5)
