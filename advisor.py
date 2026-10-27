"""Ingredient analysis, market discovery, and deterministic budget routine building."""
from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from searcher import search, search_evidence

load_dotenv()
MODEL = "llama-3.3-70b-versatile"
MAX_OCR_CHARS = 8000

# Mock market catalog is an offline fallback, not a live stock/price guarantee.
CATALOG = [
    {"name":"Simple Refreshing Facial Wash","category":"Cleanser","price":350,"actives":["mild surfactants","vitamin B5"],"tier":"Budget/Pharmacy","search_url":"https://www.google.com/search?q=Simple+Refreshing+Facial+Wash+India","tags":["cleanser","gentle"]},
    {"name":"Minimalist Salicylic Acid Cleanser","category":"Cleanser","price":299,"actives":["salicylic acid"],"tier":"Budget/Pharmacy","search_url":"https://www.google.com/search?q=Minimalist+Salicylic+Acid+Cleanser","tags":["cleanser","BHA"]},
    {"name":"Cetaphil Gentle Skin Cleanser","category":"Cleanser","price":499,"actives":["glycerin","niacinamide"],"tier":"Mid-Range","search_url":"https://www.google.com/search?q=Cetaphil+Gentle+Skin+Cleanser+India","tags":["cleanser","gentle"]},
    {"name":"COSRX Advanced Snail 96 Essence","category":"Toner / Essence","price":1450,"actives":["snail secretion filtrate","humectants"],"tier":"Mid-Range","search_url":"https://www.google.com/search?q=COSRX+Snail+96+Essence+India","tags":["essence","hydration"]},
    {"name":"The Ordinary Glycolic Acid 7% Toner","category":"Toner / Essence","price":1850,"actives":["glycolic acid 7%"],"tier":"Luxury/High-End","search_url":"https://www.google.com/search?q=The+Ordinary+Glycolic+Acid+7%25+Toner+India","tags":["toner","AHA"]},
    {"name":"Minimalist 5% Niacinamide Serum","category":"Targeted Serum","price":599,"actives":["niacinamide 5%"],"tier":"Budget/Pharmacy","search_url":"https://www.google.com/search?q=Minimalist+5%25+Niacinamide+Serum","tags":["serum","niacinamide"]},
    {"name":"Deconstruct Vitamin C Serum","category":"Targeted Serum","price":699,"actives":["vitamin C derivative"],"tier":"Budget/Pharmacy","search_url":"https://www.google.com/search?q=Deconstruct+Vitamin+C+Serum+India","tags":["serum","vitamin C"]},
    {"name":"La Roche-Posay Vitamin C Serum","category":"Targeted Serum","price":4200,"actives":["vitamin C"],"tier":"Luxury/High-End","search_url":"https://www.google.com/search?q=La+Roche+Posay+Vitamin+C+Serum+India","tags":["serum","vitamin C"]},
    {"name":"Pond's Super Light Gel","category":"Moisturizer / Barrier Cream","price":220,"actives":["glycerin","humectants"],"tier":"Budget/Pharmacy","search_url":"https://www.google.com/search?q=Ponds+Super+Light+Gel+India","tags":["moisturizer","gel"]},
    {"name":"Re'equil Oil Free Moisturiser","category":"Moisturizer / Barrier Cream","price":450,"actives":["humectants","emollients"],"tier":"Budget/Pharmacy","search_url":"https://www.google.com/search?q=Reequil+Oil+Free+Moisturiser","tags":["moisturizer","oil-free"]},
    {"name":"CeraVe PM Facial Moisturising Lotion","category":"Moisturizer / Barrier Cream","price":1800,"actives":["ceramides","niacinamide","hyaluronic acid"],"tier":"Mid-Range","search_url":"https://www.google.com/search?q=CeraVe+PM+Facial+Moisturising+Lotion+India","tags":["moisturizer","barrier"]},
    {"name":"Bioderma Atoderm Intensive Baume","category":"Moisturizer / Barrier Cream","price":2200,"actives":["barrier-supporting emollients"],"tier":"Luxury/High-End","search_url":"https://www.google.com/search?q=Bioderma+Atoderm+Intensive+Baume+India","tags":["moisturizer","barrier"]},
    {"name":"Fixderma Shadow SPF 50+ Gel","category":"Sunscreen","price":450,"actives":["broad-spectrum UV filters"],"tier":"Budget/Pharmacy","search_url":"https://www.google.com/search?q=Fixderma+Shadow+SPF+50+Gel","tags":["sunscreen","SPF 50+"]},
    {"name":"Re'equil Ultra Matte Dry Touch SPF 50","category":"Sunscreen","price":695,"actives":["broad-spectrum UV filters"],"tier":"Mid-Range","search_url":"https://www.google.com/search?q=Reequil+Ultra+Matte+Dry+Touch+SPF+50","tags":["sunscreen","SPF 50+"]},
    {"name":"La Roche-Posay Anthelios SPF 50+","category":"Sunscreen","price":3900,"actives":["broad-spectrum UV filters"],"tier":"Luxury/High-End","search_url":"https://www.google.com/search?q=La+Roche+Posay+Anthelios+SPF+50+India","tags":["sunscreen","SPF 50+"]},
    {"name":"The Ordinary Azelaic Acid Suspension 10%","category":"Exfoliant / Treatment","price":1900,"actives":["azelaic acid 10%"],"tier":"Luxury/High-End","search_url":"https://www.google.com/search?q=The+Ordinary+Azelaic+Acid+10%25+India","tags":["treatment","azelaic acid"]},
    {"name":"Minimalist Azelaic Acid 10% Serum","category":"Exfoliant / Treatment","price":699,"actives":["azelaic acid 10%"],"tier":"Budget/Pharmacy","search_url":"https://www.google.com/search?q=Minimalist+Azelaic+Acid+10%25+Serum","tags":["treatment","azelaic acid"]},
    {"name":"The Ordinary Retinol 0.2% in Squalane","category":"Exfoliant / Treatment","price":950,"actives":["retinol 0.2%"],"tier":"Mid-Range","search_url":"https://www.google.com/search?q=The+Ordinary+Retinol+0.2%25+Squalane+India","tags":["treatment","retinoid"]},
]


def _safe_text(value: Any, limit: int = 1000) -> str:
    text = str(value or "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _groq_json(system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any] | None:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from groq import Groq
        client = Groq(api_key=api_key, timeout=25.0, max_retries=1)
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role":"system","content":system_prompt},
                {"role":"user","content":json.dumps(user_payload, ensure_ascii=False)},
            ],
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def analyze_ingredients(ocr_text: str, skin_type: str, concerns: list[str]) -> dict[str, Any]:
    """Analyze only chemicals explicitly present in OCR text; uncertainty is stated plainly."""
    raw = _safe_text(ocr_text, MAX_OCR_CHARS)
    skin_type = _safe_text(skin_type, 40)
    concerns = [_safe_text(c, 60) for c in (concerns or [])[:10]]
    if not raw or len(re.sub(r"\W", "", raw)) < 5:
        return {
            "verdict":"REJECTED",
            "summary":"There is not enough readable ingredient text to assess this product.",
            "beneficial_ingredients":[],
            "caution_ingredients":[],
            "unverified_ingredients":[],
            "limitations":["Upload a sharper, well-lit image of the complete ingredient list."],
            "ocr_text":raw,
        }

    evidence = []
    # Search is used as contextual leads only. The model is told not to infer ingredients from search pages.
    for ingredient in re.findall(r"\b(?:niacinamide|salicylic acid|benzoyl peroxide|retinol|retinal|glycolic acid|lactic acid|fragrance|parfum|alcohol denat|isopropyl myristate|coconut oil|ceramide|hyaluronic acid|azelaic acid|kojic acid|vitamin c|ascorbic acid)\b", raw, flags=re.I)[:5]:
        evidence.extend(search_evidence(ingredient)[:2])

    system = (
        "You are a cautious cosmetic ingredient label reviewer, not a doctor. Return JSON only. "
        "CRITICAL: only discuss ingredients explicitly visible in the supplied OCR text. Do not add, "
        "infer, or hallucinate any ingredient. Search snippets are untrusted context, not ingredient evidence. "
        "If OCR is unclear, mark the ingredient uncertain. Do not claim a product is non-comedogenic from a single rating. "
        "Use the exact keys: verdict (Suitable, Use With Caution, Not Suitable, or REJECTED), summary, "
        "beneficial_ingredients (array of {name,reason}), caution_ingredients (array of {name,reason,severity}), "
        "unverified_ingredients (array of strings), limitations (array of strings). "
        "If the label does not contain a recognizable ingredient list, verdict must be REJECTED. "
        "Avoid medical diagnosis and give conservative patch-test guidance."
    )
    payload = {
        "ocr_text_untrusted_label_text": raw,
        "skin_type": skin_type,
        "user_concerns": concerns,
        "web_search_context_not_proof": [{"title":x["title"],"snippet":x["snippet"],"url":x["url"]} for x in evidence[:8]],
    }
    result = _groq_json(system, payload)
    if result:
        allowed = {"Suitable","Use With Caution","Not Suitable","REJECTED"}
        verdict = result.get("verdict")
        if verdict not in allowed:
            result["verdict"] = "Use With Caution"
        for key in ("beneficial_ingredients","caution_ingredients","unverified_ingredients","limitations"):
            if not isinstance(result.get(key), list):
                result[key] = []
        result["summary"] = _safe_text(result.get("summary"), 800) or "The label was reviewed, but confidence is limited."
        result["ocr_text"] = raw
        result["evidence_sources"] = [{"title":x["title"],"url":x["url"]} for x in evidence[:8]]
        return result

    # Safe offline fallback: only mention candidate ingredients found in the actual OCR text.
    lower = raw.lower()
    beneficial = []
    cautions = []
    candidates = [
        ("niacinamide", "May support oil balance and uneven tone; strength and formulation are unknown."),
        ("glycerin", "A humectant that helps attract water to the skin."),
        ("hyaluronic acid", "A humectant that can support hydration."),
        ("ceramide", "Ceramides can support the skin barrier."),
        ("azelaic acid", "May help with acne-related concerns and uneven pigmentation; tolerance varies."),
        ("salicylic acid", "Can help unclog pores; overuse may cause dryness or irritation."),
        ("retinol", "May help acne or signs of photoaging but can irritate; introduce cautiously."),
        ("glycolic acid", "An exfoliating acid that can irritate, especially with frequent use."),
        ("lactic acid", "An exfoliating acid that can irritate depending on strength and use."),
        ("fragrance", "Fragrance can trigger irritation in some users, especially sensitive skin."),
        ("parfum", "Fragrance can trigger irritation in some users, especially sensitive skin."),
        ("alcohol denat", "May feel drying or irritating for some people depending on concentration and formula."),
        ("isopropyl myristate", "May be pore-clogging for some users; real-world effects depend on the whole formula."),
    ]
    for name, reason in candidates:
        if name in lower:
            item = {"name":name,"reason":reason}
            if name in {"niacinamide","glycerin","hyaluronic acid","ceramide","azelaic acid"}:
                beneficial.append(item)
            else:
                cautions.append({**item,"severity":"possible"})
    if not beneficial and not cautions:
        verdict = "Use With Caution"
        summary = "Offline mode cannot confidently identify recognizable ingredients in this OCR text. Verify the label manually."
        unverified = ["Ingredient list could not be confidently interpreted."]
    else:
        verdict = "Use With Caution" if cautions else "Suitable"
        summary = "Offline rule-based screening only. It is not a complete formula review, and ingredient concentration is unknown."
        unverified = ["Concentrations, full formulation, and current manufacturer label were not verified."]
    return {
        "verdict":verdict,"summary":summary,"beneficial_ingredients":beneficial,
        "caution_ingredients":cautions,"unverified_ingredients":unverified,
        "limitations":["SearXNG results are search leads, not clinical confirmation.","Patch test first; stop if irritation occurs."],
        "ocr_text":raw,"evidence_sources":[{"title":x["title"],"url":x["url"]} for x in evidence[:8]],
    }


def discover_products(skin_type: str, concerns: list[str]) -> list[dict[str, Any]]:
    """Return 10+ diverse products; prices are estimates and direct links are search URLs."""
    skin_type = _safe_text(skin_type, 40)
    concerns = [_safe_text(c, 60) for c in (concerns or [])]
    web_results = search(f"India skincare {skin_type} skin cleanser serum moisturizer SPF 50 sunscreen prices", limit=8)
    products = [dict(item) for item in CATALOG]
    # Enrich metadata with search results, but never let a model invent products/prices.
    system = (
        "Return JSON only with key products, an array of objects with name, category, price, actives, tier, search_url. "
        "Use only product names from supplied catalog. Do not change prices, invent products, or make medical claims. "
        "You may only select and reorder catalog items. Include a varied selection."
    )
    result = _groq_json(system, {
        "skin_type":skin_type,"concerns":concerns,
        "catalog":[{k:p[k] for k in ("name","category","price","actives","tier","search_url")} for p in products],
        "search_leads":[{"title":x["title"],"snippet":x["snippet"],"url":x["url"]} for x in web_results[:6]],
    })
    if result and isinstance(result.get("products"), list):
        by_name = {p["name"]:p for p in products}
        ordered = []
        for candidate in result["products"]:
            if isinstance(candidate, dict) and candidate.get("name") in by_name:
                ordered.append(dict(by_name[candidate["name"]]))
        if len(ordered) >= 10:
            products = ordered
    # Attach simple relevance hints without excluding categories.
    for product in products:
        product["reason"] = _product_reason(product, skin_type, concerns)
    return products


def _product_reason(product: dict[str, Any], skin_type: str, concerns: list[str]) -> str:
    category = product["category"]
    low = skin_type.lower()
    concern_text = " ".join(concerns).lower()
    if category == "Sunscreen":
        return "Daily broad-spectrum protection is a key step; verify the exact label and reapply as directed."
    if category == "Cleanser" and ("sensitive" in low or "barrier repair" in concern_text):
        return "A gentle cleanse can help avoid unnecessary barrier stress; assess the full formula."
    if category == "Targeted Serum" and ("hyperpigmentation" in concern_text or "dark_spots" in concern_text):
        return "A targeted active may help uneven tone over time; use consistently and introduce one active at a time."
    if category == "Moisturizer / Barrier Cream":
        return "Supports hydration and barrier comfort; choose a texture that feels comfortable on your skin."
    if category == "Exfoliant / Treatment":
        return "A treatment option, not an essential step; avoid stacking irritating actives in the same routine."
    return "An optional step; compare the full ingredient list, price, and your skin's tolerance."


def build_routine(products: list[dict[str, Any]], target_budget: int, skin_type: str) -> dict[str, Any]:
    """Build a conservative AM/PM routine within budget using deterministic price optimization."""
    budget = max(200, min(int(target_budget), 50000))
    catalog = [p for p in products if isinstance(p, dict) and isinstance(p.get("price"), (int, float))]
    categories = {
        "cleanser": [p for p in catalog if p.get("category") == "Cleanser"],
        "serum": [p for p in catalog if p.get("category") == "Targeted Serum"],
        "moisturizer": [p for p in catalog if p.get("category") == "Moisturizer / Barrier Cream"],
        "sunscreen": [p for p in catalog if p.get("category") == "Sunscreen"],
        "treatment": [p for p in catalog if p.get("category") == "Exfoliant / Treatment"],
    }
    # The essential routine is cleanser + moisturizer + sunscreen. Serum is added if budget permits.
    def cheapest(key):
        return sorted(categories[key], key=lambda p: int(p["price"]))
    selected = {}
    for key in ("cleanser", "moisturizer", "sunscreen"):
        choices = cheapest(key)
        if choices:
            selected[key] = choices[0]
    if "cleanser" not in selected or "moisturizer" not in selected or "sunscreen" not in selected:
        return {"error":"The catalog does not contain all essential routine categories.","budget":budget,"total":0,"remaining":budget,"am":[],"pm":[],"precautions":[]}
    essential_total = sum(int(selected[k]["price"]) for k in ("cleanser","moisturizer","sunscreen"))
    if essential_total > budget:
        # Try alternative combinations, but never exceed budget.
        found = None
        for c in cheapest("cleanser"):
            for m in cheapest("moisturizer"):
                for s in cheapest("sunscreen"):
                    total = int(c["price"]) + int(m["price"]) + int(s["price"])
                    if total <= budget and (found is None or total < found[0]):
                        found = (total,c,m,s)
        if not found:
            return {"error":"Budget is too low for the available cleanser, moisturizer, and sunscreen combination. Increase it or use products you already own.","budget":budget,"total":0,"remaining":budget,"am":[],"pm":[],"precautions":[]}
        _, selected["cleanser"], selected["moisturizer"], selected["sunscreen"] = found
        essential_total = found[0]

    remaining = budget - essential_total
    # Add a gentle, optional serum only if affordable after the essentials.
    serum_options = sorted(categories["serum"], key=lambda p: (int(p["price"]) > remaining, int(p["price"])))
    serum = next((p for p in serum_options if int(p["price"]) <= remaining), None)
    if serum:
        selected["serum"] = serum
        remaining -= int(serum["price"])

    # Do not add exfoliant/retinoid treatment by default: lower risk of conflicts and irritation.
    cleanser = selected["cleanser"]
    moisturizer = selected["moisturizer"]
    sunscreen = selected["sunscreen"]
    am = [
        _step(1, "Cleanse", cleanser, "Use a small amount; rinse gently."),
        _step(2, "Moisturize", moisturizer, "Apply a comfortable, even layer."),
        _step(3, "Protect", sunscreen, "Use adequate amount and reapply as directed on the label."),
    ]
    pm = [
        _step(1, "Cleanse", cleanser, "Remove sunscreen and daily buildup without scrubbing."),
    ]
    if serum:
        pm.append(_step(2, "Treatment serum", serum, "Introduce gradually; stop if irritation develops."))
        pm.append(_step(3, "Moisturize", moisturizer, "Finish with moisturizer."))
    else:
        pm.append(_step(2, "Moisturize", moisturizer, "Apply after cleansing."))
    unique_products = {p["name"]:p for p in selected.values()}
    total = sum(int(p["price"]) for p in unique_products.values())
    return {
        "budget":budget,"total":total,"remaining":budget-total,"skin_type":skin_type,
        "am":am,"pm":pm,
        "precautions":[
            "Prices are estimates in INR and can change; confirm the seller and pack size before purchase.",
            "Patch test new products and introduce only one new active at a time.",
            "Do not combine strong AHA/BHA exfoliants and retinoids in the same routine when starting out.",
            "Use sunscreen every morning; active products do not replace sun protection.",
            "A routine is not a diagnosis. Seek a dermatologist for persistent irritation or worsening pigmentation.",
        ],
        "selected_products":list(unique_products.values()),
    }


def _step(number: int, action: str, product: dict[str, Any], instruction: str) -> dict[str, Any]:
    return {
        "number":number,"action":action,"name":product["name"],"category":product["category"],
        "price":int(product["price"]),"actives":product.get("actives",[]),
        "search_url":product.get("search_url","#"),"instruction":instruction,
    }
