import os, json, re, hmac, hashlib
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from urllib.parse import parse_qsl
from typing import Dict, Any, List, Optional
from threading import Lock

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

TZ = ZoneInfo("Europe/Uzhgorod")
DB_PATH = "db.json"

BOT_TOKEN = os.environ.get("BOT_TOKEN")
AI_API_KEY = os.environ.get("AI_API_KEY")  # може бути None
AI_ENDPOINT = os.environ.get("AI_ENDPOINT", "https://models.github.ai/inference")
AI_MODEL = os.environ.get("AI_MODEL", "openai/gpt-4o-mini")

# BOT_TOKEN потрібен для Telegram initData. Але для DEMO в браузері можна запуститись і без нього.
# Якщо хочеш суворо — залиш RuntimeError тільки для прод, але зараз так буде стабільніше.
if not BOT_TOKEN:
    BOT_TOKEN = "DEMO_NO_TOKEN"

LOCK = Lock()

TRIAL_DAYS = 7
DAILY_BONUS = 5
FREE_CAP = 30

COSTS = {"daily": 1, "ingredients": 1, "steps": 2, "time": 1}

ALLOWED_TAGS = {
    "vegetarian", "vegan", "pescatarian",
    "gluten_free", "lactose_free",
    "high_protein", "low_calorie",
    "quick",
}

app = FastAPI()
import random

RNG = random.Random()

# ---------- MINI AI DATA (fast templates) ----------
BASE_ING = {
    "uk": {
        "protein": ["курка", "тунець", "яйця", "йогурт", "квасоля", "сир", "лосось"],
        "carb": ["рис", "паста", "картопля", "лаваш", "хліб", "гречка", "вівсянка"],
        "veg": ["помідор", "огірок", "перець", "цибуля", "часник", "шпинат", "морква", "гриби"],
        "fat": ["оливкова олія", "масло", "сметана", "авокадо"],
        "spice": ["сіль", "перець", "паприка", "орегано", "сушений часник", "зелень", "лимон"],
        "dairy": ["молоко", "йогурт", "сир", "сметана", "вершки"],
        "gluten": ["паста", "лаваш", "хліб", "булка", "панірувальні сухарі"],
    },
    "hr": {
        "protein": ["piletina", "tuna", "jaja", "jogurt", "grah", "sir", "losos"],
        "carb": ["riža", "tjestenina", "krumpir", "tortilja", "kruh", "zob"],
        "veg": ["rajčica", "krastavac", "paprika", "luk", "češnjak", "špinat", "mrkva", "gljive"],
        "fat": ["maslinovo ulje", "maslac", "vrhnje", "avokado"],
        "spice": ["sol", "papar", "paprika", "origano", "suhi češnjak", "peršin", "limun"],
        "dairy": ["mlijeko", "jogurt", "sir", "vrhnje", "slatko vrhnje"],
        "gluten": ["tjestenina", "tortilja", "kruh", "pecivo", "mrvice"],
    },
    "en": {
        "protein": ["chicken", "tuna", "eggs", "yogurt", "beans", "cheese", "salmon"],
        "carb": ["rice", "pasta", "potato", "wrap", "bread", "oats"],
        "veg": ["tomato", "cucumber", "pepper", "onion", "garlic", "spinach", "carrot", "mushrooms"],
        "fat": ["olive oil", "butter", "sour cream", "avocado"],
        "spice": ["salt", "pepper", "paprika", "oregano", "garlic powder", "herbs", "lemon"],
        "dairy": ["milk", "yogurt", "cheese", "sour cream", "cream"],
        "gluten": ["pasta", "wrap", "bread", "bun", "breadcrumbs"],
    }
}

TITLES = {
    "uk": {
        "bowl": ["Боул", "Салат-боул", "Поке-боул"],
        "omelet": ["Омлет", "Скрембл", "Фрітата"],
        "pasta": ["Паста", "Тепла паста", "Швидка паста"],
        "wrap": ["Рол", "Лаваш-рол", "Тортилья-рол"],
        "soup": ["Суп", "Крем-суп", "Легкий суп"],
        "oats": ["Вівсянка", "Овсяна каша", "Вівсянка-боул"],
    },
    "hr": {
        "bowl": ["Bowl", "Salata-bowl", "Poke bowl"],
        "omelet": ["Omlet", "Scramble", "Frittata"],
        "pasta": ["Tjestenina", "Topla tjestenina", "Brza tjestenina"],
        "wrap": ["Wrap", "Tortilja-wrap", "Roll"],
        "soup": ["Juha", "Krem juha", "Lagano varivo"],
        "oats": ["Zobena kaša", "Kaša", "Oatmeal bowl"],
    },
    "en": {
        "bowl": ["Bowl", "Salad bowl", "Poke bowl"],
        "omelet": ["Omelet", "Scramble", "Frittata"],
        "pasta": ["Pasta", "Warm pasta", "Quick pasta"],
        "wrap": ["Wrap", "Tortilla wrap", "Roll"],
        "soup": ["Soup", "Cream soup", "Light soup"],
        "oats": ["Oatmeal", "Oats bowl", "Porridge"],
    }
}

WHY = {
    "uk": ["Швидко і просто.", "Легко під фільтри.", "Ситно та без зайвого.", "Мінімум інгредієнтів."],
    "hr": ["Brzo i jednostavno.", "Lako za filtre.", "Zasitno i lagano.", "Malo sastojaka."],
    "en": ["Fast and simple.", "Easy to match filters.", "Filling but light.", "Minimal ingredients."],
}

def _pick(lang: str, group: str, k: int = 1) -> list[str]:
    arr = BASE_ING[lang][group]
    if k <= 1:
        return [RNG.choice(arr)]
    return RNG.sample(arr, k=min(k, len(arr)))

def _remove_excluded(items: list[str], exclude: list[str]) -> list[str]:
    if not exclude:
        return items
    ex = [e.strip().lower() for e in exclude if e.strip()]
    out = []
    for it in items:
        low = it.lower()
        if any(e in low for e in ex):
            continue
        out.append(it)
    return out

def _apply_constraints(lang: str, ingredients: list[str], f: dict) -> tuple[list[str], list[str]]:
    """Returns (ingredients, tags) after applying filter flags."""
    tags = set()

    # Diet tags
    diet = f.get("diet", "any")
    if diet in ("vegetarian", "vegan", "pescatarian"):
        tags.add(diet)

    # Gluten free
    if f.get("gluten_free"):
        tags.add("gluten_free")
        ingredients = [x for x in ingredients if x not in BASE_ING[lang]["gluten"]]

    # Lactose free
    if f.get("lactose_free"):
        tags.add("lactose_free")
        ingredients = [x for x in ingredients if x not in BASE_ING[lang]["dairy"]]

    # High protein / low calorie (tags only; logic simple)
    if f.get("high_protein"):
        tags.add("high_protein")
    if f.get("low_calorie"):
        tags.add("low_calorie")

    return ingredients, sorted(tags)

def mini_ai_generate_one(lang: str, f: dict) -> dict:
    """
    Very fast dish generator.
    Uses templates + random ingredients, then checks filters.
    """
    lang = lang if lang in ("uk", "hr", "en") else "uk"
    exclude = f.get("exclude") or []

    # Choose a template "type"
    kind = RNG.choice(["bowl", "omelet", "pasta", "wrap", "soup", "oats"])

    # Base time per template (minutes)
    base_time = {
        "bowl": 12, "omelet": 10, "pasta": 20, "wrap": 15, "soup": 25, "oats": 8
    }[kind]

    # Respect max_time if set: bias towards quick templates
    max_time = int(f.get("max_time", 0) or 0)
    if max_time and max_time <= 15:
        kind = RNG.choice(["omelet", "bowl", "oats", "wrap"])
        base_time = {"omelet":10, "bowl":12, "oats":8, "wrap":15}[kind]

    # Build ingredients depending on diet
    diet = f.get("diet", "any")
    protein = []
    if diet == "vegan":
        protein = ["квасоля"] if lang == "uk" else (["grah"] if lang == "hr" else ["beans"])
    elif diet == "vegetarian":
        # eggs/cheese allowed; keep it simple
        protein = _pick(lang, "protein", 1)
    elif diet == "pescatarian":
        # fish-biased
        protein = ["тунець"] if lang == "uk" else (["tuna"] if lang == "hr" else ["tuna"])
    else:
        protein = _pick(lang, "protein", 1)

    carbs = _pick(lang, "carb", 1)
    vegs = _pick(lang, "veg", 2)
    fats = _pick(lang, "fat", 1)
    spices = _pick(lang, "spice", 2)

    ingredients = protein + carbs + vegs + fats + spices
    ingredients = _remove_excluded(ingredients, exclude)

    # If exclusions removed too much, refill with safe veg/spice
    while len(ingredients) < 6:
        ingredients += _pick(lang, "veg", 1)
        ingredients = _remove_excluded(ingredients, exclude)

    ingredients, tags = _apply_constraints(lang, ingredients, f)

    # tags must be from allowed list
    tags = [t for t in tags if t in ALLOWED_TAGS]

    # Add quick tag if time is small
    time_total = base_time + RNG.randint(-2, 4)
    time_total = max(5, time_total)
    if time_total <= 15:
        tags = sorted(set(tags + ["quick"]))

    title = f"{RNG.choice(TITLES[lang][kind])}: " + (protein[0] if protein else ingredients[0])
    why = RNG.choice(WHY[lang])

    steps = {
        "uk": [
            "Підготуй інгредієнти (наріж овочі).",
            "Змішай/приготуй основу (на пательні або в каструлі).",
            "Додай спеції, перемішай і доведи до готовності."
        ],
        "hr": [
            "Pripremi sastojke (nareži povrće).",
            "Skuhaj/napravi bazu (tava ili lonac).",
            "Dodaj začine, promiješaj i dovrši."
        ],
        "en": [
            "Prep ingredients (chop veggies).",
            "Cook the base (pan or pot).",
            "Add spices, mix, and finish."
        ],
    }[lang]

    return {
        "title": title[:80],
        "why": why[:240],
        "ingredients": [x[:120] for x in ingredients[:12]],
        "steps": [x[:200] for x in steps],
        "time_total_min": int(time_total),
        "tags": tags,
    }

def mini_ai_pool(lang: str, f: dict, n: int = 10) -> list[dict]:
    # Generate until we get enough dishes that match filters
    out = []
    tries = 0
    while len(out) < n and tries < n * 20:
        d = mini_ai_generate_one(lang, f)
        if dish_matches_filters(d, f):
            out.append(d)
        tries += 1
    return out or [mini_ai_generate_one(lang, f) for _ in range(n)]



# -------------------- BASIC ROUTES --------------------

@app.get("/health")
def health():
    return {
        "ok": True,
        "has_web_index": os.path.exists("web/index.html"),
        "has_root_index": os.path.exists("index.html"),
    }


@app.get("/", response_class=HTMLResponse)
def root():
    # Mini App HTML
    if os.path.exists("web/index.html"):
        return FileResponse("web/index.html")
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return HTMLResponse(
        "<h2>index.html not found</h2><p>Expected: web/index.html</p>",
        status_code=500
    )


# -------------------- TIME/DB --------------------

def now() -> datetime:
    return datetime.now(TZ)

def today() -> str:
    return now().date().isoformat()

def load_db() -> Dict[str, Any]:
    if not os.path.exists(DB_PATH):
        return {"users": {}, "daily": {}, "used_titles": {"uk": [], "hr": [], "en": []}}
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_db(db: Dict[str, Any]) -> None:
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

def default_filters() -> Dict[str, Any]:
    return {
        "diet": "any",
        "gluten_free": False,
        "lactose_free": False,
        "high_protein": False,
        "low_calorie": False,
        "max_time": 0,
        "exclude": [],
    }

def get_user(db: Dict[str, Any], uid: int) -> Dict[str, Any]:
    suid = str(uid)
    if suid not in db["users"]:
        created = now()
        db["users"][suid] = {
            "lang": "uk",
            "tokens": 15,
            "created_at": created.isoformat(),
            "trial_until": (created + timedelta(days=TRIAL_DAYS)).isoformat(),
            "last_bonus": "",
            "filters": default_filters(),
            "daily_paid": "",
            "favorites": [],
            "uploads": [],
        }
    u = db["users"][suid]
    u.setdefault("favorites", [])
    u.setdefault("uploads", [])
    u.setdefault("daily_paid", "")
    u.setdefault("filters", default_filters())
    u.setdefault("lang", "uk")
    u.setdefault("tokens", 15)
    return u


# -------------------- TOKENS/TRIAL --------------------

def is_trial(u: Dict[str, Any]) -> bool:
    return now() < datetime.fromisoformat(u["trial_until"])

def trial_days_left(u: Dict[str, Any]) -> int:
    d = datetime.fromisoformat(u["trial_until"]) - now()
    return max(0, int(d.total_seconds() // 86400) + 1)

def apply_bonus(u: Dict[str, Any]) -> None:
    if u.get("last_bonus") == today():
        return
    u["last_bonus"] = today()
    if not is_trial(u):
        u["tokens"] = min(int(u.get("tokens", 0)) + DAILY_BONUS, FREE_CAP)

def charge(u: Dict[str, Any], feature: str) -> bool:
    if is_trial(u):
        return True
    cost = int(COSTS.get(feature, 1))
    if int(u.get("tokens", 0)) >= cost:
        u["tokens"] = int(u["tokens"]) - cost
        return True
    return False


# -------------------- DISH FILTERS --------------------

def dish_matches_filters(d: Dict[str, Any], f: Dict[str, Any]) -> bool:
    tags = set(d.get("tags", []))
    diet = f.get("diet", "any")
    if diet != "any" and diet not in tags:
        return False
    for k in ("gluten_free", "lactose_free", "high_protein", "low_calorie"):
        if f.get(k) and k not in tags:
            return False
    max_time = int(f.get("max_time", 0) or 0)
    if max_time and int(d.get("time_total_min", 10_000)) > max_time:
        return False
    excl = [x.strip().lower() for x in (f.get("exclude") or []) if x.strip()]
    if excl:
        blob = " ".join([str(x).lower() for x in d.get("ingredients", [])])
        if any(w in blob for w in excl):
            return False
    return True


# -------------------- AI GENERATION --------------------

def strip_fences(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^```(json)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()
    return s

def lang_name(lang: str) -> str:
    return {"uk": "Ukrainian", "hr": "Croatian", "en": "English"}.get(lang, "Ukrainian")

def fallback_pool(lang: str, n: int = 10) -> List[Dict[str, Any]]:
    base = [
        {
            "title": "Омлет з сиром" if lang == "uk" else ("Omlet sa sirom" if lang == "hr" else "Cheese omelet"),
            "why": "Швидко і просто." if lang == "uk" else ("Brzo i jednostavno." if lang == "hr" else "Fast and simple."),
            "ingredients": ["яйця", "сир", "сіль", "перець"] if lang == "uk"
                           else (["jaja", "sir", "sol", "papar"] if lang == "hr" else ["eggs", "cheese", "salt", "pepper"]),
            "steps": ["Збий яйця", "Додай сир", "Посмаж 5-7 хв"] if lang == "uk"
                     else (["Umuti jaja", "Dodaj sir", "Prži 5-7 min"] if lang == "hr" else ["Beat eggs", "Add cheese", "Fry 5–7 min"]),
            "time_total_min": 10,
            "tags": ["vegetarian", "quick"],
        },
        {
            "title": "Салат з тунцем" if lang == "uk" else ("Salata s tunom" if lang == "hr" else "Tuna salad"),
            "why": "Багато білка." if lang == "uk" else ("Puno proteina." if lang == "hr" else "High protein."),
            "ingredients": ["тунець", "огірок", "помідор", "оливкова олія"] if lang == "uk"
                           else (["tuna", "krastavac", "rajčica", "maslinovo ulje"] if lang == "hr" else ["tuna", "cucumber", "tomato", "olive oil"]),
            "steps": ["Наріж овочі", "Додай тунець", "Заправ олією"] if lang == "uk"
                     else (["Nareži povrće", "Dodaj tunu", "Začini uljem"] if lang == "hr" else ["Chop veggies", "Add tuna", "Dress with oil"]),
            "time_total_min": 12,
            "tags": ["pescatarian", "high_protein", "quick"],
        },
        {
            "title": "Вівсянка з бананом" if lang == "uk" else ("Zobena kaša s bananom" if lang == "hr" else "Oatmeal with banana"),
            "why": "Легкий сніданок." if lang == "uk" else ("Lagani doručak." if lang == "hr" else "Easy breakfast."),
            "ingredients": ["вівсянка", "банан", "молоко/вода"] if lang == "uk"
                           else (["zob", "banana", "mlijeko/voda"] if lang == "hr" else ["oats", "banana", "milk/water"]),
            "steps": ["Залий вівсянку", "Вари 5 хв", "Додай банан"] if lang == "uk"
                     else (["Prelij zob", "Kuhaj 5 min", "Dodaj bananu"] if lang == "hr" else ["Add liquid", "Cook 5 min", "Add banana"]),
            "time_total_min": 8,
            "tags": ["vegetarian", "quick"],
        },
    ]
    out = []
    i = 0
    while len(out) < n:
        item = base[i % len(base)].copy()
        if i >= len(base):
            item["title"] = f"{item['title']} #{(i//len(base))+1}"
        out.append(item)
        i += 1
    return out[:n]

def ai_pool(lang: str, forbidden: List[str], n: int = 10) -> List[Dict[str, Any]]:
    # Якщо нема ключа — не ліземо в AI
    if not AI_API_KEY:
        return fallback_pool(lang, n)

    try:
        from openai import OpenAI
        client = OpenAI(base_url=AI_ENDPOINT, api_key=AI_API_KEY)

        forb = ", ".join(forbidden[-120:])
        system = "You are a creative chef. Output ONLY valid JSON array. No markdown."
        user = f"""
Generate {n} different dish ideas in {lang_name(lang)} for today.
Forbidden titles: [{forb}]

Return JSON ARRAY of objects with keys:
title, why, ingredients[], steps[], time_total_min, tags[]
tags MUST be English from: {sorted(list(ALLOWED_TAGS))}
Only JSON ARRAY.
""".strip()

        r = client.chat.completions.create(
            model=AI_MODEL,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=1.1,
        )

        data = json.loads(strip_fences(r.choices[0].message.content or "[]"))
        out = []
        for it in data:
            if not isinstance(it, dict):
                continue
            title = str(it.get("title", "")).strip()
            if not title:
                continue
            tags = [t for t in (it.get("tags") or []) if t in ALLOWED_TAGS]
            out.append({
                "title": title[:80],
                "why": str(it.get("why",""))[:240],
                "ingredients": [str(x)[:120] for x in (it.get("ingredients") or []) if str(x).strip()],
                "steps": [str(x)[:200] for x in (it.get("steps") or []) if str(x).strip()],
                "time_total_min": int(it.get("time_total_min", 30) or 30),
                "tags": tags,
            })
        if len(out) < 3:
            return fallback_pool(lang, n)
        return out[:n]
    except Exception:
        return fallback_pool(lang, n)


def get_pool(db: Dict[str, Any], lang: str) -> List[Dict[str, Any]]:
    d = today()
    db.setdefault("daily", {}).setdefault(d, {})

    if lang in db["daily"][d] and isinstance(db["daily"][d][lang].get("pool"), list):
        return db["daily"][d][lang]["pool"]

    db.setdefault("used_titles", {}).setdefault(lang, [])
    pool = ai_pool(lang, db["used_titles"][lang], 10)

    db["daily"][d][lang] = {"pool": pool, "generated_at": now().isoformat()}
    for x in pool:
        db["used_titles"][lang].append(x["title"])
    db["used_titles"][lang] = db["used_titles"][lang][-500:]
    return pool

def pick_daily(db: Dict[str, Any], u: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    lang = u.get("lang", "uk")
    pool = get_pool(db, lang)
    f = u.get("filters", default_filters())
    matches = [i for i, d in enumerate(pool) if dish_matches_filters(d, f)]
    if not matches:
        return None
    return pool[matches[0]]


# -------------------- TELEGRAM INITDATA VERIFY --------------------

def validate_init_data(init_data: str, bot_token: str, max_age_sec: int = 86400) -> int:
    pairs = dict(parse_qsl(init_data, keep_blank_values=True))
    recv_hash = pairs.pop("hash", None)
    if not recv_hash:
        raise HTTPException(401, "No hash")

    data_check_string = "\n".join([f"{k}={pairs[k]}" for k in sorted(pairs.keys())])
    secret_key = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    calc_hash = hmac.new(secret_key, data_check_string.encode(), hashlib.sha256).hexdigest()
    if calc_hash != recv_hash:
        raise HTTPException(401, "Bad signature")

    auth_date = int(pairs.get("auth_date", "0"))
    if auth_date and (now().timestamp() - auth_date) > max_age_sec:
        raise HTTPException(401, "InitData expired")

    user_raw = pairs.get("user")
    if not user_raw:
        raise HTTPException(401, "No user")
    user = json.loads(user_raw)
    return int(user["id"])

def uid_from_init(init_data: str) -> int:
    if not init_data:
        raise HTTPException(401, "Missing initData")
    return validate_init_data(init_data, BOT_TOKEN)


# -------------------- API --------------------

def demo_status():
    return {
        "lang": "uk",
        "trial": True,
        "trial_days_left": TRIAL_DAYS,
        "tokens": 15,
        "filters": default_filters(),
        "demo": True
    }

def demo_dish():
    return {
        "title": "Demo: Омлет за 10 хв",
        "why": "Демо-режим (відкрито не в Telegram). Відкрий Mini App через бота для персоналізації.",
        "ingredients": ["2 яйця", "сіль", "перець", "трохи масла"],
        "steps": ["Збий яйця з сіллю.", "Розігрій пательню з маслом.", "Вилий яйця, готуй 2-3 хв."],
        "time_total_min": 10,
        "tags": ["quick", "high_protein"]
    }

@app.get("/api/status")
def api_status(x_telegram_init_data: str = Header(default="")):
    if not x_telegram_init_data:
        return demo_status()

    user_id = uid_from_init(x_telegram_init_data)
    db = load_db()
    u = get_user(db, user_id)
    apply_bonus(u)
    save_db(db)
    return {
        "lang": u["lang"],
        "trial": is_trial(u),
        "trial_days_left": trial_days_left(u),
        "tokens": u["tokens"],
        "filters": u["filters"],
        "demo": False
    }

@app.post("/api/lang")
def api_lang(payload: Dict[str, Any], x_telegram_init_data: str = Header(default="")):
    # DEMO: просто приймаємо, щоб UI не ламався в браузері
    if not x_telegram_init_data:
        return {"ok": True, "demo": True}

    user_id = uid_from_init(x_telegram_init_data)
    lang = payload.get("lang", "uk")
    if lang not in ("uk", "hr", "en"):
        lang = "uk"
    db = load_db()
    u = get_user(db, user_id)
    u["lang"] = lang
    save_db(db)
    return {"ok": True, "demo": False}

@app.post("/api/filters")
def api_filters(payload: Dict[str, Any], x_telegram_init_data: str = Header(default="")):
    if not x_telegram_init_data:
        return {"ok": True, "demo": True}

    user_id = uid_from_init(x_telegram_init_data)
    db = load_db()
    u = get_user(db, user_id)

    f = u.get("filters", default_filters())
    for k in f.keys():
        if k in payload:
            f[k] = payload[k]
    u["filters"] = f

    save_db(db)
    return {"ok": True, "demo": False}

@app.post("/api/daily")
def api_daily(x_telegram_init_data: str = Header(default="")):
    if not x_telegram_init_data:
        return {"ok": True, "dish": demo_dish(), "demo": True}

    user_id = uid_from_init(x_telegram_init_data)

    with LOCK:
        db = load_db()
        u = get_user(db, user_id)
        apply_bonus(u)

        if not is_trial(u):
            if u.get("daily_paid") != today():
                if not charge(u, "daily"):
                    save_db(db)
                    raise HTTPException(402, "NO_TOKENS")
                u["daily_paid"] = today()

        dish = pick_daily(db, u)
        save_db(db)

    return {"ok": True, "dish": dish, "demo": False}

@app.post("/api/action")
def api_action(payload: Dict[str, Any], x_telegram_init_data: str = Header(default="")):
    action = payload.get("action")
    if action not in ("ingredients", "steps", "time"):
        raise HTTPException(400, "bad action")

    # DEMO: щоб UI в браузері не ламався
    if not x_telegram_init_data:
        d = demo_dish()
        if action == "ingredients":
            return {"ok": True, "data": d["ingredients"], "demo": True}
        if action == "steps":
            return {"ok": True, "data": d["steps"], "demo": True}
        return {"ok": True, "data": d["time_total_min"], "demo": True}

    user_id = uid_from_init(x_telegram_init_data)

    db = load_db()
    u = get_user(db, user_id)
    apply_bonus(u)

    if not charge(u, action):
        save_db(db)
        raise HTTPException(402, "NO_TOKENS")

    dish = pick_daily(db, u)
    save_db(db)

    if not dish:
        return {"ok": True, "data": None, "demo": False}
    if action == "ingredients":
        return {"ok": True, "data": dish.get("ingredients", []), "demo": False}
    if action == "steps":
        return {"ok": True, "data": dish.get("steps", []), "demo": False}
    return {"ok": True, "data": dish.get("time_total_min", 0), "demo": False}
