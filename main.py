import os, json, re, hmac, hashlib, random
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from urllib.parse import parse_qsl
from typing import Dict, Any, List, Optional
from threading import Lock

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

# ==================== CONFIG ====================
TZ = ZoneInfo("Europe/Uzhgorod")
DB_PATH = "db.json"

BOT_TOKEN = os.environ.get("BOT_TOKEN", "")  # REQUIRED for Telegram initData + webhook + Stars
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")  # optional but recommended

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

# Telegram Stars packs (edit as you want)
STAR_PACKS = {
    "small":  {"stars": 25,  "tokens": 50,  "title": "50 tokens",  "desc": "Token pack (small)"},
    "medium": {"stars": 75,  "tokens": 180, "title": "180 tokens", "desc": "Token pack (medium)"},
    "large":  {"stars": 200, "tokens": 550, "title": "550 tokens", "desc": "Token pack (large)"},
}

LOCK = Lock()
RNG = random.Random()

app = FastAPI()

# ==================== BASIC ROUTES ====================

@app.get("/health")
def health():
    return {
        "ok": True,
        "has_bot_token": bool(BOT_TOKEN),
        "has_webhook_secret": bool(WEBHOOK_SECRET),
        "files": {
            "web/index.html": os.path.exists("web/index.html"),
            "index.html": os.path.exists("index.html"),
            "db.json": os.path.exists(DB_PATH),
        }
    }

@app.get("/", response_class=HTMLResponse)
def root():
    if os.path.exists("web/index.html"):
        return FileResponse("web/index.html")
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return HTMLResponse(
        "<h2>index.html not found</h2><p>Expected: web/index.html</p>",
        status_code=500
    )

# ==================== TIME/DB ====================

def now() -> datetime:
    return datetime.now(TZ)

def today() -> str:
    return now().date().isoformat()

def load_db() -> Dict[str, Any]:
    if not os.path.exists(DB_PATH):
        return {"users": {}, "daily": {}}
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
    if "users" not in db:
        db["users"] = {}
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
    u.setdefault("lang", "uk")
    u.setdefault("tokens", 15)
    u.setdefault("filters", default_filters())
    u.setdefault("daily_paid", "")
    u.setdefault("favorites", [])
    u.setdefault("uploads", [])
    u.setdefault("last_bonus", "")
    u.setdefault("trial_until", (now() + timedelta(days=TRIAL_DAYS)).isoformat())
    return u

# ==================== TOKENS/TRIAL ====================

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

# ==================== FILTER MATCH ====================

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

# ==================== MINI AI (FAST LOCAL) ====================

BASE_ING = {
    "uk": {
        "protein": ["курка", "тунець", "яйця", "йогурт", "квасоля", "сир", "лосось", "індичка"],
        "carb": ["рис", "паста", "картопля", "лаваш", "хліб", "гречка", "вівсянка", "кіноа"],
        "veg": ["помідор", "огірок", "перець", "цибуля", "часник", "шпинат", "морква", "гриби", "капуста"],
        "fat": ["оливкова олія", "масло", "сметана", "авокадо"],
        "spice": ["сіль", "перець", "паприка", "орегано", "зелень", "лимон", "соєвий соус"],
        "dairy": ["молоко", "йогурт", "сир", "сметана", "вершки"],
        "gluten": ["паста", "лаваш", "хліб", "булка", "панірувальні сухарі"],
    },
    "hr": {
        "protein": ["piletina", "tuna", "jaja", "jogurt", "grah", "sir", "losos", "puretina"],
        "carb": ["riža", "tjestenina", "krumpir", "tortilja", "kruh", "zob", "kvinoja"],
        "veg": ["rajčica", "krastavac", "paprika", "luk", "češnjak", "špinat", "mrkva", "gljive", "kupus"],
        "fat": ["maslinovo ulje", "maslac", "vrhnje", "avokado"],
        "spice": ["sol", "papar", "paprika", "origano", "peršin", "limun", "sojin umak"],
        "dairy": ["mlijeko", "jogurt", "sir", "vrhnje", "slatko vrhnje"],
        "gluten": ["tjestenina", "tortilja", "kruh", "pecivo", "mrvice"],
    },
    "en": {
        "protein": ["chicken", "tuna", "eggs", "yogurt", "beans", "cheese", "salmon", "turkey"],
        "carb": ["rice", "pasta", "potato", "wrap", "bread", "oats", "quinoa"],
        "veg": ["tomato", "cucumber", "pepper", "onion", "garlic", "spinach", "carrot", "mushrooms", "cabbage"],
        "fat": ["olive oil", "butter", "sour cream", "avocado"],
        "spice": ["salt", "pepper", "paprika", "oregano", "herbs", "lemon", "soy sauce"],
        "dairy": ["milk", "yogurt", "cheese", "sour cream", "cream"],
        "gluten": ["pasta", "wrap", "bread", "bun", "breadcrumbs"],
    },
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

STEPS = {
    "uk": [
        "Підготуй інгредієнти (наріж овочі).",
        "Приготуй основу (на пательні або в каструлі).",
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
}

def _pick(lang: str, group: str, k: int = 1) -> List[str]:
    arr = BASE_ING[lang][group]
    if k <= 1:
        return [RNG.choice(arr)]
    return RNG.sample(arr, k=min(k, len(arr)))

def _remove_excluded(items: List[str], exclude: List[str]) -> List[str]:
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

def _apply_constraints(lang: str, ingredients: List[str], f: Dict[str, Any]) -> (List[str], List[str]):
    tags = set()

    diet = f.get("diet", "any")
    if diet in ("vegetarian", "vegan", "pescatarian"):
        tags.add(diet)

    if f.get("gluten_free"):
        tags.add("gluten_free")
        ingredients = [x for x in ingredients if x not in BASE_ING[lang]["gluten"]]

    if f.get("lactose_free"):
        tags.add("lactose_free")
        ingredients = [x for x in ingredients if x not in BASE_ING[lang]["dairy"]]

    if f.get("high_protein"):
        tags.add("high_protein")
    if f.get("low_calorie"):
        tags.add("low_calorie")

    tags = [t for t in tags if t in ALLOWED_TAGS]
    return ingredients, sorted(tags)

def mini_ai_generate_one(lang: str, f: Dict[str, Any]) -> Dict[str, Any]:
    lang = lang if lang in ("uk", "hr", "en") else "uk"
    exclude = f.get("exclude") or []

    kind = RNG.choice(["bowl", "omelet", "pasta", "wrap", "soup", "oats"])
    base_time = {"bowl": 12, "omelet": 10, "pasta": 20, "wrap": 15, "soup": 25, "oats": 8}[kind]

    max_time = int(f.get("max_time", 0) or 0)
    if max_time and max_time <= 15:
        kind = RNG.choice(["omelet", "bowl", "oats", "wrap"])
        base_time = {"omelet": 10, "bowl": 12, "oats": 8, "wrap": 15}[kind]

    diet = f.get("diet", "any")
    if diet == "vegan":
        protein = ["квасоля"] if lang == "uk" else (["grah"] if lang == "hr" else ["beans"])
    elif diet == "pescatarian":
        protein = ["тунець"] if lang == "uk" else (["tuna"] if lang == "hr" else ["tuna"])
    else:
        protein = _pick(lang, "protein", 1)

    carbs = _pick(lang, "carb", 1)
    vegs = _pick(lang, "veg", 2)
    fats = _pick(lang, "fat", 1)
    spices = _pick(lang, "spice", 2)

    ingredients = protein + carbs + vegs + fats + spices
    ingredients = _remove_excluded(ingredients, exclude)

    while len(ingredients) < 6:
        ingredients += _pick(lang, "veg", 1)
        ingredients = _remove_excluded(ingredients, exclude)

    ingredients, tags = _apply_constraints(lang, ingredients, f)

    time_total = max(5, base_time + RNG.randint(-2, 4))
    if time_total <= 15:
        tags = sorted(set(tags + ["quick"]))

    title = f"{RNG.choice(TITLES[lang][kind])}: {protein[0]}"
    why = RNG.choice(WHY[lang])

    dish = {
        "title": title[:80],
        "why": why[:240],
        "ingredients": [x[:120] for x in ingredients[:12]],
        "steps": [x[:200] for x in STEPS[lang]],
        "time_total_min": int(time_total),
        "tags": [t for t in tags if t in ALLOWED_TAGS],
    }
    return dish

def mini_ai_pool(lang: str, f: Dict[str, Any], n: int = 10) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    tries = 0
    while len(out) < n and tries < n * 30:
        d = mini_ai_generate_one(lang, f)
        if dish_matches_filters(d, f):
            out.append(d)
        tries += 1
    if not out:
        out = [mini_ai_generate_one(lang, f) for _ in range(n)]
    return out[:n]

# ==================== DAILY POOL (cache by day+lang+filters) ====================

def get_pool(db: Dict[str, Any], lang: str, f: Dict[str, Any]) -> List[Dict[str, Any]]:
    d = today()
    db.setdefault("daily", {}).setdefault(d, {})
    bucket = db["daily"][d].setdefault(lang, {})

    filter_key = json.dumps(f, ensure_ascii=False, sort_keys=True)
    if bucket.get("filter_key") == filter_key and isinstance(bucket.get("pool"), list):
        return bucket["pool"]

    pool = mini_ai_pool(lang, f, 10)
    bucket["pool"] = pool
    bucket["filter_key"] = filter_key
    bucket["generated_at"] = now().isoformat()
    return pool

def pick_daily(db: Dict[str, Any], u: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    lang = u.get("lang", "uk")
    f = u.get("filters", default_filters())

    pool = get_pool(db, lang, f)
    matches = [d for d in pool if dish_matches_filters(d, f)]
    if not matches:
        return None
    return matches[0]

# ==================== TELEGRAM INITDATA VERIFY ====================

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
    if not BOT_TOKEN:
        raise HTTPException(500, "BOT_TOKEN is not set on server")
    return validate_init_data(init_data, BOT_TOKEN)

# ==================== TELEGRAM API HELPERS ====================

async def tg_api(method: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if not BOT_TOKEN:
        raise HTTPException(500, "BOT_TOKEN is not set on server")

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(url, json=payload)
    data = r.json()

    if not data.get("ok"):
        raise HTTPException(500, f"Telegram API error: {data}")

    return data

# ==================== API ====================

@app.get("/api/status")
def api_status(x_telegram_init_data: str = Header(default="")):
    # DEMO (browser / not inside Telegram WebApp)
    if not x_telegram_init_data:
        return {
            "lang": "uk",
            "trial": True,
            "trial_days_left": TRIAL_DAYS,
            "tokens": 15,
            "filters": default_filters(),
            "demo": True
        }

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
    user_id = uid_from_init(x_telegram_init_data)
    lang = payload.get("lang", "uk")
    if lang not in ("uk", "hr", "en"):
        lang = "uk"

    db = load_db()
    u = get_user(db, user_id)
    u["lang"] = lang
    save_db(db)
    return {"ok": True}

@app.post("/api/filters")
def api_filters(payload: Dict[str, Any], x_telegram_init_data: str = Header(default="")):
    user_id = uid_from_init(x_telegram_init_data)

    db = load_db()
    u = get_user(db, user_id)

    f = u.get("filters", default_filters())
    for k in f.keys():
        if k in payload:
            f[k] = payload[k]
    u["filters"] = f
    save_db(db)
    return {"ok": True}

@app.post("/api/daily")
def api_daily(x_telegram_init_data: str = Header(default="")):
    # DEMO in browser (still not empty)
    if not x_telegram_init_data:
        dish = mini_ai_generate_one("uk", default_filters())
        return {"ok": True, "dish": dish, "demo": True}

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
    user_id = uid_from_init(x_telegram_init_data)
    action = payload.get("action")
    if action not in ("ingredients", "steps", "time"):
        raise HTTPException(400, "bad action")

    with LOCK:
        db = load_db()
        u = get_user(db, user_id)
        apply_bonus(u)

        if not charge(u, action):
            save_db(db)
            raise HTTPException(402, "NO_TOKENS")

        dish = pick_daily(db, u)
        save_db(db)

    if not dish:
        return {"ok": True, "data": None}
    if action == "ingredients":
        return {"ok": True, "data": dish.get("ingredients", [])}
    if action == "steps":
        return {"ok": True, "data": dish.get("steps", [])}
    return {"ok": True, "data": dish.get("time_total_min", 0)}

# ==================== TELEGRAM STARS: CREATE INVOICE LINK ====================

@app.post("/api/stars/create_link")
async def api_stars_create_link(payload: Dict[str, Any], x_telegram_init_data: str = Header(default="")):
    # must be inside Telegram (need uid)
    user_id = uid_from_init(x_telegram_init_data)

    pack = payload.get("pack", "small")
    if pack not in STAR_PACKS:
        raise HTTPException(400, "bad pack")

    p = STAR_PACKS[pack]

    # payload that will come back in successful_payment.invoice_payload
    invoice_payload = f"tokens:{user_id}:{pack}:{int(now().timestamp())}"

    # Stars invoice: currency="XTR", provider_token="" , prices = one item, amount = Stars count
    res = await tg_api("createInvoiceLink", {
        "title": p["title"],
        "description": p["desc"],
        "payload": invoice_payload,
        "provider_token": "",
        "currency": "XTR",
        "prices": [{"label": p["title"], "amount": int(p["stars"])}],
    })

    return {
        "ok": True,
        "link": res["result"],
        "pack": pack,
        "stars": p["stars"],
        "tokens": p["tokens"],
    }

# ==================== TELEGRAM WEBHOOK (STARS CONFIRMATION) ====================

def _webhook_secret_ok(req: Request) -> bool:
    # Option A: Telegram header if you set it via setWebhook secret_token
    h = req.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
    if WEBHOOK_SECRET and h and h == WEBHOOK_SECRET:
        return True
    # Option B: query parameter ?s=SECRET (easy setup)
    if WEBHOOK_SECRET:
        s = req.query_params.get("s", "")
        return s == WEBHOOK_SECRET
    # If no secret configured, accept (not recommended)
    return True

@app.post("/tg/webhook")
async def tg_webhook(request: Request):
    if not _webhook_secret_ok(request):
        raise HTTPException(403, "Bad webhook secret")

    update = await request.json()

    # 1) Must answer pre_checkout_query
    if "pre_checkout_query" in update:
        pcq = update["pre_checkout_query"]
        await tg_api("answerPreCheckoutQuery", {
            "pre_checkout_query_id": pcq["id"],
            "ok": True
        })
        return {"ok": True}

    # 2) successful_payment arrives in message
    msg = update.get("message") or update.get("edited_message")
    if msg and msg.get("successful_payment"):
        sp = msg["successful_payment"]
        inv_payload = sp.get("invoice_payload", "")

        # expected: tokens:{user_id}:{pack}:{ts}
        m = re.match(r"^tokens:(\d+):([a-z]+):(\d+)$", inv_payload)
        if not m:
            return {"ok": True}

        user_id = int(m.group(1))
        pack = m.group(2)
        if pack not in STAR_PACKS:
            return {"ok": True}

        tokens_add = int(STAR_PACKS[pack]["tokens"])

        with LOCK:
            db = load_db()
            u = get_user(db, user_id)
            u["tokens"] = int(u.get("tokens", 0)) + tokens_add
            save_db(db)

        return {"ok": True}

    return {"ok": True}
