import os, json, re, hmac, hashlib, random
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from urllib.parse import parse_qsl
from typing import Dict, Any, List, Optional

from threading import Lock

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse

TZ = ZoneInfo("Europe/Uzhgorod")
DB_PATH = "db.json"

BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
AI_API_KEY = os.environ.get("AI_API_KEY", "")
AI_ENDPOINT = os.environ.get("AI_ENDPOINT", "https://models.github.ai/inference")
AI_MODEL = os.environ.get("AI_MODEL", "openai/gpt-4o-mini")

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


# -------------------- BASIC ROUTES --------------------

@app.get("/health")
def health():
    return {
        "ok": True,
        "exists": {
            "web/index.html": os.path.exists("web/index.html"),
            "db.json": os.path.exists(DB_PATH),
        },
        "env": {
            "BOT_TOKEN": bool(BOT_TOKEN),
            "AI_API_KEY": bool(AI_API_KEY),
            "AI_MODEL": AI_MODEL,
        }
    }


@app.get("/", response_class=HTMLResponse)
def root():
    if os.path.exists("web/index.html"):
        return FileResponse("web/index.html")
    return HTMLResponse("<h2>Missing web/index.html</h2>", status_code=500)


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
            "daily_choice": {},  # { "YYYY-MM-DD": { "uk": dish_obj, "hr": dish_obj, ... } }
        }

    u = db["users"][suid]
    u.setdefault("daily_paid", "")
    u.setdefault("filters", default_filters())
    u.setdefault("daily_choice", {})
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


# -------------------- FILTERS --------------------

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


# -------------------- FAST LOCAL "MINI AI" GENERATOR --------------------
# (швидко, без зовнішніх API, і видає багато різних страв)

def _tr(lang: str, uk: str, hr: str, en: str) -> str:
    return {"uk": uk, "hr": hr, "en": en}.get(lang, uk)

def _seed_for(uid: int, lang: str, extra: str = "") -> int:
    # стабільний seed на день, щоб "страва дня" не скакала сама по собі
    base = f"{today()}|{uid}|{lang}|{extra}"
    return int(hashlib.sha256(base.encode("utf-8")).hexdigest()[:12], 16)

def local_pool(lang: str, n: int = 30) -> List[Dict[str, Any]]:
    # База компонентів
    proteins = [
        (_tr(lang, "курка", "piletina", "chicken"), "high_protein"),
        (_tr(lang, "тунець", "tuna", "tuna"), "high_protein"),
        (_tr(lang, "яйця", "jaja", "eggs"), "high_protein"),
        (_tr(lang, "тофу", "tofu", "tofu"), "vegan"),
        (_tr(lang, "квасоля", "grah", "beans"), "vegan"),
        (_tr(lang, "лосось", "losos", "salmon"), "high_protein"),
    ]
    carbs = [
        _tr(lang, "рис", "riža", "rice"),
        _tr(lang, "паста", "tjestenina", "pasta"),
        _tr(lang, "картопля", "krumpir", "potato"),
        _tr(lang, "гречка", "heljda", "buckwheat"),
        _tr(lang, "тортилья", "tortilja", "tortilla"),
    ]
    veggies = [
        _tr(lang, "помідор", "rajčica", "tomato"),
        _tr(lang, "огірок", "krastavac", "cucumber"),
        _tr(lang, "перець", "paprika", "pepper"),
        _tr(lang, "шпинат", "špinat", "spinach"),
        _tr(lang, "морква", "mrkva", "carrot"),
        _tr(lang, "цибуля", "luk", "onion"),
        _tr(lang, "броколі", "brokula", "broccoli"),
    ]
    sauces = [
        _tr(lang, "йогуртовий соус", "jogurt umak", "yogurt sauce"),
        _tr(lang, "томатний соус", "umak od rajčice", "tomato sauce"),
        _tr(lang, "соєвий соус", "soja umak", "soy sauce"),
        _tr(lang, "оливкова олія", "maslinovo ulje", "olive oil"),
    ]

    templates = [
        ("bowl", _tr(lang, "Боул", "Bowl", "Bowl")),
        ("salad", _tr(lang, "Салат", "Salata", "Salad")),
        ("wrap", _tr(lang, "Врап", "Wrap", "Wrap")),
        ("pasta", _tr(lang, "Паста", "Tjestenina", "Pasta")),
        ("omelet", _tr(lang, "Омлет", "Omlet", "Omelet")),
        ("stir", _tr(lang, "Стір-фрай", "Stir-fry", "Stir-fry")),
    ]

    out: List[Dict[str, Any]] = []
    rng = random.Random(_seed_for(0, lang, "POOL"))

    for i in range(n):
        kind, kind_name = rng.choice(templates)
        prot, prot_tag = rng.choice(proteins)
        carb = rng.choice(carbs)
        veg1 = rng.choice(veggies)
        veg2 = rng.choice(veggies)
        sauce = rng.choice(sauces)

        # title
        if kind == "salad":
            title = _tr(lang, f"{kind_name} з {prot} і {veg1}", f"{kind_name} s {prot} i {veg1}", f"{kind_name} with {prot} and {veg1}")
        elif kind == "wrap":
            title = _tr(lang, f"{kind_name}: {prot} + {veg1}", f"{kind_name}: {prot} + {veg1}", f"{kind_name}: {prot} + {veg1}")
        elif kind == "omelet":
            title = _tr(lang, f"{kind_name} з {veg1}", f"{kind_name} s {veg1}", f"{kind_name} with {veg1}")
        elif kind == "pasta":
            title = _tr(lang, f"{kind_name} з {sauce}", f"{kind_name} s {sauce}", f"{kind_name} with {sauce}")
        else:
            title = _tr(lang, f"{kind_name} {prot} + {carb}", f"{kind_name} {prot} + {carb}", f"{kind_name} {prot} + {carb}")

        # tags
        tags = set(["quick"])
        if prot_tag == "vegan":
            tags.add("vegan")
        else:
            # якщо не tofu/beans — може бути vegetarian/pescatarian
            if prot in (_tr(lang, "тунець", "tuna", "tuna"), _tr(lang, "лосось", "losos", "salmon")):
                tags.add("pescatarian")
            elif prot in (_tr(lang, "яйця", "jaja", "eggs"),):
                tags.add("vegetarian")

        if prot_tag == "high_protein":
            tags.add("high_protein")

        time_total = rng.choice([10, 12, 15, 18, 20, 25, 30])

        ingredients = [
            prot,
            carb if kind not in ("salad", "omelet") else "",
            veg1, veg2,
            sauce,
            _tr(lang, "сіль", "sol", "salt"),
            _tr(lang, "перець", "papar", "pepper"),
        ]
        ingredients = [x for x in ingredients if x]

        steps = [
            _tr(lang, "Підготуй інгредієнти (поріж овочі).", "Pripremi sastojke (nareži povrće).", "Prep ingredients (chop veggies)."),
            _tr(lang, f"Приготуй {prot} (або відкрий консерву).", f"Pripremi {prot} (ili otvori konzervu).", f"Prepare {prot} (or open can)."),
            _tr(lang, f"Змішай з {sauce} та приправами.", f"Pomiješaj sa {sauce} i začinima.", f"Mix with {sauce} and spices."),
            _tr(lang, "Подавай одразу.", "Posluži odmah.", "Serve immediately."),
        ]

        why = _tr(
            lang,
            "Швидко, просто і можна змінювати під себе.",
            "Brzo, jednostavno i možeš prilagoditi.",
            "Fast, simple, and easy to customize."
        )

        out.append({
            "title": title[:80],
            "why": why[:240],
            "ingredients": [x[:120] for x in ingredients],
            "steps": [x[:200] for x in steps],
            "time_total_min": int(time_total),
            "tags": sorted(list(tags)),
        })

    return out


def get_pool(db: Dict[str, Any], lang: str) -> List[Dict[str, Any]]:
    d = today()
    db.setdefault("daily", {}).setdefault(d, {})
    if lang in db["daily"][d] and isinstance(db["daily"][d][lang].get("pool"), list):
        return db["daily"][d][lang]["pool"]

    pool = local_pool(lang, 30)
    db["daily"][d][lang] = {"pool": pool, "generated_at": now().isoformat()}
    return pool


def pick_for_user(db: Dict[str, Any], u: Dict[str, Any], force: bool = False) -> Optional[Dict[str, Any]]:
    lang = u.get("lang", "uk")
    d = today()

    # cached choice for today+lang
    u.setdefault("daily_choice", {})
    u["daily_choice"].setdefault(d, {})
    if (not force) and (lang in u["daily_choice"][d]):
        return u["daily_choice"][d][lang]

    pool = get_pool(db, lang)
    f = u.get("filters", default_filters())

    matches = [dish for dish in pool if dish_matches_filters(dish, f)]
    if not matches:
        u["daily_choice"][d].pop(lang, None)
        return None

    # pick random but stable unless force
    if force:
        rng = random.Random(_seed_for(int(u.get("tokens", 0)) + int(now().timestamp()), lang, f"FORCE|{random.random()}"))
    else:
        # stable "dish of day" per user
        rng = random.Random(_seed_for(int(u.get("tokens", 0)) + 999, lang, f"USER|{u.get('created_at','')}"))

    dish = rng.choice(matches)
    u["daily_choice"][d][lang] = dish
    return dish


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
        # demo browser mode
        return 0
    if not BOT_TOKEN:
        raise HTTPException(500, "BOT_TOKEN is missing on server")
    return validate_init_data(init_data, BOT_TOKEN)


# -------------------- API --------------------

@app.get("/api/status")
def api_status(x_telegram_init_data: str = Header(default="")):
    user_id = uid_from_init(x_telegram_init_data)

    with LOCK:
        db = load_db()
        u = get_user(db, user_id)
        apply_bonus(u)
        save_db(db)

    return {
        "lang": u["lang"],
        "trial": is_trial(u),
        "trial_days_left": trial_days_left(u),
        "tokens": int(u.get("tokens", 0)),
        "filters": u.get("filters", default_filters()),
        "demo": (user_id == 0),
    }


@app.post("/api/lang")
def api_lang(payload: Dict[str, Any], x_telegram_init_data: str = Header(default="")):
    user_id = uid_from_init(x_telegram_init_data)
    lang = payload.get("lang", "uk")
    if lang not in ("uk", "hr", "en"):
        lang = "uk"

    with LOCK:
        db = load_db()
        u = get_user(db, user_id)
        u["lang"] = lang
        # when changing language, force re-pick for that lang on next refresh
        save_db(db)

    return {"ok": True}


@app.post("/api/filters")
def api_filters(payload: Dict[str, Any], x_telegram_init_data: str = Header(default="")):
    user_id = uid_from_init(x_telegram_init_data)

    with LOCK:
        db = load_db()
        u = get_user(db, user_id)
        f = u.get("filters", default_filters())

        for k in f.keys():
            if k in payload:
                f[k] = payload[k]

        u["filters"] = f
        # invalidate today choice (so refresh gives new matching dish)
        d = today()
        u.setdefault("daily_choice", {})
        u["daily_choice"].setdefault(d, {})
        u["daily_choice"][d].pop(u.get("lang", "uk"), None)

        save_db(db)

    return {"ok": True}


@app.post("/api/daily")
def api_daily(
    x_telegram_init_data: str = Header(default=""),
    force: int = Query(default=0)  # 1 = force new dish
):
    user_id = uid_from_init(x_telegram_init_data)

    with LOCK:
        db = load_db()
        u = get_user(db, user_id)
        apply_bonus(u)

        # after trial: charge ONLY once per day for "daily" (if not demo)
        if user_id != 0 and (not is_trial(u)):
            if u.get("daily_paid") != today():
                if not charge(u, "daily"):
                    save_db(db)
                    raise HTTPException(402, "NO_TOKENS")
                u["daily_paid"] = today()

        dish = pick_for_user(db, u, force=bool(force))
        save_db(db)

    return {"ok": True, "dish": dish, "demo": (user_id == 0)}


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

        # demo mode: no charge
        if user_id != 0:
            if not charge(u, action):
                save_db(db)
                raise HTTPException(402, "NO_TOKENS")

        dish = pick_for_user(db, u, force=False)
        save_db(db)

    if not dish:
        return {"ok": True, "data": None}

    if action == "ingredients":
        return {"ok": True, "data": dish.get("ingredients", [])}
    if action == "steps":
        return {"ok": True, "data": dish.get("steps", [])}
    return {"ok": True, "data": dish.get("time_total_min", 0)}
