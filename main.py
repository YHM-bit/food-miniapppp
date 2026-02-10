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
AI_API_KEY = os.environ.get("AI_API_KEY")
AI_ENDPOINT = os.environ.get("AI_ENDPOINT", "https://models.github.ai/inference")
AI_MODEL = os.environ.get("AI_MODEL", "openai/gpt-4o-mini")

# На проді можна прибрати ці RuntimeError, але для дебагу краще хай падає
if not BOT_TOKEN:
    raise RuntimeError("Set BOT_TOKEN env var.")
if not AI_API_KEY:
    raise RuntimeError("Set AI_API_KEY env var.")

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
        "files": {
            "web/index.html": os.path.exists("web/index.html"),
            "index.html": os.path.exists("index.html"),
        }
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
            # для сумісності з твоїм старим ботом
            "favorites": [],
            "uploads": [],
        }
    u = db["users"][suid]
    u.setdefault("favorites", [])
    u.setdefault("uploads", [])
    u.setdefault("daily_paid", "")
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

def ai_pool(lang: str, forbidden: List[str], n: int = 10) -> List[Dict[str, Any]]:
    """
    Tries AI first. If AI fails (403/no access/network/etc) -> returns a safe fallback pool.
    This keeps the Mini App working.
    """
    def fallback_pool() -> List[Dict[str, Any]]:
        # Minimal but valid pool (tags must be in ALLOWED_TAGS)
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
        # expand to n by repeating with slight variations
        out = []
        i = 0
        while len(out) < n:
            item = base[i % len(base)].copy()
            item["title"] = f"{item['title']} #{(i//len(base))+1}" if i >= len(base) else item["title"]
            out.append(item)
            i += 1
        return out[:n]

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
            title = str(it.get("title","")).strip()
            if not title:
                continue
            tags = [t for t in (it.get("tags") or []) if t in ALLOWED_TAGS]
            out.append({
                "title": title[:80],
                "why": str(it.get("why",""))[:240],
                "ingredients": [str(x)[:120] for x in (it.get("ingredients") or []) if str(x).strip()],
                "steps": [str(x)[:200] for x in (it.get("steps") or []) if str(x).strip()],
                "time_total_min": int(it.get("time_total_min",30) or 30),
                "tags": tags,
            })
        if len(out) < 3:
            return fallback_pool()
        return out[:n]
    except Exception as e:
        # Any failure (403/no access/network/json issues) -> fallback
        return fallback_pool()


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


@app.get("/api/status")
def api_status(x_telegram_init_data: str = Header(default="")):
    # ✅ DEMO mode (when opened in browser, no Telegram initData)
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
    # ✅ DEMO mode (browser)
    if not x_telegram_init_data:
        # show a static demo dish so UI is never empty
        dish = {
            "title": "Demo: Омлет за 10 хв",
            "why": "Демо-режим (відкрито не в Telegram). Відкрий Mini App через бота для персоналізації.",
            "ingredients": ["2 яйця", "сіль", "перець", "трохи масла"],
            "steps": ["Збий яйця з сіллю.", "Розігрій пательню з маслом.", "Вилий яйця, готуй 2-3 хв."],
            "time_total_min": 10,
            "tags": ["quick", "high_protein"]
        }
        return {"ok": True, "dish": dish, "demo": True}

    user_id = uid_from_init(x_telegram_init_data)
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

