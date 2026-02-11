import os, json, re, hmac, hashlib, urllib.request, urllib.parse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from urllib.parse import parse_qsl
from typing import Dict, Any, List, Optional
from threading import Lock

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

TZ = ZoneInfo("Europe/Uzhgorod")
DB_PATH = "db.json"

BOT_TOKEN = os.environ.get("BOT_TOKEN")

# AI optional now (mini-app must still work without it)
AI_API_KEY = os.environ.get("AI_API_KEY", "")
AI_ENDPOINT = os.environ.get("AI_ENDPOINT", "https://models.github.ai/inference")
AI_MODEL = os.environ.get("AI_MODEL", "openai/gpt-4o-mini")

# Webhook secret (recommended)
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")

if not BOT_TOKEN:
    raise RuntimeError("Set BOT_TOKEN env var.")

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

# ⭐ Stars packs (you can change)
STARS_PACKS = {
    # pack_id: {stars_price, tokens_granted, title}
    "p50":  {"stars": 50,  "tokens": 80,  "title": "Starter Pack"},
    "p100": {"stars": 100, "tokens": 180, "title": "Plus Pack"},
    "p250": {"stars": 250, "tokens": 520, "title": "Pro Pack"},
}

app = FastAPI()


# -------------------- utils --------------------

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
    u.setdefault("tokens", 0)
    return u

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

def strip_fences(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^```(json)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()
    return s

def lang_name(lang: str) -> str:
    return {"uk": "Ukrainian", "hr": "Croatian", "en": "English"}.get(lang, "Ukrainian")

def stable_pick_index(uid: int, day: str, n: int) -> int:
    # deterministic “random” index for uid+date
    h = hashlib.sha256(f"{uid}:{day}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) % max(1, n)

# -------------------- AI / fallback pool --------------------

def fallback_pool(lang: str, n: int = 10) -> List[Dict[str, Any]]:
    # 10 quick recipes, localized titles/why, English tags only.
    if lang == "hr":
        items = [
            ("Omlet sa sirom", "Brzo i jednostavno.", ["jaja", "sir", "sol", "papar"], ["Umuti jaja", "Dodaj sir", "Prži 5–7 min"], 10, ["vegetarian","quick","high_protein"]),
            ("Tost s avokadom", "Gotovo za par minuta.", ["kruh", "avokado", "sol", "limun"], ["Zgnječi avokado", "Namaži na tost", "Začini"], 7, ["vegan","quick"]),
            ("Salata s tunom", "Puno proteina.", ["tuna", "krastavac", "rajčica", "maslinovo ulje"], ["Nareži povrće", "Dodaj tunu", "Začini"], 12, ["pescatarian","high_protein","quick"]),
            ("Pasta aglio e olio", "Minimalno sastojaka.", ["tjestenina", "češnjak", "maslinovo ulje"], ["Skuhaj tjesteninu", "U tavi ulje+češnjak", "Pomiješaj"], 15, ["vegetarian","quick"]),
            ("Zobena kaša s bananom", "Lagani doručak.", ["zob", "banana", "mlijeko/voda"], ["Kuhaj 5 min", "Dodaj bananu"], 8, ["vegetarian","quick"]),
            ("Riža s povrćem", "Jednostavno i zasitno.", ["riža", "mrkva", "grašak", "soja umak"], ["Skuhaj rižu", "Dodaj povrće", "Začini"], 25, ["vegan"]),
            ("Jogurt bowl", "Bez kuhanja.", ["jogurt", "voće", "med"], ["Stavi u zdjelu", "Dodaj voće", "Prelij medom"], 5, ["vegetarian","quick"]),
            ("Piletina na tavi", "Proteinski obrok.", ["piletina", "sol", "papar", "ulje"], ["Začini", "Ispeci 10–12 min"], 15, ["high_protein","quick"]),
            ("Salata od slanutka", "Vegan + proteini.", ["slanutak", "luk", "rajčica", "limun"], ["Pomiješaj", "Začini"], 10, ["vegan","high_protein","quick"]),
            ("Juha od rajčice", "Toplo i brzo.", ["rajčica", "temeljac", "sol"], ["Zagrij", "Kuhaj 10 min"], 15, ["vegan"]),
        ]
    elif lang == "en":
        items = [
            ("Cheese omelet", "Fast and simple.", ["eggs", "cheese", "salt", "pepper"], ["Beat eggs", "Add cheese", "Fry 5–7 min"], 10, ["vegetarian","quick","high_protein"]),
            ("Avocado toast", "Ready in minutes.", ["bread", "avocado", "salt", "lemon"], ["Mash avocado", "Spread on toast", "Season"], 7, ["vegan","quick"]),
            ("Tuna salad", "High protein.", ["tuna", "cucumber", "tomato", "olive oil"], ["Chop veggies", "Add tuna", "Dress"], 12, ["pescatarian","high_protein","quick"]),
            ("Pasta aglio e olio", "Minimal ingredients.", ["pasta", "garlic", "olive oil"], ["Boil pasta", "Warm oil+garlic", "Mix"], 15, ["vegetarian","quick"]),
            ("Oatmeal with banana", "Easy breakfast.", ["oats", "banana", "milk/water"], ["Cook 5 min", "Add banana"], 8, ["vegetarian","quick"]),
            ("Veggie rice bowl", "Simple and filling.", ["rice", "carrot", "peas", "soy sauce"], ["Cook rice", "Add veggies", "Season"], 25, ["vegan"]),
            ("Yogurt bowl", "No cooking.", ["yogurt", "fruit", "honey"], ["Add yogurt", "Add fruit", "Drizzle honey"], 5, ["vegetarian","quick"]),
            ("Pan chicken", "Protein meal.", ["chicken", "salt", "pepper", "oil"], ["Season", "Pan-fry 10–12 min"], 15, ["high_protein","quick"]),
            ("Chickpea salad", "Vegan + protein.", ["chickpeas", "onion", "tomato", "lemon"], ["Mix", "Season"], 10, ["vegan","high_protein","quick"]),
            ("Tomato soup", "Warm and quick.", ["tomatoes", "broth", "salt"], ["Heat", "Simmer 10 min"], 15, ["vegan"]),
        ]
    else:
        items = [
            ("Омлет з сиром", "Швидко і просто.", ["яйця", "сир", "сіль", "перець"], ["Збий яйця", "Додай сир", "Посмаж 5–7 хв"], 10, ["vegetarian","quick","high_protein"]),
            ("Тост з авокадо", "Готово за кілька хвилин.", ["хліб", "авокадо", "сіль", "лимон"], ["Розімни авокадо", "Намаж на тост", "Приправ"], 7, ["vegan","quick"]),
            ("Салат з тунцем", "Багато білка.", ["тунець", "огірок", "помідор", "оливкова олія"], ["Наріж овочі", "Додай тунець", "Заправ"], 12, ["pescatarian","high_protein","quick"]),
            ("Паста aglio e olio", "Мінімум продуктів.", ["паста", "часник", "оливкова олія"], ["Відвари пасту", "У сковорідці олія+часник", "Змішай"], 15, ["vegetarian","quick"]),
            ("Вівсянка з бананом", "Легкий сніданок.", ["вівсянка", "банан", "молоко/вода"], ["Вари 5 хв", "Додай банан"], 8, ["vegetarian","quick"]),
            ("Рис з овочами", "Просто і ситно.", ["рис", "морква", "горошок", "соєвий соус"], ["Звари рис", "Додай овочі", "Приправ"], 25, ["vegan"]),
            ("Йогурт-бол", "Без готування.", ["йогурт", "фрукти", "мед"], ["Поклади йогурт", "Додай фрукти", "Полий медом"], 5, ["vegetarian","quick"]),
            ("Курка на пательні", "Протеїново.", ["курка", "сіль", "перець", "олія"], ["Приправ", "Обсмаж 10–12 хв"], 15, ["high_protein","quick"]),
            ("Салат з нутом", "Веган + білок.", ["нут", "цибуля", "помідор", "лимон"], ["Змішай", "Приправ"], 10, ["vegan","high_protein","quick"]),
            ("Томатний суп", "Тепло і швидко.", ["помідори", "бульйон", "сіль"], ["Підігрій", "Провари 10 хв"], 15, ["vegan"]),
        ]

    out = []
    for t, why, ing, steps, mins, tags in items[:n]:
        out.append({
            "title": t,
            "why": why,
            "ingredients": ing,
            "steps": steps,
            "time_total_min": mins,
            "tags": [x for x in tags if x in ALLOWED_TAGS],
        })
    return out

def ai_pool(lang: str, forbidden: List[str], n: int = 10) -> List[Dict[str, Any]]:
    # If AI not configured -> fallback (fast)
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
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=1.0,
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
                "why": str(it.get("why", ""))[:240],
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

def pick_daily(db: Dict[str, Any], uid: int, u: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    lang = u.get("lang", "uk")
    pool = get_pool(db, lang)
    f = u.get("filters", default_filters())
    matches = [d for d in pool if dish_matches_filters(d, f)]
    if not matches:
        return None
    idx = stable_pick_index(uid, today(), len(matches))
    return matches[idx]


# -------------------- Telegram initData verify --------------------

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

def tg_api(method: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimal Telegram Bot API call without extra deps.
    """
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"
    data = urllib.parse.urlencode(payload, doseq=True).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
        j = json.loads(raw)
        if not j.get("ok"):
            raise RuntimeError(f"Telegram API error: {raw}")
        return j


# -------------------- basic routes --------------------

@app.get("/health")
def health():
    return {
        "ok": True,
        "time": now().isoformat(),
        "has_index_web": os.path.exists("web/index.html"),
        "has_index_root": os.path.exists("index.html"),
        "has_ai_key": bool(AI_API_KEY),
        "stars_packs": STARS_PACKS,
        "webhook_secret_set": bool(WEBHOOK_SECRET),
    }

@app.get("/", response_class=HTMLResponse)
def root():
    # Mini App HTML
    if os.path.exists("web/index.html"):
        return FileResponse("web/index.html")
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return HTMLResponse("<h2>index.html not found</h2><p>Expected: web/index.html</p>", status_code=500)


# -------------------- API for Mini App --------------------

@app.get("/api/status")
def api_status(x_telegram_init_data: str = Header(default="")):
    # demo mode for browser
    if not x_telegram_init_data:
        return {
            "lang": "uk",
            "trial": True,
            "trial_days_left": TRIAL_DAYS,
            "tokens": 15,
            "filters": default_filters(),
            "demo": True,
            "packs": STARS_PACKS,
        }

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
        "tokens": u["tokens"],
        "filters": u["filters"],
        "demo": False,
        "packs": STARS_PACKS,
    }

@app.post("/api/lang")
async def api_lang(payload: Dict[str, Any], x_telegram_init_data: str = Header(default="")):
    user_id = uid_from_init(x_telegram_init_data)
    lang = payload.get("lang", "uk")
    if lang not in ("uk", "hr", "en"):
        lang = "uk"
    with LOCK:
        db = load_db()
        u = get_user(db, user_id)
        u["lang"] = lang
        save_db(db)
    return {"ok": True}

@app.post("/api/filters")
async def api_filters(payload: Dict[str, Any], x_telegram_init_data: str = Header(default="")):
    user_id = uid_from_init(x_telegram_init_data)
    with LOCK:
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
    # demo mode for browser
    if not x_telegram_init_data:
        dish = fallback_pool("uk", 10)[stable_pick_index(1, today(), 10)]
        dish["title"] = "DEMO: " + dish["title"]
        dish["why"] = "Демо-режим (не Telegram). Відкрий Mini App через бота для персоналізації."
        return {"ok": True, "dish": dish, "demo": True}

    user_id = uid_from_init(x_telegram_init_data)
    with LOCK:
        db = load_db()
        u = get_user(db, user_id)
        apply_bonus(u)

        # after trial: charge daily only once/day
        if not is_trial(u):
            if u.get("daily_paid") != today():
                if not charge(u, "daily"):
                    save_db(db)
                    raise HTTPException(402, "NO_TOKENS")
                u["daily_paid"] = today()

        dish = pick_daily(db, user_id, u)
        save_db(db)

    return {"ok": True, "dish": dish, "demo": False}

@app.post("/api/action")
async def api_action(payload: Dict[str, Any], x_telegram_init_data: str = Header(default="")):
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

        dish = pick_daily(db, user_id, u)
        save_db(db)

    if not dish:
        return {"ok": True, "data": None}
    if action == "ingredients":
        return {"ok": True, "data": dish.get("ingredients", [])}
    if action == "steps":
        return {"ok": True, "data": dish.get("steps", [])}
    return {"ok": True, "data": dish.get("time_total_min", 0)}


# -------------------- Stars: create invoice link --------------------

@app.post("/api/buy")
async def api_buy(payload: Dict[str, Any], x_telegram_init_data: str = Header(default="")):
    """
    Returns invoice link for Telegram Stars purchase.
    Frontend should call tg.openInvoice(link).
    """
    user_id = uid_from_init(x_telegram_init_data)

    pack_id = str(payload.get("pack", "p50"))
    if pack_id not in STARS_PACKS:
        raise HTTPException(400, "bad pack")

    pack = STARS_PACKS[pack_id]
    stars_price = int(pack["stars"])
    tokens_granted = int(pack["tokens"])

    # build invoice payload (must be <= 128 bytes ideally, keep short)
    invoice_payload = f"buy:{pack_id}:{user_id}:{int(now().timestamp())}"

    title = pack.get("title", "Token Pack")
    description = f"Adds {tokens_granted} tokens to your Cook Today wallet."

    # ⭐ Stars currency is XTR. provider_token MUST be omitted. :contentReference[oaicite:0]{index=0}
    prices_json = json.dumps([{"label": f"{tokens_granted} tokens", "amount": stars_price}], ensure_ascii=False)

    try:
        res = tg_api("createInvoiceLink", {
            "title": title,
            "description": description,
            "payload": invoice_payload,
            "currency": "XTR",
            "prices": prices_json,
        })
    except Exception as e:
        raise HTTPException(500, f"invoice_error: {e}")

    link = res["result"]
    return {"ok": True, "link": link, "pack": pack_id}


# -------------------- Stars: webhook receiver --------------------

@app.post("/tg/webhook")
async def tg_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: str = Header(default="")
):
    # Optional secret check
    if WEBHOOK_SECRET:
        if x_telegram_bot_api_secret_token != WEBHOOK_SECRET:
            return JSONResponse({"ok": True}, status_code=200)

    update = await request.json()

    # 1) pre_checkout_query -> MUST answer OK, or payment will fail
    if "pre_checkout_query" in update:
        pcq = update["pre_checkout_query"]
        try:
            tg_api("answerPreCheckoutQuery", {
                "pre_checkout_query_id": pcq["id"],
                "ok": "true",
            })
        except Exception:
            # Even if answering fails, return 200 so Telegram doesn't spam retries forever.
            pass
        return {"ok": True}

    # 2) successful_payment -> add tokens
    msg = update.get("message") or update.get("business_message")
    if msg and msg.get("successful_payment"):
        sp = msg["successful_payment"]

        # Stars payment has currency XTR. :contentReference[oaicite:1]{index=1}
        currency = sp.get("currency")
        invoice_payload = sp.get("invoice_payload", "")
        if currency == "XTR" and invoice_payload.startswith("buy:"):
            try:
                _, pack_id, uid_str, _ts = invoice_payload.split(":", 3)
                uid_int = int(uid_str)
            except Exception:
                return {"ok": True}

            if pack_id in STARS_PACKS:
                tokens_granted = int(STARS_PACKS[pack_id]["tokens"])
                with LOCK:
                    db = load_db()
                    u = get_user(db, uid_int)
                    u["tokens"] = int(u.get("tokens", 0)) + tokens_granted
                    save_db(db)

                # optional notify
                try:
                    chat_id = msg["chat"]["id"]
                    tg_api("sendMessage", {
                        "chat_id": chat_id,
                        "text": f"✅ Payment received. +{tokens_granted} tokens added!",
                    })
                except Exception:
                    pass

        return {"ok": True}

    return {"ok": True}
