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
# AI env залишаю, але тепер не використовується — генерація локальна і швидка
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
        "env": {"BOT_TOKEN": bool(BOT_TOKEN)},
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
            # генератор буде памʼятати останні N страв, щоб не повторювало
            "recent_titles": {"uk": [], "hr": [], "en": []},
        }
    u = db["users"][suid]
    u.setdefault("lang", "uk")
    u.setdefault("tokens", 15)
    u.setdefault("filters", default_filters())
    u.setdefault("daily_paid", "")
    u.setdefault("recent_titles", {"uk": [], "hr": [], "en": []})
    u["recent_titles"].setdefault("uk", [])
    u["recent_titles"].setdefault("hr", [])
    u["recent_titles"].setdefault("en", [])
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


# -------------------- FAST "MINI AI" GENERATOR (PERSONALIZED) --------------------

def _tr(lang: str, uk: str, hr: str, en: str) -> str:
    return {"uk": uk, "hr": hr, "en": en}.get(lang, uk)

def _norm_words(xs: List[str]) -> List[str]:
    return [re.sub(r"\s+", " ", (x or "").strip().lower()) for x in xs if (x or "").strip()]

def _contains_any(text: str, banned: List[str]) -> bool:
    t = (text or "").lower()
    return any(b in t for b in banned)

def _rng(uid: int, salt: str) -> random.Random:
    base = f"{uid}|{today()}|{salt}|{now().strftime('%H%M%S')}|{random.random()}"
    seed = int(hashlib.sha256(base.encode("utf-8")).hexdigest()[:16], 16)
    return random.Random(seed)

def _make_tags(diet: str, f: Dict[str, Any], base_tags: List[str]) -> List[str]:
    tags = set(base_tags)
    if diet in ("vegetarian", "vegan", "pescatarian"):
        tags.add(diet)
    for k in ("gluten_free", "lactose_free", "high_protein", "low_calorie"):
        if f.get(k):
            tags.add(k)
    return [t for t in tags if t in ALLOWED_TAGS]

def _choose_time(f: Dict[str, Any], rng: random.Random) -> int:
    mx = int(f.get("max_time", 0) or 0)
    if mx <= 0:
        return rng.choice([10, 12, 15, 18, 20, 25, 30, 35, 40, 45])
    # під max_time
    candidates = [t for t in [8,10,12,15,18,20,25,30,35,40,45,60] if t <= mx]
    return rng.choice(candidates) if candidates else mx

def _build_ingredients(lang: str, diet: str, f: Dict[str, Any], rng: random.Random) -> List[str]:
    # Категорії (просте, швидке, “різне”)
    proteins = {
        "vegan": [
            _tr(lang,"тофу","tofu","tofu"),
            _tr(lang,"квасоля","grah","beans"),
            _tr(lang,"нут","slanutak","chickpeas"),
            _tr(lang,"сочевиця","leća","lentils"),
        ],
        "vegetarian": [
            _tr(lang,"яйця","jaja","eggs"),
            _tr(lang,"сир","sir","cheese"),
            _tr(lang,"йогурт","jogurt","yogurt"),
        ],
        "pescatarian": [
            _tr(lang,"тунець","tuna","tuna"),
            _tr(lang,"лосось","losos","salmon"),
            _tr(lang,"сардини","srdele","sardines"),
        ],
        "any": [
            _tr(lang,"курка","piletina","chicken"),
            _tr(lang,"індичка","puretina","turkey"),
            _tr(lang,"тунець","tuna","tuna"),
            _tr(lang,"яйця","jaja","eggs"),
            _tr(lang,"тофу","tofu","tofu"),
        ],
    }

    carbs_gf = [
        _tr(lang,"рис","riža","rice"),
        _tr(lang,"картопля","krumpir","potato"),
        _tr(lang,"гречка","heljda","buckwheat"),
        _tr(lang,"кіноа","kvinoja","quinoa"),
    ]
    carbs = carbs_gf + [
        _tr(lang,"паста","tjestenina","pasta"),
        _tr(lang,"тортилья","tortilja","tortilla"),
        _tr(lang,"хліб","kruh","bread"),
    ]

    veggies = [
        _tr(lang,"помідор","rajčica","tomato"),
        _tr(lang,"огірок","krastavac","cucumber"),
        _tr(lang,"перець","paprika","pepper"),
        _tr(lang,"шпинат","špinat","spinach"),
        _tr(lang,"морква","mrkva","carrot"),
        _tr(lang,"цибуля","luk","onion"),
        _tr(lang,"броколі","brokula","broccoli"),
        _tr(lang,"гриби","gljive","mushrooms"),
    ]

    sauces = [
        _tr(lang,"оливкова олія","maslinovo ulje","olive oil"),
        _tr(lang,"соєвий соус","soja umak","soy sauce"),
        _tr(lang,"томатний соус","umak od rajčice","tomato sauce"),
        _tr(lang,"йогуртовий соус","jogurt umak","yogurt sauce"),
        _tr(lang,"лимонний сік","limunov sok","lemon juice"),
    ]

    # exclude
    excl = _norm_words(f.get("exclude") or [])
    # diet resolve
    if diet not in ("vegetarian","vegan","pescatarian"):
        diet = "any"

    # lactose_free: avoid dairy items by choosing sauce that isn't yogurt/cheese, and not picking dairy protein
    p_list = proteins[diet][:]
    if f.get("lactose_free"):
        # remove dairy
        p_list = [p for p in p_list if not _contains_any(p, ["сир","йогурт","cheese","yogurt","sir","jogurt"])]
        sauces2 = [s for s in sauces if not _contains_any(s, ["йогурт","jogurt","yogurt"])]
    else:
        sauces2 = sauces[:]

    # gluten_free: choose gf carbs
    carb_list = carbs_gf[:] if f.get("gluten_free") else carbs[:]

    # pick items with exclude filtering
    def pick_from(options: List[str]) -> str:
        opts = [o for o in options if (o and not _contains_any(o, excl))]
        if not opts:
            return ""
        return rng.choice(opts)

    prot = pick_from(p_list) or pick_from(proteins["any"])
    carb = pick_from(carb_list)
    v1 = pick_from(veggies)
    v2 = pick_from(veggies)
    sauce = pick_from(sauces2)

    base = [prot, carb, v1, v2, sauce,
            _tr(lang,"сіль","sol","salt"),
            _tr(lang,"перець","papar","pepper")]

    # remove empties and duplicates
    out = []
    seen = set()
    for x in base:
        if not x:
            continue
        key = x.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(x)

    # final exclude check
    blob = " ".join([x.lower() for x in out])
    if any(e in blob for e in excl):
        # якщо раптом щось прослизнуло — прибрати
        out = [x for x in out if not _contains_any(x, excl)]

    return out

def _build_title(lang: str, diet: str, f: Dict[str, Any], ingredients: List[str], rng: random.Random) -> str:
    # templates based on chosen ingredients
    prot = ingredients[0] if ingredients else _tr(lang,"страва","jelo","dish")
    carb = ""
    for x in ingredients[1:]:
        if x.lower() in ["рис","riža","rice","паста","tjestenina","pasta","гречка","heljda","buckwheat","картопля","krumpir","potato","тортилья","tortilja","tortilla","хліб","kruh","bread","кіноа","kvinoja","quinoa"]:
            carb = x
            break

    ideas = [
        _tr(lang, f"Швидка страва: {prot}", f"Brzo jelo: {prot}", f"Quick dish: {prot}"),
        _tr(lang, f"{prot} боул", f"{prot} bowl", f"{prot} bowl"),
        _tr(lang, f"Салат з {prot}", f"Salata s {prot}", f"Salad with {prot}"),
        _tr(lang, f"Врап з {prot}", f"Wrap s {prot}", f"Wrap with {prot}"),
        _tr(lang, f"{prot} + овочі", f"{prot} + povrće", f"{prot} + veggies"),
    ]
    if carb:
        ideas += [
            _tr(lang, f"{prot} з {carb}", f"{prot} s {carb}", f"{prot} with {carb}"),
            _tr(lang, f"{carb} з {prot}", f"{carb} s {prot}", f"{carb} with {prot}"),
        ]

    # extra variations
    if f.get("high_protein"):
        ideas.append(_tr(lang, f"High-protein: {prot}", f"High-protein: {prot}", f"High-protein: {prot}"))
    if f.get("low_calorie"):
        ideas.append(_tr(lang, f"Легка страва: {prot}", f"Lagano jelo: {prot}", f"Light dish: {prot}"))

    return rng.choice(ideas)[:80]

def _build_steps(lang: str, f: Dict[str, Any], rng: random.Random) -> List[str]:
    # simple, universal steps, small randomness
    s = [
        _tr(lang, "Підготуй інгредієнти (поріж овочі).", "Pripremi sastojke (nareži povrće).", "Prep ingredients (chop veggies)."),
        _tr(lang, "Приготуй основний продукт (або відкрий консерву).", "Pripremi glavni sastojak (ili otvori konzervu).", "Cook the main ingredient (or open can)."),
        _tr(lang, "Змішай усе та додай соус і спеції.", "Sve pomiješaj i dodaj umak i začine.", "Mix everything, add sauce & spices."),
        _tr(lang, "Подавай одразу.", "Posluži odmah.", "Serve immediately."),
    ]
    # swap one step sometimes
    if rng.random() < 0.35:
        s[1] = _tr(lang, "Підігрій на пательні 5–7 хв.", "Zagrij na tavi 5–7 min.", "Heat in a pan for 5–7 min.")
    if rng.random() < 0.25:
        s.insert(2, _tr(lang, "Спробуй на смак і підкоригуй сіль/перець.", "Probaj i dotjeraj sol/papar.", "Taste and adjust salt/pepper."))
    return s[:6]

def generate_dish_for_user(uid: int, lang: str, f: Dict[str, Any], recent_titles: List[str]) -> Optional[Dict[str, Any]]:
    rng = _rng(uid, f"GEN|{lang}")
    diet = (f.get("diet") or "any")
    if diet not in ("any","vegetarian","vegan","pescatarian"):
        diet = "any"

    # зробимо кілька спроб, щоб:
    # 1) не повторювалось
    # 2) пройшло фільтри
    # 3) не містило exclude
    max_tries = 40
    recent_set = set((x or "").strip().lower() for x in (recent_titles or [])[-12:])

    for _ in range(max_tries):
        time_total = _choose_time(f, rng)
        ingredients = _build_ingredients(lang, diet, f, rng)
        title = _build_title(lang, diet, f, ingredients, rng)
        why = _tr(
            lang,
            "Згенеровано під твої фільтри — швидко й просто.",
            "Generirano prema tvojim filterima — brzo i jednostavno.",
            "Generated for your filters — fast and simple."
        )
        tags = _make_tags(diet, f, base_tags=["quick"])

        dish = {
            "title": title,
            "why": why,
            "ingredients": ingredients,
            "steps": _build_steps(lang, f, rng),
            "time_total_min": int(time_total),
            "tags": tags,
        }

        # filter check
        if not dish_matches_filters(dish, f):
            continue

        # avoid repeats
        if dish["title"].strip().lower() in recent_set:
            continue

        return dish

    # якщо дуже строгі фільтри й нічого не вийшло
    return None


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
        return 0  # demo
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

        save_db(db)

    return {"ok": True}


@app.post("/api/daily")
def api_daily(
    x_telegram_init_data: str = Header(default=""),
    force: int = Query(default=0)  # 1 = завжди нова страва
):
    user_id = uid_from_init(x_telegram_init_data)

    with LOCK:
        db = load_db()
        u = get_user(db, user_id)
        apply_bonus(u)

        # charge once per day for daily (not demo)
        if user_id != 0 and (not is_trial(u)):
            if u.get("daily_paid") != today():
                if not charge(u, "daily"):
                    save_db(db)
                    raise HTTPException(402, "NO_TOKENS")
                u["daily_paid"] = today()

        lang = u.get("lang", "uk")
        f = u.get("filters", default_filters())

        # якщо НЕ force — робимо "страву дня" стабільну (одна на день)
        # якщо force=1 — генеруємо кожен раз нову
        dkey = today()
        db.setdefault("daily", {}).setdefault(dkey, {})
        userkey = str(user_id)
        db["daily"][dkey].setdefault(userkey, {})
        db["daily"][dkey][userkey].setdefault(lang, {})

        if not force and db["daily"][dkey][userkey][lang].get("dish"):
            dish = db["daily"][dkey][userkey][lang]["dish"]
        else:
            recent = u.get("recent_titles", {}).get(lang, [])
            dish = generate_dish_for_user(user_id, lang, f, recent)

            # save dish for today (only if not force OR if you want last generated)
            db["daily"][dkey][userkey][lang]["dish"] = dish

            # remember recent titles to avoid repeats
            if dish and dish.get("title"):
                u["recent_titles"][lang].append(dish["title"])
                u["recent_titles"][lang] = u["recent_titles"][lang][-25:]

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

        if user_id != 0:
            if not charge(u, action):
                save_db(db)
                raise HTTPException(402, "NO_TOKENS")

        lang = u.get("lang", "uk")
        dkey = today()
        userkey = str(user_id)

        # беремо останню згенеровану "daily" (без force)
        dish = None
        try:
            dish = db.get("daily", {}).get(dkey, {}).get(userkey, {}).get(lang, {}).get("dish")
        except Exception:
            dish = None

        # якщо ще нема — згенерувати раз
        if not dish:
            f = u.get("filters", default_filters())
            recent = u.get("recent_titles", {}).get(lang, [])
            dish = generate_dish_for_user(user_id, lang, f, recent)
            db.setdefault("daily", {}).setdefault(dkey, {})
            db["daily"][dkey].setdefault(userkey, {})
            db["daily"][dkey][userkey].setdefault(lang, {})
            db["daily"][dkey][userkey][lang]["dish"] = dish

            if dish and dish.get("title"):
                u["recent_titles"][lang].append(dish["title"])
                u["recent_titles"][lang] = u["recent_titles"][lang][-25:]

        save_db(db)

    if not dish:
        return {"ok": True, "data": None}

    if action == "ingredients":
        return {"ok": True, "data": dish.get("ingredients", [])}
    if action == "steps":
        return {"ok": True, "data": dish.get("steps", [])}
    return {"ok": True, "data": dish.get("time_total_min", 0)}
