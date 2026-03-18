import os, json, re, hmac, hashlib, random
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


if not BOT_TOKEN:
    raise RuntimeError("Set BOT_TOKEN env var.")

LOCK = Lock()

TRIAL_DAYS = 7
DAILY_BONUS = 5
FREE_CAP = 30

COSTS = {"daily": 1, "ingredients": 1, "steps": 2, "time": 1}

ALLOWED_TAGS = {
    "vegetarian", "vegan", "pescatarian",
    "Gluten free", "Lactose free",
    "High protein", "Low calorie",
    "quick",
}

app = FastAPI()




@app.get("/health")
def health():
    return {
        "ok": True,
        "files": {
            "web/index.html": os.path.exists("web/index.html"),
            "index.html": os.path.exists("index.html"),
            "db.json": os.path.exists("db.json"),
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




def now() -> datetime:
    return datetime.now(TZ)

def today() -> str:
    return now().date().isoformat()

def load_db() -> Dict[str, Any]:
    if not os.path.exists(DB_PATH):
        return {"users": {}, "daily": {}, "used_titles": {"ua": [], "hr": [], "en": []}}
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_db(db: Dict[str, Any]) -> None:
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

def default_filters() -> Dict[str, Any]:
    return {
        "diet": "any",              
        "Gluten free": False,
        "Lactose free": False,
        "High protein": False,
        "Low calorie": False,
        "Max time": 0,              
        "exclude": [],              
    }

def get_user(db: Dict[str, Any], uid: int) -> Dict[str, Any]:
    suid = str(uid)
    if suid not in db["users"]:
        created = now()
        db["users"][suid] = {
            "lang": "ua",
            "tokens": 15,
            "created_at": created.isoformat(),
            "trial_until": (created + timedelta(days=TRIAL_DAYS)).isoformat(),
            "last_bonus": "",
            "filters": default_filters(),
            "daily_paid": "",
            "favorites": [],
            "uploads": [],
            "last_dish_sig": "",  
        }
    u = db["users"][suid]
    u.setdefault("favorites", [])
    u.setdefault("uploads", [])
    u.setdefault("daily_paid", "")
    u.setdefault("last_dish_sig", "")
    if "filters" not in u:
        u["filters"] = default_filters()
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




def _tr(lang: str, ua: str, hr: str, en: str) -> str:
    if lang == "hr":
        return hr
    if lang == "en":
        return en
    return ua

def _norm_words(items: List[str]) -> List[str]:
    out = []
    for x in items or []:
        s = str(x).strip().lower()
        if not s:
            continue
        out.append(s)
    return out

def _contains_any(text: str, words: List[str]) -> bool:
    t = (text or "").lower()
    for w in words:
        if w and w in t:
            return True
    return False

def _qty(lang: str, a_ua: str, a_hr: str, a_en: str) -> str:
    return _tr(lang, a_ua, a_hr, a_en)

def _fmt_item(name: str, qty: str) -> str:
    name = (name or "").strip()
    qty = (qty or "").strip()
    if not name:
        return ""
    return f"{name} — {qty}" if qty else name

def _avoid_exclude(items: List[str], excl: List[str]) -> List[str]:
    out = []
    for it in items:
        if not it:
            continue
        if _contains_any(it, excl):
            continue
        out.append(it)
    return out

def _pick_distinct(rng: random.Random, options: List[str], k: int) -> List[str]:
    opts = [o for o in options if o]
    rng.shuffle(opts)
    return opts[:k]




def dish_matches_filters(d: Dict[str, Any], f: Dict[str, Any]) -> bool:
    tags = set(d.get("tags", []))
    diet = f.get("diet", "any")
    if diet != "any" and diet not in tags:
        return False
    for k in ("Gluten free", "Lactose free", "High protein", "Low calorie"):
        if f.get(k) and k not in tags:
            return False
    Max time = int(f.get("Max time", 0) or 0)
    if Max time and int(d.get("time_total_min", 10_000)) > Max time:
        return False
    excl = [x.strip().lower() for x in (f.get("exclude") or []) if str(x).strip()]
    if excl:
        blob = " ".join([str(x).lower() for x in d.get("ingredients", [])])
        if any(w in blob for w in excl):
            return False
    return True




def _build_title(lang: str, diet: str, f: Dict[str, Any], rng: random.Random) -> str:
    quick = f.get("Max time", 0) and int(f.get("Max time", 0)) <= 15
    if quick:
        base = _tr(lang, "Швидка страва", "Brzo jelo", "Quick dish")
    else:
        base = _tr(lang, "Страва дня", "Jelo dana", "Dish of the day")

    variants = [
        _tr(lang, "боул", "bowl", "bowl"),
        _tr(lang, "салат", "salata", "salad"),
        _tr(lang, "сковорідка", "tava", "pan"),
        _tr(lang, "паста", "tjestenina", "pasta"),
        _tr(lang, "рис", "riža", "rice"),
    ]

    
    if f.get("Gluten free"):
        variants = [v for v in variants if v not in [_tr(lang, "паста","tjestenina","pasta")]]

    v = rng.choice(variants)
    return f"{base}: {v}".strip()


def _build_ingredients(lang: str, diet: str, f: Dict[str, Any], rng: random.Random) -> List[str]:
    """
    Returns ingredients with quantities. Still respects gluten/lactose/exclude.
    """
    excl = _norm_words(f.get("exclude") or [])

    def pick_from(options: List[str]) -> str:
        opts = [o for o in options if (o and not _contains_any(o, excl))]
        return rng.choice(opts) if opts else ""

    
    proteins = {
        "vegan": [
            _tr(lang,"тофу","tofu","tofu"),
            _tr(lang,"квасоля (консерва)","grah (konzerva)","beans (canned)"),
            _tr(lang,"нут (консерва)","slanutak (konzerva)","chickpeas (canned)"),
            _tr(lang,"сочевиця","leća","lentils"),
        ],
        "vegetarian": [
            _tr(lang,"яйця","jaja","eggs"),
            _tr(lang,"сир","sir","cheese"),
            _tr(lang,"грецький йогурт","grčki jogurt","Greek yogurt"),
        ],
        "pescatarian": [
            _tr(lang,"тунець (консерва)","tuna (konzerva)","tuna (canned)"),
            _tr(lang,"лосось","losos","salmon"),
            _tr(lang,"сардини (консерва)","srdele (konzerva)","sardines (canned)"),
        ],
        "any": [
            _tr(lang,"куряче філе","pileći file","chicken breast"),
            _tr(lang,"індичка","puretina","turkey"),
            _tr(lang,"тунець (консерва)","tuna (konzerva)","tuna (canned)"),
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
        _tr(lang,"кукурудза (консерва)","kukuruz (konzerva)","corn (canned)"),
    ]

    sauces = [
        _tr(lang,"оливкова олія","maslinovo ulje","olive oil"),
        _tr(lang,"соєвий соус","soja umak","soy sauce"),
        _tr(lang,"томатний соус","umak od rajčice","tomato sauce"),
        _tr(lang,"лимонний сік","limunov sok","lemon juice"),
    ]
    dairy_sauces = [
        _tr(lang,"йогуртовий соус","jogurt umak","yogurt sauce"),
        _tr(lang,"сметана/йогурт","vrhnje/jogurt","sour cream/yogurt"),
    ]

    if diet not in ("vegetarian","vegan","pescatarian"):
        diet = "any"

   
    p_list = proteins[diet][:]
    if f.get("Lactose free"):
        p_list = [p for p in p_list if not _contains_any(p, ["сир","йогурт","cheese","yogurt","sir","jogurt","vrhnje"])]
        sauce_pool = sauces[:]
    else:
        sauce_pool = sauces + dairy_sauces

    
    carb_pool = carbs_gf[:] if f.get("Gluten free") else carbs[:]

    prot = pick_from(p_list) or pick_from(proteins["any"])
    carb = pick_from(carb_pool)
    veg_pick = _pick_distinct(rng, _avoid_exclude(veggies, excl), 3)
    sauce = pick_from(sauce_pool)

    
    protein_qty = _qty(lang, "120–180 г", "120–180 g", "120–180 g")
    if f.get("High protein"):
        protein_qty = _qty(lang, "180–250 г", "180–250 g", "180–250 g")

    carb_qty = _qty(lang, "60–80 г (сух.)", "60–80 g (suho)", "60–80 g (dry)")
    if f.get("Low calorie"):
        carb_qty = _qty(lang, "40–60 г (сух.)", "40–60 g (suho)", "40–60 g (dry)")

    veg_qty = _qty(lang, "1 шт", "1 kom", "1 pc")
    oil_qty = _qty(lang, "1 ст. л.", "1 žlica", "1 tbsp")
    sauce_qty = _qty(lang, "1–2 ст. л.", "1–2 žlice", "1–2 tbsp")

    
    if _contains_any(prot, ["яйц", "egg", "jaja"]):
        protein_qty = _qty(lang, "2–3 шт", "2–3 kom", "2–3 pcs")
    if _contains_any(prot, ["тунець", "tuna", "сардин", "srdele", "sardines"]):
        protein_qty = _qty(lang, "1 банка", "1 konzerva", "1 can")
    if _contains_any(prot, ["квасол", "grah", "beans", "нут", "slanutak", "chickpeas"]):
        protein_qty = _qty(lang, "1/2–1 банка", "1/2–1 konzerva", "1/2–1 can")
    if _contains_any(carb, ["картоп", "krumpir", "potato"]):
        carb_qty = _qty(lang, "2–3 шт", "2–3 kom", "2–3 pcs")
    if _contains_any(carb, ["хліб", "kruh", "bread"]):
        carb_qty = _qty(lang, "2 скибки", "2 kriške", "2 slices")
    if _contains_any(carb, ["тортиль", "tortil"]):
        carb_qty = _qty(lang, "1–2 шт", "1–2 kom", "1–2 pcs")

    items = []
    items.append(_fmt_item(prot, protein_qty))
    if carb:
        items.append(_fmt_item(carb, carb_qty))

    for v in veg_pick:
        vq = veg_qty
        if _contains_any(v, ["шпинат","špinat","spinach"]):
            vq = _qty(lang, "1 жменя", "1 šaka", "1 handful")
        if _contains_any(v, ["брокол","brok","broccoli"]):
            vq = _qty(lang, "150–200 г", "150–200 g", "150–200 g")
        if _contains_any(v, ["гриб","gljiv","mushroom"]):
            vq = _qty(lang, "150 г", "150 g", "150 g")
        items.append(_fmt_item(v, vq))

    if sauce:
        if _contains_any(sauce, ["оливкова", "maslinovo", "olive oil"]):
            items.append(_fmt_item(sauce, oil_qty))
        else:
            items.append(_fmt_item(sauce, sauce_qty))

    items.append(_fmt_item(_tr(lang,"сіль","sol","salt"), _qty(lang,"за смаком","po ukusu","to taste")))
    items.append(_fmt_item(_tr(lang,"перець","papar","pepper"), _qty(lang,"за смаком","po ukusu","to taste")))

    items = _avoid_exclude(items, excl)

    out, seen = [], set()
    for it in items:
        key = it.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(it)

    return out


def _build_steps(lang: str, f: Dict[str, Any], rng: random.Random) -> List[str]:
    """
    More detailed steps with micro-timing. Still universal and fast.
    """
    steps = []

    steps.append(_tr(
        lang,
        "1) Підготовка (2–4 хв): помий/наріж овочі. Якщо є консерва — злий рідину та промий.",
        "1) Priprema (2–4 min): operi/nareži povrće. Ako je konzerva — ocijedi i isperi.",
        "1) Prep (2–4 min): wash/chop veggies. If canned — drain and rinse."
    ))

    if f.get("Gluten free"):
        steps.append(_tr(
            lang,
            "2) Перевір: використовуй лише безглютенову основу (рис/гречка/картопля/кіноа).",
            "2) Provjeri: koristi bezglutensku bazu (riža/heljda/krumpir/kvinoja).",
            "2) Check: use gluten-free base (rice/buckwheat/potato/quinoa)."
        ))

    if f.get("Lactose free"):
        steps.append(_tr(
            lang,
            "3) Без лактози: уникай сиру/йогурту, заправляй олією, лимоном або соєвим соусом.",
            "3) Bez laktoze: izbjegni sir/jogurt; začini uljem, limunom ili soja umakom.",
            "3) Lactose-free: avoid dairy; dress with oil, lemon or soy sauce."
        ))

    cook_variant = rng.choice(["pan", "mix", "bowl"])
    if cook_variant == "pan":
        steps.append(_tr(
            lang,
            "4) Сковорідка (6–10 хв): розігрій на середньому вогні, додай 1 ст. л. олії. Обсмаж основний продукт 4–7 хв. Овочі додай наприкінці на 1–2 хв.",
            "4) Tava (6–10 min): zagrij na srednje, dodaj 1 žlicu ulja. Prži glavni sastojak 4–7 min. Povrće dodaj zadnje 1–2 min.",
            "4) Pan (6–10 min): heat on medium, add 1 tbsp oil. Cook main ingredient 4–7 min. Add veggies for the last 1–2 min."
        ))
    elif cook_variant == "mix":
        steps.append(_tr(
            lang,
            "4) Швидке змішування (2–3 хв): у мисці змішай основу + білок + овочі. Додай соус, перемішай.",
            "4) Brzo miješanje (2–3 min): u zdjeli spoji bazu + protein + povrće. Dodaj umak i promiješaj.",
            "4) Quick mix (2–3 min): in a bowl combine base + protein + veggies. Add sauce and mix."
        ))
    else:
        steps.append(_tr(
            lang,
            "4) Боул (3–5 хв): виклади основу, зверху білок і овочі. Заправ соусом.",
            "4) Bowl (3–5 min): stavi bazu, zatim protein i povrće. Prelij umakom.",
            "4) Bowl (3–5 min): add base, top with protein and veggies. Dress with sauce."
        ))

    steps.append(_tr(
        lang,
        "5) Смак (1 хв): посоли/поперчи. Якщо хочеш-додай лимонний сік або ще трішки соусу.",
        "5) Okus (1 min): posoli/papri. Po želji dodaj limun ili još malo umaka.",
        "5) Taste (1 min): add salt/pepper. Optionally add lemon or a bit more sauce."
    ))

    if f.get("High protein"):
        steps.append(_tr(
            lang,
            "6) High protein порада: збільш порцію білка і зменш пасту/хліб.",
            "6) High protein savjet: povećaj protein, smanji tjesteninu/kruh.",
            "6) High protein tip: increase protein, reduce pasta/bread."
        ))

    if f.get("Low calorie"):
        steps.append(_tr(
            lang,
            "7) Low calorie порада: більше овочів, менше олії (1 ч. л. замість 1 ст. л.).",
            "7) Low calorie savjet: više povrća, manje ulja (1 žličica umjesto 1 žlice).",
            "7) Low calorie tip: more veggies, less oil (1 tsp instead of 1 tbsp)."
        ))

    steps.append(_tr(
        lang,
        "Фініш: подавай одразу. Смачного 🙂",
        "Finish: posluži odmah. Dobar tek 🙂",
        "Finish: serve right away. Enjoy 🙂"
    ))

    return steps[:10]


def _estimate_time(f: Dict[str, Any], rng: random.Random) -> int:
    Max time = int(f.get("Max time", 0) or 0)
    if Max time:
        return max(8, min(Max time, rng.randint(8, Max time)))
    
    return rng.randint(10, 25)

def _tags_for_filters(f: Dict[str, Any]) -> List[str]:
    tags = ["quick"]
    diet = f.get("diet", "any")
    if diet in ("vegetarian","vegan","pescatarian"):
        tags.append(diet)
    for k in ("Gluten free","Lactose free","High protein","Low calorie"):
        if f.get(k):
            tags.append(k)
   
    out = []
    for t in tags:
        if t in ALLOWED_TAGS and t not in out:
            out.append(t)
    return out

def _sig_for_dish(title: str, ingredients: List[str], tags: List[str]) -> str:
    base = (title or "") + "|" + "|".join(ingredients[:6]) + "|" + "|".join(tags)
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:18]


def generate_dish_for_user(u: Dict[str, Any], refresh_seed: Optional[str] = None) -> Dict[str, Any]:
    """
    Deterministic per-day/per-user, but when refresh_seed changes (e.g. "refresh click"),
    you get a different dish.
    """
    lang = u.get("lang", "ua")
    f = u.get("filters") or default_filters()
    diet = f.get("diet", "any")

   
    seed_base = f"{u.get('created_at','')}-{today()}-{lang}-{json.dumps(f, sort_keys=True)}"
    if refresh_seed:
        seed_base += f"-{refresh_seed}"
    rng = random.Random(hashlib.sha256(seed_base.encode("utf-8")).hexdigest())

    
    last_sig = u.get("last_dish_sig", "")
    for _ in range(6):
        title = _build_title(lang, diet, f, rng)
        ingredients = _build_ingredients(lang, diet, f, rng)
        steps = _build_steps(lang, f, rng)
        time_total_min = _estimate_time(f, rng)
        tags = _tags_for_filters(f)

        why = _tr(
            lang,
            "Згенеровано під твої фільтри. Мінімум часу, максимум користі.",
            "Generirano prema tvojim filterima. Malo vremena, puno koristi.",
            "Generated for your filters. Minimal time, maximum value."
        )

        dish = {
            "title": title,
            "why": why,
            "ingredients": ingredients,
            "steps": steps,
            "time_total_min": int(time_total_min),
            "tags": tags,
        }

       
        if not dish_matches_filters(dish, f):
            continue

        sig = _sig_for_dish(title, ingredients, tags)
        if sig != last_sig:
            u["last_dish_sig"] = sig
            return dish

    
    u["last_dish_sig"] = ""
    return {
        "title": _tr(lang, "Страва дня: демо", "Jelo dana: demo", "Dish of the day: demo"),
        "why": _tr(lang, "Не вдалося підібрати під фільтри, спробуй змінити фільтри.", "Nije moguće po filtrima, promijeni filtre.", "Couldn't match filters, try changing filters."),
        "ingredients": [_tr(lang,"вода — 200 мл","voda — 200 ml","water — 200 ml")],
        "steps": [_tr(lang,"Спробуй інші фільтри.","Probaj druge filtre.","Try other filters.")],
        "time_total_min": 5,
        "tags": ["quick"],
    }




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
    
    if not x_telegram_init_data:
        return {
            "lang": "ua",
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
    lang = payload.get("lang", "ua")
    if lang not in ("ua", "hr", "en"):
        lang = "ua"
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
    
    if isinstance(f.get("exclude"), str):
        f["exclude"] = [x.strip() for x in f["exclude"].split(",") if x.strip()]
    u["filters"] = f

    
    u["last_dish_sig"] = ""
    save_db(db)
    return {"ok": True}


@app.post("/api/daily")
def api_daily(payload: Dict[str, Any] = None, x_telegram_init_data: str = Header(default="")):
    
    if not x_telegram_init_data:
       
        refresh_seed = None
        if isinstance(payload, dict):
            refresh_seed = payload.get("refresh_seed")
        dish = {
            "title": "Demo: Dish generator",
            "why": "DEMO режим (відкрито не в Telegram). Відкрий Mini App через бота для персоналізації.",
            "ingredients": ["2 eggs — 2 pcs", "salt — to taste", "olive oil — 1 tbsp"],
            "steps": ["Prep 2–4 min", "Cook 6–10 min", "Serve"],
            "time_total_min": 12,
            "tags": ["quick", "High protein"]
        }
        return {"ok": True, "dish": dish, "demo": True}

    user_id = uid_from_init(x_telegram_init_data)

    
    refresh_seed = None
    if isinstance(payload, dict):
        refresh_seed = payload.get("refresh_seed")

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

        dish = generate_dish_for_user(u, refresh_seed=refresh_seed)
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

    
    dish = generate_dish_for_user(u, refresh_seed=None)
    save_db(db)

    if not dish:
        return {"ok": True, "data": None}
    if action == "ingredients":
        return {"ok": True, "data": dish.get("ingredients", [])}
    if action == "steps":
        return {"ok": True, "data": dish.get("steps", [])}
    return {"ok": True, "data": dish.get("time_total_min", 0)}
