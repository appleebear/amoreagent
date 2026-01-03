import os
import json
import sqlite3
import re
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# -------------------------
# Config
# -------------------------
SQLITE_PATH = os.getenv("SQLITE_PATH", "./amoremall.sqlite")
PERSONA_PATH = os.getenv("PERSONA_PATH", "./persona_logic_v2.json")
PERSONA_BRAND_MATCH_PATH = os.getenv("PERSONA_BRAND_MATCH_PATH", "./persona_brand_matching.json")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

AVAILABLE_LLM_MODELS = [
    "gpt-5.2",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
]

AVAILABLE_LANGUAGES = [
    {"code": "ko", "label": "한국어"},
    {"code": "en", "label": "영어"},
    {"code": "zh", "label": "중국어"},
    {"code": "ja", "label": "일본어"},
    {"code": "ar", "label": "아랍어"},
    {"code": "hi", "label": "힌디어"},
    {"code": "fr", "label": "프랑스어"},
    {"code": "de", "label": "독일어"},
    {"code": "es", "label": "스페인어"},
    {"code": "th", "label": "태국어"},
    {"code": "vi", "label": "베트남어"},
    {"code": "id", "label": "인도네시아어"},
]

LANG_LABEL_BY_CODE = {x["code"]: x["label"] for x in AVAILABLE_LANGUAGES}

# Fallback brand tier map (used when index metadata is missing)
BRAND_TIER_FALLBACK = {
    "설화수": "luxury",
    "헤라": "prestige",
    "에이피뷰티": "luxury",
    "아이오페": "premium",
    "프리메라": "premium",
    "한율": "premium",
    "라네즈": "daily",
    "이니스프리": "daily",
    "에뛰드": "daily",
    "에스쁘아": "premium",
    "에스트라": "derma",
    "일리윤": "daily",
    "비레디": "daily",
    "오딧세이": "daily",
    "해피바스": "daily",
    "홀리추얼": "daily",
    "코스알엑스": "derma",
    "피카소": "tools",
}

FAISS_PRODUCTS_DIR = os.getenv("FAISS_PRODUCTS_DIR", "./data/faiss_products")
FAISS_PERSONAS_DIR = os.getenv("FAISS_PERSONAS_DIR", "./data/faiss_personas")

DEFAULT_MAX_TITLE = 40
DEFAULT_MAX_BODY = 350

#
# Output length policy
# - Korean keeps strict char limits (title/body).
# - Other languages: keep the message short like marketing copy.
#   We use conservative char limits plus word/sentence constraints for alphabetic/space-delimited languages.
LANG_CHAR_LIMITS = {
    "ko": (40, 350),   # keep as-is
    "ja": (34, 260),
    "zh": (34, 260),
    # Alphabetic / space-delimited languages (tighter than before to avoid long explanations)
    "en": (48, 360),
    "fr": (50, 380),
    "de": (50, 380),
    "es": (50, 380),
    "id": (50, 380),
    "vi": (50, 380),
    "th": (50, 380),
    # Abjad / Indic scripts (keep concise; still allow slightly longer than ko but far less than previous)
    "ar": (50, 380),
    "hi": (50, 380),
}

# Word/sentence constraints for non-Korean output (marketing copy; prevents verbosity).
# Only applied for space-delimited languages.
LANG_WORD_LIMITS = {
    "en": (10, 70),
    "fr": (10, 75),
    "de": (10, 75),
    "es": (10, 75),
    "id": (10, 75),
    "vi": (10, 75),
    "th": (10, 75),
    "ar": (10, 75),
    "hi": (10, 75),
}
LANG_SENTENCE_LIMITS = {
    # keep copy-like brevity
    "en": 3,
    "fr": 3,
    "de": 3,
    "es": 3,
    "id": 3,
    "vi": 3,
    "th": 3,
    "ar": 3,
    "hi": 3,
    # CJK: sentence-ish limit; we still primarily use chars
    "ja": 3,
    "zh": 3,
}

# Channel-aware caps (non-Korean). Push must be short; Kakao/Email can be longer.
CHANNEL_SENTENCE_CAP = {
    "push": 2,
    "sms": 2,
    "kakao": 4,
    "email": 5,
}

# Word caps apply only for space-delimited languages (see LANG_WORD_LIMITS).
CHANNEL_WORD_CAP = {
    "push": (8, 45),     # title/body words
    "sms": (8, 45),
    "kakao": (10, 90),
    "email": (12, 120),
}

CATEGORY_LEVEL_TO_COLUMN = {
    "large": "categoryLarge",
    "middle": "categoryMiddle",
    "small": "categorySmall",
}

def _limits_for_language_and_channel(code: str, channel: str) -> Tuple[int, int]:
    """Channel-aware output limits.
    - push: app notification → short
    - kakao: 알림톡 → medium
    - email: longer allowed
    """
    lang = (code or "ko").strip().lower()
    ch = (channel or "").strip().lower()

    base_title, base_body = LANG_CHAR_LIMITS.get(lang, (DEFAULT_MAX_TITLE, DEFAULT_MAX_BODY))

    # Korean: keep existing baseline, but adjust by channel
    if lang == "ko":
        if ch == "push":
            return (28, 140)
        if ch == "email":
            return (50, 520)
        # kakao/sms/others
        return (40, 350)

    # Non-Korean: allow more room for kakao/email than push
    if ch == "push":
        return (min(base_title, 36), min(base_body, 200))
    if ch == "email":
        return (max(base_title, 60), max(base_body, 520))
    if ch == "kakao":
        return (max(base_title, 54), max(base_body, 440))

    return (base_title, base_body)

# -------------------------
# Utilities
# -------------------------
def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _safe_int(x, default=None):
    try:
        return int(x) if x is not None else default
    except Exception:
        return default

def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try to find the first {...} block
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    block = m.group(0)
    try:
        return json.loads(block)
    except Exception:
        # common repair: replace smart quotes
        block2 = block.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
        try:
            return json.loads(block2)
        except Exception:
            return None

def _truncate_hard(s: str, max_len: int) -> str:
    if s is None:
        return ""
    s = s.strip()
    if len(s) <= max_len:
        return s
    # leave room for ellipsis
    return (s[: max_len - 1].rstrip() + "…") if max_len >= 2 else s[:max_len]

# --- Non-Korean brevity enforcement helpers ---
def _count_words(s: str) -> int:
    if not s:
        return 0
    return len([w for w in re.split(r"\s+", s.strip()) if w])

def _truncate_words(s: str, max_words: int) -> str:
    if not s:
        return ""
    words = [w for w in re.split(r"\s+", s.strip()) if w]
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).rstrip() + "…"

def _trim_sentences_by_lang(s: str, lang: str, max_sentences: int) -> str:
    if not s:
        return ""
    lang = (lang or "ko").strip().lower()

    # Sentence split patterns
    if lang in ("ja", "zh"):
        # Split on 。！？ plus newlines
        parts = re.split(r"(?<=[。！？])\s*|\n+", s.strip())
    else:
        # Default: split on . ! ? plus newlines
        parts = re.split(r"(?<=[\.\!\?])\s+|\n+", s.strip())

    parts = [p.strip() for p in parts if p and p.strip()]
    if len(parts) <= max_sentences:
        return "\n".join(parts) if "\n" in s else " ".join(parts)
    trimmed = parts[:max_sentences]
    return "\n".join(trimmed) if "\n" in s else " ".join(trimmed)

def _objective_kor(obj: str) -> str:
    return {
        "acquisition": "획득(첫구매)",
        "repurchase": "재구매",
        "upsell": "업셀",
        "cross_sell": "교차판매",
        "winback": "윈백(휴면/이탈)",
    }.get(obj, obj)

def _channel_kor(ch: str) -> str:
    return {
        "push": "푸시",
        "kakao": "카카오 알림톡",
        "sms": "SMS",
        "email": "이메일",
    }.get(ch, ch)

def _style_kor(st: str) -> str:
    return {
        "soft": "부드럽게",
        "crisp": "간결하게",
        "emotive": "감성적으로",
    }.get(st, st)

def _price_band(price_discounted: Optional[int]) -> str:
    # heuristic band for objective-aware picking
    if price_discounted is None:
        return "unknown"
    if price_discounted < 25000:
        return "low"
    if price_discounted < 70000:
        return "mid"
    return "high"

def _infer_brand_tier_fallback(brand_name: str) -> str:
    b = (brand_name or "").strip()
    return BRAND_TIER_FALLBACK.get(b, "unknown")

def _infer_gift_fit_fallback(p: Dict[str, Any]) -> str:
    # Conservative heuristic: treat sets/kits and higher price as better for gifting.
    name = str(p.get("name") or "").lower()
    cat = str(p.get("categoryTitle") or "").lower()
    price = _safe_int(p.get("price_discounted"))
    text = name + " " + cat
    if any(k in text for k in ["세트", "기획", "키트", "gift", "set", "kit"]):
        return "high"
    if price is not None and price >= 70000:
        return "high"
    if price is not None and price >= 25000:
        return "mid"
    return "low"

def _brand_tone_pack(brand_name: str) -> Dict[str, Any]:
    # Extend as needed. Unknown brands fall back to neutral guidance.
    tones = {
        "설화수": {
            "keywords": ["품격", "절제된 감성", "리추얼", "우아함"],
            "do": ["고급스러운 자기관리 톤", "감각적이되 과장 금지", "존댓말 기반"],
            "dont": ["의학적 효능/치료 단정", "과장된 단정", "공포 마케팅"],
            "sentenceStyle": "중간 길이, 리추얼·감각 묘사",
            "cta": "부드럽게 제안",
        },
        "헤라": {
            "keywords": ["도회적", "세련", "자신감", "프리미엄"],
            "do": ["간결한 고급감", "자기표현/룩 완성"],
            "dont": ["과도한 귀여움", "치료/완치 표현"],
            "sentenceStyle": "짧~중간, 확신 있는 문장",
            "cta": "세련되게 행동 유도",
        },
        "라네즈": {
            "keywords": ["산뜻", "트렌디", "수분", "데일리"],
            "do": ["가볍고 경쾌", "사용감 중심", "루틴 제안"],
            "dont": ["지나친 격식", "치료/완치 표현"],
            "sentenceStyle": "짧~중간, 명확한 이점",
            "cta": "가볍게 행동 유도",
        },
        "이니스프리": {
            "keywords": ["자연스러움", "편안", "클린", "일상"],
            "do": ["친근하고 담백", "부담 낮추기"],
            "dont": ["과장", "공포 마케팅"],
            "sentenceStyle": "짧고 다정",
            "cta": "편하게 추천",
        },
        "에스트라": {
            "keywords": ["장벽", "저자극", "더마", "안정감"],
            "do": ["민감/장벽 케어 중심", "근거형 표현(개인차)"],
            "dont": ["질병/염증 치료 단정", "의학적 오인"],
            "sentenceStyle": "짧~중간, 안전감 있는 문장",
            "cta": "조심스럽게 제안",
        },
        "일리윤": {
            "keywords": ["가족", "보습", "순함", "데일리"],
            "do": ["포근한 톤", "온가족/데일리"],
            "dont": ["과장", "치료 표현"],
            "sentenceStyle": "짧고 따뜻",
            "cta": "부담 없이 제안",
        },
        "프리메라": {
            "keywords": ["클린", "식물 유래", "균형", "건강한 루틴"],
            "do": ["차분한 클린뷰티 톤", "지속가능/성분"],
            "dont": ["과학 오인", "치료 표현"],
            "sentenceStyle": "중간, 담백",
            "cta": "루틴 제안",
        },
        "한율": {
            "keywords": ["균형", "차분", "보습", "결"],
            "do": ["차분한 서사", "결·보습 중심"],
            "dont": ["질병/염증 치료 암시"],
            "sentenceStyle": "중간, 차분",
            "cta": "은은한 제안",
        },
        "에뛰드": {
            "keywords": ["발랄", "가성비", "재미", "컬러"],
            "do": ["경쾌하고 친근", "포인트 강조"],
            "dont": ["과도한 격식"],
            "sentenceStyle": "짧고 리듬감 있게",
            "cta": "가볍게 참여 유도",
        },
        "에스쁘아": {
            "keywords": ["프로", "트렌디", "메이크업", "완성"],
            "do": ["선명하고 자신감"],
            "dont": ["치료 표현"],
            "sentenceStyle": "짧~중간, 임팩트",
            "cta": "명확한 CTA",
        },
        "아이오페": {
            "keywords": ["기술", "스킨솔루션", "탄탄", "신뢰"],
            "do": ["기술/솔루션 톤(과장 금지)", "명확한 효익"],
            "dont": ["의약품 오인"],
            "sentenceStyle": "중간, 논리적" ,
            "cta": "확신 있게 제안",
        },
        "코스알엑스": {
            "keywords": ["성분", "핵심", "실용", "가성비"],
            "do": ["핵심 성분/효익 중심", "간결"],
            "dont": ["치료 단정"],
            "sentenceStyle": "짧고 명확" ,
            "cta": "바로 써보게" ,
        },
        "비레디": {
            "keywords": ["남성 그루밍", "간편", "실용", "쿨"],
            "do": ["간결/실용", "부담 낮추기"],
            "dont": ["과한 감성"],
            "sentenceStyle": "짧고 직관" ,
            "cta": "바로 적용" ,
        },
        "오딧세이": {
            "keywords": ["남성", "클래식", "신뢰", "정돈"],
            "do": ["깔끔/정갈"],
            "dont": ["치료 표현"],
            "sentenceStyle": "짧~중간" ,
            "cta": "정중한 제안" ,
        },
        "해피바스": {
            "keywords": ["향", "샤워", "기분전환", "데일리"],
            "do": ["기분/향 묘사", "상쾌"],
            "dont": ["과장"],
            "sentenceStyle": "짧~중간" ,
            "cta": "가볍게 추천" ,
        },
        "홀리추얼": {
            "keywords": ["두피", "스칼프", "클린", "리프레시"],
            "do": ["두피/헤어 루틴 중심"],
            "dont": ["질병 치료 단정"],
            "sentenceStyle": "짧~중간" ,
            "cta": "루틴 제안" ,
        },
        "에이피뷰티": {
            "keywords": ["하이엔드", "집중 케어", "프리미엄"],
            "do": ["절제된 고급감"],
            "dont": ["과장"],
            "sentenceStyle": "중간" ,
            "cta": "부드럽게 제안" ,
        },
    }
    return tones.get(brand_name, {
        "keywords": ["신뢰", "개인화", "데일리"],
        "do": ["고객 니즈를 먼저 언급", "과장 없이 명확한 효익"],
        "dont": ["의학적 효능/치료 단정", "공포 마케팅"],
        "sentenceStyle": "짧~중간, 읽기 쉬운 문장",
        "cta": "부담 없이 제안",
    })

# -------------------------
# Data access
# -------------------------
def _connect_sqlite(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _distinct_brands(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute(
        """SELECT DISTINCT brandName FROM products
           WHERE brandName IS NOT NULL AND brandName != ''
           ORDER BY brandName"""
    ).fetchall()
    return [r[0] for r in rows]

def _distinct_categories(conn: sqlite3.Connection) -> List[str]:
    """Return all distinct categoryTitle values for UI dropdown."""
    rows = conn.execute(
        """SELECT DISTINCT categoryTitle FROM products
           WHERE categoryTitle IS NOT NULL AND categoryTitle != ''
           ORDER BY categoryTitle"""
    ).fetchall()
    return [r[0] for r in rows]

def _category_tree(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Return hierarchical category data (large > middle > small)."""
    rows = conn.execute(
        """SELECT DISTINCT categoryLarge, categoryMiddle, categorySmall
           FROM products
           WHERE categoryLarge IS NOT NULL AND categoryLarge != ''
           ORDER BY categoryLarge, categoryMiddle, categorySmall"""
    ).fetchall()

    tree: Dict[str, Dict[str, set]] = {}
    for r in rows:
        large = (r["categoryLarge"] or "").strip()
        middle = (r["categoryMiddle"] or "").strip()
        small = (r["categorySmall"] or "").strip()
        if not large:
            continue
        tree.setdefault(large, {})
        if middle:
            tree[large].setdefault(middle, set())
            if small:
                tree[large][middle].add(small)

    preferred = ["스킨케어", "메이크업"]
    ordered_larges: List[str] = []
    for name in preferred:
        if name in tree:
            ordered_larges.append(name)
    ordered_larges.extend(sorted([k for k in tree.keys() if k not in preferred]))

    out: List[Dict[str, Any]] = []
    for large in ordered_larges:
        mids = tree[large]
        out.append({
            "large": large,
            "middles": [
                {"middle": m, "smalls": sorted(list(smalls))}
                for m, smalls in sorted(mids.items())
            ]
        })
    return out

# --- Price range: DB-backed min/max for price_discounted ---
def _price_range_discounted(conn: sqlite3.Connection) -> Tuple[Optional[int], Optional[int]]:
    """Return (min_price_discounted, max_price_discounted) from products.

    We use price_discounted as the primary price signal for the UI.
    """
    row = conn.execute(
        """SELECT MIN(price_discounted) AS min_p, MAX(price_discounted) AS max_p
           FROM products
           WHERE price_discounted IS NOT NULL"""
    ).fetchone()
    if not row:
        return (None, None)
    return (_safe_int(row["min_p"]), _safe_int(row["max_p"]))

# -------------------------
# RAG retrieval + selection
# -------------------------
def _group_best_by_product(docs_with_scores: List[Tuple[Document, float]], top_n_unique: int = 10):
    # FAISS score is distance (smaller is closer) for L2, or -similarity depending config.
    best = {}
    for d, s in docs_with_scores:
        prod_sn = d.metadata.get("onlineProdSn")
        if prod_sn is None:
            continue
        if (prod_sn not in best) or (s < best[prod_sn][1]):
            best[prod_sn] = (d, s)
    items = sorted(best.values(), key=lambda x: x[1])
    return items[:top_n_unique]

def _diversify_candidates_by_brand(
    cands: List[Dict[str, Any]],
    max_per_brand: int = 2,
    max_total: int = 12,
) -> List[Dict[str, Any]]:
    """
    Prevent one brand from dominating due to chunking / frequent generic keywords.
    Keep relevance order, but cap how many candidates from the same brand are allowed
    in the pool that feeds the final picker.

    We do a two-pass selection:
    1) relevance-ordered pass with per-brand cap
    2) if still short, fill remaining slots without cap
    """
    if not cands:
        return []

    out: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}

    for p in cands:
        b = (p.get("brandName") or "").strip()
        if counts.get(b, 0) >= max_per_brand:
            continue
        out.append(p)
        counts[b] = counts.get(b, 0) + 1
        if len(out) >= max_total:
            return out

    # Fill remaining slots (if any) without cap
    for p in cands:
        if p in out:
            continue
        out.append(p)
        if len(out) >= max_total:
            break

    return out


# -------------------------
# Persona-aware re-ranking and tier signals
# -------------------------

# ----------- Metadata-fit scoring utilities -----------
def _brand_tier_rank(t: str) -> int:
    # Higher is more premium (used for luxury personas / upsell)
    return {
        "luxury": 4,
        "prestige": 3,
        "premium": 2,
        "daily": 1,
        "derma": 1,
        "tools": 0,
        "unknown": 0,
    }.get((t or "").strip(), 0)


def _metadata_fit_score(
    p: Dict[str, Any],
    persona: Optional[Dict[str, Any]],
    objective: str,
    concerns: List[str],
    notes: str,
    price_sensitivity: str,
    preferred_brands: Optional[List[str]] = None,
) -> float:
    """
    Higher is better. We prefer persona-appropriate brandTier/priceTier/giftFit first,
    and use RAG distance only as a tie-breaker downstream.

    This intentionally avoids hard filters to reduce empty results.
    """
    score = 0.0

    bt = (p.get("brandTier") or "unknown").strip()
    pt = (p.get("priceTier") or "unknown").strip()
    gf = (p.get("giftFit") or "unknown").strip()

    concerns_text = " ".join([str(x) for x in (concerns or [])]).lower()
    notes_text = str(notes or "").lower()
    text_all = (concerns_text + " " + notes_text)

    makeup_intent = any(k in text_all for k in [
        "메이크업", "립", "아이", "섀도", "파운데이션", "쿠션",
        "브러시", "툴", "makeup", "brush", "lip", "eye"
    ])
    gifting_intent = any(k in text_all for k in ["선물", "기프트", "present", "gift"])

    pref = _infer_persona_pref(persona or {}) if persona else {"is_luxury": False, "min_price_tier": None}
    is_lux = bool(pref.get("is_luxury"))

    # 1) Tools/accessories: demote unless makeup intent
    if bt == "tools" and not makeup_intent:
        score -= 120.0
    elif bt == "tools" and makeup_intent:
        score += 10.0

    # 2) Gifting: reward giftFit high / punish low
    if gifting_intent:
        if gf == "high":
            score += 80.0
        elif gf == "mid":
            score += 35.0
        elif gf == "low":
            score -= 90.0

    # 3) Luxury persona: reward premium tiers, punish low price tiers
    if is_lux:
        score += 40.0 * _brand_tier_rank(bt)
        if _tier_rank(pt) <= _tier_rank("low"):
            score -= 220.0
        if bt in ["daily", "derma", "tools"]:
            score -= 90.0

    # 4) Price sensitivity: align to priceTier
    ps = (price_sensitivity or "med").strip().lower()
    if ps in ["high", "sensitive", "low_budget"]:
        # prefer low/mid
        if pt == "low":
            score += 35.0
        elif pt == "mid":
            score += 15.0
        elif pt == "high":
            score -= 40.0
    elif ps in ["low", "insensitive", "high_budget"]:
        # allow mid/high
        if pt == "high":
            score += 25.0
        elif pt == "mid":
            score += 15.0
        elif pt == "low":
            score -= 10.0

    # 5) Objective: gentle alignment
    if objective == "upsell":
        if pt in ["mid", "high"]:
            score += 25.0
        else:
            score -= 10.0
    elif objective == "acquisition":
        if pt == "high":
            score -= 15.0
        else:
            score += 8.0
    elif objective == "winback":
        # prefer discounted if available
        dr = p.get("discount_rate") or 0
        try:
            if float(dr) >= 20:
                score += 12.0
        except Exception:
            pass

    return score

def _tier_rank(t: str) -> int:
    return {"unknown": 0, "low": 1, "mid": 2, "high": 3}.get((t or "").strip(), 0)


def _infer_persona_pref(
    persona: Dict[str, Any],
    persona_id: Optional[str] = None,
    has_explicit_brand_mapping: bool = False,
) -> Dict[str, Any]:
    """Infer coarse preferences from persona label text (conservative heuristic).

    IMPORTANT: If persona_brand_matching.json provides an explicit mapping for this persona,
    we MUST NOT let heuristic inference (e.g., "Luxury" in the name) override or inject
    brand/tier behavior. In that case, we return a neutral preference.
    """
    if has_explicit_brand_mapping:
        return {
            "is_luxury": False,
            "min_price_tier": None,
            "pref_source": "persona_brand_matching.json",
        }

    name_ko = str(persona.get("name_ko") or "")
    name_en = str(persona.get("name_en") or "")
    label = (name_ko + " " + name_en).lower()

    pref: Dict[str, Any] = {
        "is_luxury": False,
        "min_price_tier": None,
        "pref_source": "heuristic",
    }

    if any(k in label for k in [
        "lux", "luxe", "luxury",
        "프리미엄", "프레스티지", "럭셔리", "명품", "고가", "하이엔드"
    ]):
        pref["is_luxury"] = True
        pref["min_price_tier"] = "mid"

    return pref


def _rerank_products(
    cands: List[Dict[str, Any]],
    persona: Optional[Dict[str, Any]],
    objective: str,
    concerns: List[str],
    notes: str,
    preferred_brands: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Persona-aware re-ranking (soft constraints).

    - Discourage tools/accessories dominance (e.g., Picasso brushes) unless makeup intent.
    - For luxury personas, heavily penalize low-priceTier.
    - For gifting intent (derived from notes/concerns), penalize giftFit=low and low-priceTier.

    We do not hard-filter to avoid empty results; we apply penalties.
    """
    if not cands:
        return []

    has_mapping = bool(preferred_brands)
    pref = _infer_persona_pref(persona or {}, has_explicit_brand_mapping=has_mapping) if persona else {"is_luxury": False, "min_price_tier": None}

    concerns_text = " ".join([str(x) for x in (concerns or [])]).lower()
    notes_text = str(notes or "").lower()
    text_all = (concerns_text + " " + notes_text)

    makeup_intent = any(k in text_all for k in [
        "메이크업", "립", "아이", "섀도", "파운데이션", "쿠션",
        "브러시", "툴", "makeup", "brush", "lip", "eye"
    ])

    gifting_intent = any(k in text_all for k in ["선물", "기프트", "present", "gift"])

    def penalty(p: Dict[str, Any]) -> float:
        pen = 0.0
        bt = (p.get("brandTier") or "unknown").strip()
        pt = (p.get("priceTier") or "unknown").strip()
        gf = (p.get("giftFit") or "unknown").strip()

        # Tools dominance (Picasso etc.) — discourage unless makeup intent
        # If the brand is explicitly preferred for this persona (via mapping JSON), do not penalize.
        brand_name = str(p.get("brandName") or "").strip()
        is_preferred_brand = bool(preferred_brands) and (brand_name in (preferred_brands or []))
        if bt == "tools" and not makeup_intent and not is_preferred_brand:
            pen += 60.0

        # Gifting intent — avoid low gift-fit and very low price
        if gifting_intent:
            if gf == "low":
                pen += 50.0
            if _tier_rank(pt) <= _tier_rank("low"):
                pen += 40.0

        # Luxury personas — very strong penalty for low price tier and non-prestige-ish tiers
        if pref.get("is_luxury"):
            if _tier_rank(pt) <= _tier_rank("low"):
                pen += 220.0
            # daily/tools tends to break luxury feel
            if bt in ["daily", "tools"]:
                pen += 80.0
            min_pt = pref.get("min_price_tier")
            if min_pt and _tier_rank(pt) < _tier_rank(str(min_pt)):
                pen += 80.0

        return pen

    return sorted(cands, key=lambda x: float(x.get("_rag_distance", 9999.0)) + penalty(x))

def _objective_pick(cands: List[Dict[str, Any]], objective: str, brand: str) -> List[Dict[str, Any]]:
    # cands: sorted by relevance already
    picked: List[Dict[str, Any]] = []
    def take_if(fn):
        for p in cands:
            if p in picked:
                continue
            if fn(p):
                picked.append(p)
                return
    if not cands:
        return []

    if objective == "upsell":
        take_if(lambda p: _price_band(p.get("price_discounted")) in ("mid", "high"))
        take_if(lambda p: _price_band(p.get("price_discounted")) == "high")
        take_if(lambda p: True)
    elif objective == "cross_sell":
        take_if(lambda p: True)
        first_cat = picked[0].get("categoryLarge") if picked else None
        take_if(lambda p: (p.get("categoryLarge") != first_cat) if first_cat else True)
        take_if(lambda p: True)
    elif objective == "repurchase":
        if brand and brand != "ANY":
            take_if(lambda p: p.get("brandName") == brand)
        take_if(lambda p: True)
        take_if(lambda p: True)
    elif objective == "winback":
        take_if(lambda p: _price_band(p.get("price_discounted")) != "high")
        # prefer discounted (higher discount_rate)
        take_if(lambda p: (p.get("discount_rate") or 0) >= 20)
        take_if(lambda p: True)
    else:
        # acquisition
        take_if(lambda p: _price_band(p.get("price_discounted")) != "high")
        take_if(lambda p: True)
        take_if(lambda p: True)

    return picked[:3]

def _rag_summary(persona_name: str, objective: str, concerns: List[str], picked: List[Dict[str, Any]]) -> str:
    c = ", ".join([x for x in concerns if x]) if concerns else "(입력 없음)"
    lines = [
        f"[페르소나] {persona_name}",
        f"[목적] {_objective_kor(objective)}",
        f"[키워드] {c}",
        "",
        "[선정 근거(요약)]",
    ]
    for i, p in enumerate(picked, 1):
        price = p.get("price_discounted")
        price_s = f"{int(price):,}원" if isinstance(price, int) else "-"
        dr = p.get("discount_rate")
        dr_s = f"{round(float(dr))}%" if dr is not None else "-"
        lines.append(f"{i}. [{p.get('brandName','-')}] {p.get('name','-')} · {p.get('categoryTitle','-')} · {price_s} (할인 {dr_s})")
        # simple overlap reason
        desc = (p.get("detail_text") or "").lower()
        hits = [kw for kw in (concerns or []) if kw and kw.lower() in desc]
        if hits:
            lines.append(f"   - 상세설명에서 키워드 매칭: {', '.join(hits[:6])}")
        else:
            lines.append("   - 카테고리/브랜드/가격대/목적 적합도 기반 선정")
    return "\n".join(lines)

# -------------------------
# API models
# -------------------------
class GenerateRequest(BaseModel):
    apiKey: Optional[str] = Field(default=None, description="(데모용) 브라우저에서 전달되는 OpenAI 키. 서버 환경변수 OPENAI_API_KEY가 우선.")
    model: Optional[str] = Field(default=None, description="(선택) 사용할 LLM 모델. 예: gpt-5.2, gpt-5-mini, gpt-5-nano, gpt-4.1")
    persona_id: str
    objective: str
    channel: str
    brand: str = "ANY"
    category_large: str = "ANY"
    category_middle: str = "ANY"
    category_small: str = "ANY"
    concerns: List[str] = Field(default_factory=list)
    price_sensitivity: str = "med"
    style: str = "soft"
    language: str = "ko"
    notes: str = ""
    use_price_filter: bool = False
    price_min: Optional[int] = None
    price_max: Optional[int] = None

# -------------------------
# App init
# -------------------------
app = FastAPI(title="Amoremall RAG Agent")

# Static UI
from pathlib import Path
UI_DIR = Path(__file__).parent / "static"
INDEX_HTML = UI_DIR / "index.html"

@app.get("/", response_class=HTMLResponse)
def ui():
    try:
        return INDEX_HTML.read_text(encoding="utf-8")
    except Exception as e:
        return f"UI 파일을 읽을 수 없습니다: {e}"

app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")

# Load persona logic
try:
    persona_logic = _load_json(PERSONA_PATH)
except Exception as e:
    persona_logic = {"error": str(e), "personas": {}}

# Load persona-brand matching (optional)
try:
    persona_brand_matching = _load_json(PERSONA_BRAND_MATCH_PATH)
except Exception as e:
    persona_brand_matching = {"error": str(e), "personas": [], "brands": []}

# Build: persona_id -> preferred brands (sorted by affinity score desc)
_persona_preferred_brands: Dict[str, List[str]] = {}
try:
    for b in (persona_brand_matching.get("brands") or []):
        brand_name = str(b.get("name") or "").strip()
        scores = b.get("persona_affinity_score_0to5") or {}
        if not brand_name or not isinstance(scores, dict):
            continue
        for pid, sc in scores.items():
            try:
                sc_f = float(sc)
            except Exception:
                sc_f = 0.0
            if sc_f <= 0:
                continue
            _persona_preferred_brands.setdefault(str(pid), []).append(brand_name)

    # De-duplicate while preserving score order by re-sorting using original affinity
    # (stable and conservative)
    _brand_affinity_by_persona: Dict[str, Dict[str, float]] = {}
    for b in (persona_brand_matching.get("brands") or []):
        bn = str(b.get("name") or "").strip()
        scores = b.get("persona_affinity_score_0to5") or {}
        if bn and isinstance(scores, dict):
            for pid, sc in scores.items():
                try:
                    _brand_affinity_by_persona.setdefault(str(pid), {})[bn] = float(sc)
                except Exception:
                    pass

    for pid, brands in list(_persona_preferred_brands.items()):
        # sort by affinity desc, then name for stability
        aff = _brand_affinity_by_persona.get(pid, {})
        brands_sorted = sorted(set(brands), key=lambda x: (-float(aff.get(x, 0.0)), x))
        _persona_preferred_brands[pid] = brands_sorted
except Exception:
    # Never break the app because of matching metadata
    _persona_preferred_brands = {}

def _preferred_brands_for_persona(persona_id: str, top_k: int = 5) -> List[str]:
    """Return top-K preferred brands for a persona based on persona_brand_matching.json.

    If the file is missing/invalid or the persona has no mapping, returns an empty list.
    """
    pid = str(persona_id or "").strip()
    if not pid:
        return []
    brands = _persona_preferred_brands.get(pid) or []
    return brands[: max(0, int(top_k))]

def _persona_brand_bonus(brand_name: str, preferred_brands: List[str]) -> float:
    """Bonus for selecting persona-preferred brands when user didn't lock a brand.

    Higher-ranked preferred brands get a bigger bonus.
    """
    b = str(brand_name or "").strip()
    if not b or not preferred_brands:
        return 0.0
    try:
        idx = preferred_brands.index(b)
    except ValueError:
        return 0.0
    # rank 0 -> +45, rank 1 -> +30, rank 2 -> +22, rank 3 -> +16, ...
    return max(8.0, 45.0 / float(idx + 1))

# DB connection
try:
    conn = _connect_sqlite(SQLITE_PATH)
except Exception as e:
    conn = None

# Vector stores
_products_vs: Optional[FAISS] = None
_personas_vs: Optional[FAISS] = None

def _load_vectorstores():
    global _products_vs, _personas_vs
    if _products_vs is None:
        if not os.path.isdir(FAISS_PRODUCTS_DIR):
            raise RuntimeError(f"FAISS 상품 인덱스를 찾을 수 없습니다: {FAISS_PRODUCTS_DIR} (먼저 python build_index.py 실행)")
        _products_vs = FAISS.load_local(
            FAISS_PRODUCTS_DIR,
            OpenAIEmbeddings(model=OPENAI_EMBED_MODEL),
            allow_dangerous_deserialization=True,
        )
    if _personas_vs is None:
        if not os.path.isdir(FAISS_PERSONAS_DIR):
            raise RuntimeError(f"FAISS 페르소나 인덱스를 찾을 수 없습니다: {FAISS_PERSONAS_DIR} (먼저 python build_index.py 실행)")
        _personas_vs = FAISS.load_local(
            FAISS_PERSONAS_DIR,
            OpenAIEmbeddings(model=OPENAI_EMBED_MODEL),
            allow_dangerous_deserialization=True,
        )

def _persona_name(persona_id: str) -> Tuple[str, str]:
    p = (persona_logic.get("personas") or {}).get(persona_id) or {}
    return (p.get("name_ko") or persona_id, p.get("name_en") or "")

def _make_llm(api_key_from_req: Optional[str], model_from_req: Optional[str]) -> ChatOpenAI:
    # priority: server env var > request payload
    if api_key_from_req and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = api_key_from_req

    chosen = OPENAI_MODEL
    if model_from_req:
        m = str(model_from_req).strip()
        if m in AVAILABLE_LLM_MODELS:
            chosen = m

    # 일부 모델은 temperature 파라미터를 제한적으로 지원함
    if chosen in {"gpt-5-mini", "gpt-5-nano"}:
        # 해당 모델은 기본값(1)만 허용
        return ChatOpenAI(model=chosen, temperature=1)

    return ChatOpenAI(model=chosen, temperature=0.95)

# -------------------------
# Routes
# -------------------------
@app.get("/api/meta")
def meta():
    if conn is None:
        return JSONResponse(status_code=500, content={"error": f"SQLite 연결 실패: {SQLITE_PATH}"})
    personas = []
    for pid, p in (persona_logic.get("personas") or {}).items():
        personas.append({"id": pid, "name_ko": p.get("name_ko", pid), "name_en": p.get("name_en", "")})
    personas = sorted(personas, key=lambda x: x["id"])
    brands = _distinct_brands(conn)
    categories = _distinct_categories(conn)
    category_tree = _category_tree(conn)
    min_price, max_price = _price_range_discounted(conn)
    return {
        "personas": personas,
        "brands": brands,
        "categories": categories,
        "category_tree": category_tree,
        "price_range_discounted": {"min": min_price, "max": max_price},
        "models": AVAILABLE_LLM_MODELS,
        "languages": AVAILABLE_LANGUAGES,
        "server_hint": "서버가 정상 동작 중입니다. (최초 실행 전 build_index.py로 인덱스를 생성하세요.)",
    }

@app.post("/api/generate")
def generate(req: GenerateRequest):
    if conn is None:
        return PlainTextResponse(f"SQLite 연결 실패: {SQLITE_PATH}", status_code=500)

    # 데모 모드: 브라우저에서 전달된 키를 사용 (서버 환경변수가 우선)
    if req.apiKey and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = req.apiKey

    # Load vector stores (lazy)
    try:
        _load_vectorstores()
    except Exception as e:
        return PlainTextResponse(str(e), status_code=500)

    persona_ko, persona_en = _persona_name(req.persona_id)
    persona_obj = (persona_logic.get("personas") or {}).get(req.persona_id) or {}

    # Persona-brand mapping (JSON) is the source of truth. If it exists, disable heuristic inference.
    preferred_brands = _preferred_brands_for_persona(req.persona_id, top_k=6)
    has_mapping = bool(preferred_brands)

    pref = _infer_persona_pref(
        persona_obj,
        persona_id=req.persona_id,
        has_explicit_brand_mapping=has_mapping,
    ) if persona_obj else {"is_luxury": False, "min_price_tier": None}
    title_max, body_max = _limits_for_language_and_channel(req.language, req.channel)

    def _normalize_category_value(value: Optional[str]) -> str:
        v = (value or "").strip()
        return v if v else "ANY"

    category_large = _normalize_category_value(getattr(req, "category_large", "ANY"))
    category_middle = _normalize_category_value(getattr(req, "category_middle", "ANY"))
    category_small = _normalize_category_value(getattr(req, "category_small", "ANY"))

    selected_level: Optional[str] = None
    selected_value: Optional[str] = None
    if category_small != "ANY":
        selected_level, selected_value = "small", category_small
    elif category_middle != "ANY":
        selected_level, selected_value = "middle", category_middle
    elif category_large != "ANY":
        selected_level, selected_value = "large", category_large
    selected_column = CATEGORY_LEVEL_TO_COLUMN.get(selected_level) if selected_level else None

    debug_lines: List[str] = []
    debug_lines.append(f"persona_id={req.persona_id} ({persona_ko}/{persona_en})")
    debug_lines.append(
        f"objective={req.objective} channel={req.channel} brand={req.brand} "
        f"category_large={category_large} category_middle={category_middle} category_small={category_small}"
    )
    debug_lines.append(f"concerns={req.concerns}")
    debug_lines.append(f"category_selected_level={selected_level or 'NONE'} value={selected_value or 'ANY'}")
    debug_lines.append(f"price_sensitivity={req.price_sensitivity} style={req.style}")
    debug_lines.append(f"model={req.model or OPENAI_MODEL}")
    debug_lines.append(f"language={req.language}")
    debug_lines.append(f"use_price_filter={req.use_price_filter} price_min={req.price_min} price_max={req.price_max}")
    if preferred_brands:
        debug_lines.append(f"persona_preferred_brands(top6)={preferred_brands}")
    else:
        debug_lines.append("persona_preferred_brands(top6)=[]")

    # 1) Persona retrieval (RAG on persona JSON)
    persona_query = f"{req.persona_id} {persona_ko} {persona_en} {_objective_kor(req.objective)} {' '.join(req.concerns)} {req.notes}"
    persona_docs = _personas_vs.similarity_search(persona_query, k=2) if _personas_vs else []
    persona_context = "\n\n".join([d.page_content for d in persona_docs])[:2500]

    # 2) Product retrieval (RAG on product DB)
    # Retrieve more, then brand-aware filter/penalty
    # When brand is ANY, bias the retrieval query toward persona-preferred brands (if available)
    preferred_hint = ""
    if (not req.brand or req.brand == "ANY") and preferred_brands:
        preferred_hint = " 추천브랜드:" + " ".join(preferred_brands)

    prod_query = (
        f"{persona_ko} {_objective_kor(req.objective)} {_channel_kor(req.channel)} "
        f"{' '.join(req.concerns)} {req.notes} 가격민감도:{req.price_sensitivity} 브랜드:{req.brand} "
        f"카테고리대:{category_large} 카테고리중:{category_middle} 카테고리소:{category_small}"
        f"{preferred_hint}"
    )
    k_vec = 120 if (not req.brand or req.brand == "ANY") else 60
    docs_scores = _products_vs.similarity_search_with_score(prod_query, k=k_vec) if _products_vs else []

    # Extra retrieval to ensure persona-preferred brands show up when brand is ANY
    if (not req.brand or req.brand == "ANY") and preferred_brands and _products_vs:
        for pb in preferred_brands[:4]:
            try:
                q = f"{pb} 브랜드:{pb} {pb} 아모레몰 상품"
                extra = _products_vs.similarity_search_with_score(q, k=24)
                docs_scores = (docs_scores or []) + (extra or [])
            except Exception:
                pass
        debug_lines.append(f"preferred_brand_boost_added={preferred_brands[:4]}")

    # If a specific brand is selected, run an additional brand-focused retrieval to ensure
    # the candidate pool includes that brand (vector search can miss it when concerns/notes are empty).
    if req.brand and req.brand != "ANY" and _products_vs:
        brand_query = f"{req.brand} 브랜드:{req.brand} {req.brand} {req.brand} 아모레몰 상품"
        brand_scores = _products_vs.similarity_search_with_score(brand_query, k=30)
        docs_scores = (docs_scores or []) + (brand_scores or [])
        debug_lines.append(f"brand_boost_query_added={req.brand} brand_scores={len(brand_scores or [])}")

    # If brand is ANY and persona is luxury, run an additional tier-hint retrieval to avoid
    # domination by high-text-density low-tier brands.
    if (not req.brand or req.brand == "ANY") and _products_vs and pref.get("is_luxury") and (not has_mapping):
        tier_hint_query = "프레스티지 럭셔리 고가 프리미엄 설화수 헤라 에이피뷰티 아이오페"
        tier_scores = _products_vs.similarity_search_with_score(tier_hint_query, k=60)
        docs_scores = (docs_scores or []) + (tier_scores or [])
        debug_lines.append(f"tier_hint_query_added=1 tier_scores={len(tier_scores or [])}")

    # When brand is ANY, we need a wider candidate set so metadata-fit can work
    # (otherwise the top-k is dominated by a few “high-text-density” brands).
    top_unique = 40 if (not req.brand or req.brand == "ANY") else 18
    grouped = _group_best_by_product(docs_scores, top_n_unique=top_unique)
    debug_lines.append(f"grouped_unique_products={len(grouped)} from_docs_scores={len(docs_scores or [])}")

    # Pull structured product rows for grouped candidates
    cand_products: List[Dict[str, Any]] = []
    for doc, score in grouped:
        prod_sn = doc.metadata.get("onlineProdSn")
        if prod_sn is None:
            continue
        row = conn.execute(
            """SELECT onlineProdSn, onlineProdCode, name, brandName, categoryTitle, categoryLarge, categoryMiddle, categorySmall,
                      price_before, price_discounted, discount_rate, url, detail_text
                 FROM products WHERE onlineProdSn = ?""",
            (prod_sn,),
        ).fetchone()
        if not row:
            continue
        p = dict(row)
        p["_rag_distance"] = float(score)
        # carry metadata signals from index (used for reranking), fallback if missing, treat literal "unknown" as missing
        bt_meta = doc.metadata.get("brandTier")
        pt_meta = doc.metadata.get("priceTier")
        gf_meta = doc.metadata.get("giftFit")

        # Treat literal "unknown" as missing so fallbacks can improve quality
        p["brandTier"] = (bt_meta if (bt_meta and bt_meta != "unknown") else _infer_brand_tier_fallback(p.get("brandName")))
        p["priceTier"] = (pt_meta if (pt_meta and pt_meta != "unknown") else _price_band(_safe_int(p.get("price_discounted"))))
        p["giftFit"] = (gf_meta if (gf_meta and gf_meta != "unknown") else _infer_gift_fit_fallback(p))
        # brand preference: soft rerank by penalty
        if req.brand and req.brand != "ANY" and p.get("brandName") != req.brand:
            p["_rag_distance"] += 0.15
        cand_products.append(p)

    cand_products.sort(key=lambda x: x.get("_rag_distance", 9999.0))

    # 가격 필터 적용: use_price_filter가 True일 때만
    if req.use_price_filter:
        before = len(cand_products)
        cand_products = [
            p for p in cand_products
            if p.get("price_discounted") is not None
            and (req.price_min is None or p.get("price_discounted") >= req.price_min)
            and (req.price_max is None or p.get("price_discounted") <= req.price_max)
        ]
        debug_lines.append(
            f"price_filter_applied remaining={len(cand_products)}/{before}"
        )

    # Category filter (hierarchical: large > middle > small)
    # - Soft bias via distance penalty
    # - If we can form a full recommendation set (>=3), enforce strictly later
    if selected_column and selected_value:
        for p in cand_products:
            if (p.get(selected_column) or "").strip() != selected_value:
                p["_rag_distance"] = float(p.get("_rag_distance", 9999.0)) + 0.20
        cand_products.sort(key=lambda x: x.get("_rag_distance", 9999.0))

        matched_cat_now = [
            p for p in cand_products if (p.get(selected_column) or "").strip() == selected_value
        ]
        debug_lines.append(
            f"category_filter_level={selected_level} value={selected_value} matched_candidates={len(matched_cat_now)}/{len(cand_products)}"
        )

        # SQL fallback: inject category products if vector pool missed them
        if len(matched_cat_now) == 0:
            try:
                rows = conn.execute(
                    f"""SELECT onlineProdSn, onlineProdCode, name, brandName, categoryTitle, categoryLarge, categoryMiddle, categorySmall,
                              price_before, price_discounted, discount_rate, url, detail_text
                       FROM products
                       WHERE {selected_column} = ?
                       ORDER BY COALESCE(discount_rate, 0) DESC, COALESCE(price_discounted, 0) DESC
                       LIMIT 20""",
                    (selected_value,),
                ).fetchall()
                injected = 0
                seen = set([p.get("onlineProdSn") for p in cand_products])
                for r in rows:
                    p = dict(r)
                    if p.get("onlineProdSn") in seen:
                        continue
                    p["_rag_distance"] = 9.2 + (injected * 0.01)
                    p["brandTier"] = _infer_brand_tier_fallback(p.get("brandName"))
                    p["priceTier"] = _price_band(_safe_int(p.get("price_discounted")))
                    p["giftFit"] = _infer_gift_fit_fallback(p)
                    cand_products.append(p)
                    injected += 1
                    if injected >= 20:
                        break
                if injected:
                    debug_lines.append(
                        f"category_sql_fallback_injected_level={selected_level} value={selected_value} count={injected}"
                    )
                    cand_products.sort(key=lambda x: x.get("_rag_distance", 9999.0))
            except Exception as e:
                debug_lines.append(f"category_sql_fallback_error={e}")

    # Brand SQL fallback: if user selected a brand but none of the grouped candidates match it,
    # inject a small set of that brand's products from SQLite so brand selection works predictably.
    if req.brand and req.brand != "ANY":
        matched_now = [p for p in cand_products if (p.get("brandName") == req.brand)]
        if len(matched_now) == 0:
            try:
                rows = conn.execute(
                    """SELECT onlineProdSn, onlineProdCode, name, brandName, categoryTitle, categoryLarge, categoryMiddle, categorySmall,
                              price_before, price_discounted, discount_rate, url, detail_text
                       FROM products
                       WHERE brandName = ?
                       ORDER BY COALESCE(discount_rate, 0) DESC, COALESCE(price_discounted, 0) DESC
                       LIMIT 12""",
                    (req.brand,),
                ).fetchall()
                injected = 0
                for r in rows:
                    p = dict(r)
                    # push behind true RAG hits but still usable by pick/rerank
                    p["_rag_distance"] = 9.0 + (injected * 0.01)
                    p["brandTier"] = _infer_brand_tier_fallback(p.get("brandName"))
                    p["priceTier"] = _price_band(_safe_int(p.get("price_discounted")))
                    p["giftFit"] = _infer_gift_fit_fallback(p)
                    cand_products.append(p)
                    injected += 1
                debug_lines.append(f"brand_sql_fallback_injected={req.brand} count={injected}")
                cand_products.sort(key=lambda x: x.get("_rag_distance", 9999.0))
            except Exception as e:
                debug_lines.append(f"brand_sql_fallback_error={e}")

    # Luxury ANY SQL augmentation: if retrieval is dominated by low-tier/low-price items,
    # inject premium/luxury candidates from SQLite so fit-scoring has real options.
    if (not req.brand or req.brand == "ANY") and pref.get("is_luxury") and (not has_mapping):
        mid_high = [p for p in cand_products if _tier_rank(p.get("priceTier")) >= _tier_rank("mid")]
        non_tools = [p for p in cand_products if (p.get("brandTier") != "tools")]
        if len(mid_high) < 6 or len(non_tools) < 8:
            premium_brands = [b for b, t in BRAND_TIER_FALLBACK.items() if t in ("premium", "prestige", "luxury")]
            try:
                q_marks = ",".join(["?"] * len(premium_brands))
                rows = conn.execute(
                    f"""SELECT onlineProdSn, onlineProdCode, name, brandName, categoryTitle, categoryLarge, categoryMiddle, categorySmall,
                               price_before, price_discounted, discount_rate, url, detail_text
                        FROM products
                        WHERE brandName IN ({q_marks})
                          AND COALESCE(price_discounted, 0) >= 25000
                        ORDER BY COALESCE(price_discounted, 0) DESC, COALESCE(discount_rate, 0) DESC
                        LIMIT 40""",
                    tuple(premium_brands),
                ).fetchall()
                injected = 0
                seen = set([p.get("onlineProdSn") for p in cand_products])
                for r in rows:
                    p = dict(r)
                    if p.get("onlineProdSn") in seen:
                        continue
                    p["_rag_distance"] = 8.5 + (injected * 0.01)  # behind true RAG but ahead of empties
                    p["brandTier"] = _infer_brand_tier_fallback(p.get("brandName"))
                    p["priceTier"] = _price_band(_safe_int(p.get("price_discounted")))
                    p["giftFit"] = _infer_gift_fit_fallback(p)
                    cand_products.append(p)
                    injected += 1
                    if injected >= 24:
                        break
                if injected:
                    debug_lines.append(f"luxury_any_sql_injected={injected}")
                    cand_products.sort(key=lambda x: x.get("_rag_distance", 9999.0))
            except Exception as e:
                debug_lines.append(f"luxury_any_sql_inject_error={e}")

    # Persona-aware re-ranking using brandTier/priceTier/giftFit metadata
    cand_products = _rerank_products(
        cand_products,
        persona_obj,
        req.objective,
        req.concerns or [],
        req.notes,
        preferred_brands=preferred_brands,
    )

    # Metadata-fit-first sorting (brandTier/priceTier/giftFit), RAG distance as tie-breaker.
    for p in cand_products:
        p["_fit_score"] = _metadata_fit_score(
            p,
            persona_obj,
            req.objective,
            req.concerns or [],
            req.notes,
            req.price_sensitivity,
            preferred_brands=preferred_brands,
        )
        # Persona-brand matching bonus (only when user didn't lock a brand)
        if (not req.brand or req.brand == "ANY") and preferred_brands:
            p["_fit_score"] = float(p.get("_fit_score", 0.0)) + _persona_brand_bonus(p.get("brandName") or "", preferred_brands)

    # Soft brand-diversity regularizer (ONLY for brand=ANY).
    # If the candidate set is dominated by a single brand, give a small bonus to less frequent brands
    # so metadata-fit can surface reasonable alternatives without hard caps.
    if not req.brand or req.brand == "ANY":
        freq: Dict[str, int] = {}
        for p in cand_products:
            b = (p.get("brandName") or "").strip()
            freq[b] = freq.get(b, 0) + 1
        max_f = max(freq.values()) if freq else 1

        for p in cand_products:
            b = (p.get("brandName") or "").strip()
            f = freq.get(b, 0)
            # bonus: rare brands +22, dominant brands ~0
            bonus = 22.0 * (1.0 - (float(f) / float(max_f)))
            p["_fit_score"] = float(p.get("_fit_score", 0.0)) + bonus

    cand_products.sort(
        key=lambda x: (-float(x.get("_fit_score", 0.0)), float(x.get("_rag_distance", 9999.0)))
    )

    # Brand selection: if user explicitly picked a brand and we have enough candidates for it,
    # enforce a strict brand-only candidate list (fallback to soft penalty if insufficient).
    if req.brand and req.brand != "ANY":
        matched = [p for p in cand_products if (p.get("brandName") == req.brand)]
        debug_lines.append(f"brand_filter_requested={req.brand} matched_candidates={len(matched)}/{len(cand_products)}")
        # If we can form a full recommendation set, keep only the selected brand.
        if len(matched) >= 3:
            cand_products = matched

    # Category selection: if user explicitly picked a category and we have enough candidates for it,
    # enforce a strict category-only candidate list (fallback to soft penalty if insufficient).
    if selected_column and selected_value:
        matched_c = [p for p in cand_products if (p.get(selected_column) or "").strip() == selected_value]
        # If we can form a full recommendation set, keep only the selected category.
        if len(matched_c) >= 3:
            cand_products = matched_c

    # Pool: for brand=ANY, we avoid brand quota; we rely on metadata-fit-first ordering.
    # For brand-specific selection, cand_products may already be restricted.
    pool = cand_products[:12]

    # Objective-aware final pick
    picked = _objective_pick(pool, req.objective, req.brand)

    rag_summary = _rag_summary(persona_ko, req.objective, req.concerns, picked)

    # 3) LLM generation
    tone_pack = _brand_tone_pack(req.brand if req.brand and req.brand != "ANY" else (picked[0].get("brandName") if picked else ""))
    tone_brand = req.brand if req.brand and req.brand != "ANY" else (picked[0].get("brandName") if picked else "브랜드 무관")

    products_for_prompt = []
    for p in picked:
        products_for_prompt.append({
            "brand": p.get("brandName"),
            "name": p.get("name"),
            "category": p.get("categoryTitle"),
            "price_discounted": _safe_int(p.get("price_discounted")),
            "discount_rate": p.get("discount_rate"),
            "url": p.get("url"),
            "detail_excerpt": (p.get("detail_text") or "")[:500],
        })

    system = (
        "너는 아모레퍼시픽 아모레몰의 CRM/카피라이터다. 사람(마케터)이 실제로 쓴 것처럼 자연스럽고 ‘말맛’ 있는 한국어로 개인화 메시지를 작성한다.\n"
        "아래 원칙을 반드시 지켜라.\n"
        "\n"
        "[출력 포맷] 반드시 JSON만 출력: {\"title\": string, \"body\": string}\n"
        f"- title은 {title_max}자 이내, body는 {body_max}자 이내(문자 수 기준)\n"
        "- (중요) 한국어가 아닌 경우, 장문 설명을 피하고 ‘마케팅 카피’처럼 짧게 쓸 것.\n"
        "- (중요) 한국어가 아닌 경우, 불필요한 배경 설명/근거 나열 금지. 핵심만 2~3문장으로.\n"
        "- 설명/여백/추가 텍스트 금지. JSON 문법(따옴표/중괄호/쉼표) 절대 깨지지 않게.\n"
        "\n"
        "[언어]\n"
        f"- 아래 language_code에 맞춰 제목과 본문을 해당 언어로만 작성하라: {req.language} ({LANG_LABEL_BY_CODE.get(req.language, '지정 언어')})\n"
        "- 기계 번역투를 피하고, 해당 언어권의 자연스러운 마케팅 문체를 사용하라.\n"
        "- 한 메시지 안에서 언어를 혼용하지 말 것.\n"
        "\n"
        "[길이 제한(비한국어)]\n"
        "- 한국어(ko)는 문자 수 제한만 지킨다.\n"
        "- 그 외 언어는 가능한 한 2~3문장으로 끝내고, 불필요한 설명을 넣지 말라.\n"
        f"- 만약 언어가 공백 기준 단어 계산이 가능한 언어(en/fr/de/es/id/vi/th/ar/hi)라면: title은 10단어 이내, body는 70~75단어 이내로 작성하라.\n"
        "\n"
        "[말맛/자연스러움]\n"
        "- 기계적인 나열(\"~합니다. ~합니다.\") 금지. 문장 길이와 어미를 자연스럽게 섞어 리듬을 만들어라.\n"
        "- 너무 딱딱한 문어체만 쓰지 말고, 일상적인 표현(예: ‘부담 없이’, ‘가볍게’, ‘딱 좋게’, ‘손이 가는’)을 섞어라.\n"
        "- 한 문장에 정보가 과밀해지면 1줄 줄바꿈(\\n) 1회까지 허용한다(기본은 한 단락).\n"
        "- 이모지는 기본 0개. 감성 톤이 필요할 때만 0~1개까지(남발 금지).\n"
        "\n"
        "[구성(권장)]\n"
        "- 공감 1문장(고민/상황을 먼저 짚기) → 해결 느낌 1문장(부담 낮추기) → 제품 1~3개 자연스러운 연결 → 가벼운 CTA 1문장\n"
        "- ‘지금 당장’ 압박/공포 조장 금지. 대신 ‘오늘/이번 주’처럼 부드러운 시간 표현은 가능.\n"
        "\n"
        "[스타일 적용]\n"
        "- style=감성적으로: 감각/기분/루틴의 장면을 1개 넣어라(예: ‘세안 후 당김이 줄어드는 느낌’).\n"
        "- style=부드럽게: 다정하고 낮은 톤, ‘~해요/~해보세요’ 위주.\n"
        "- style=간결하게: 군더더기 제거, 짧은 문장 2~3개로 끝내라.\n"
        "\n"
        "[개인화/상품 반영]\n"
        "- customer_keywords(고민/상황)를 먼저 언급하고, 그에 맞는 제품을 ‘루틴’으로 엮어라.\n"
        "- picked_products의 detail_excerpt에서는 ‘사용감/특징’만 안전하게 요약해라. 성분/효능은 단정하지 말 것.\n"
        "- 가격/할인 정보는 필요할 때만 1회, 정보성으로 짧게.\n"
        "\n"
        "[컴플라이언스]\n"
        "- 의학적 효능/치료/완치/염증 개선을 단정하지 말 것(‘개인차가 있을 수 있어요’, ‘컨디션에 따라’ 같은 완충 표현 사용).\n"
        "- 경쟁사 비방/비교 금지.\n"
        "\n"
        "[채널 가이드]\n"
        "- 푸시(push)/SMS: 스마트폰 알림 길이(짧게). 1~2문장, 핵심만.\n"
        "- 알림톡(kakao): 푸시보다 여유 있게. 3~4문장까지 가능.\n"
        "- 이메일(email): 가장 길게 허용. 4~5문장까지 가능(그래도 장문 설명은 금지).\n"
        "\n"
        "[예시(형식만 참고, 그대로 복사 금지)]\n"
        "- title: \"요즘 건조함, 오늘은 수분부터\"\n"
        "- body: \"요즘 세안 후 당김이 거슬린다면 오늘은 수분감부터 편하게 채워보는 게 좋아요. 가볍게 올라오는 타입으로 시작해보고, 컨디션 좋을 땐 한 번 더 레이어링해도 부담이 덜하거든요. 지금 할인 중인 제품으로 먼저 가볍게 써보세요.\"\n"
    )

    user_prompt = {
        "persona": {"id": req.persona_id, "name_ko": persona_ko, "name_en": persona_en},
        "objective": _objective_kor(req.objective),
        "channel": _channel_kor(req.channel),
        "style": _style_kor(req.style),
        "language": {"code": req.language, "label": LANG_LABEL_BY_CODE.get(req.language, req.language)},
        "price_sensitivity": req.price_sensitivity,
        "brand_tone": {"brand": tone_brand, **tone_pack},
        "customer_notes": req.notes,
        "customer_keywords": req.concerns,
        "persona_rag_context": persona_context,
        "picked_products": products_for_prompt,
        "constraints": {
            "title_max_chars": title_max,
            "body_max_chars": body_max,
            "format": "json_only"
        }
    }

    llm = _make_llm(req.apiKey, req.model)
    try:
        resp = llm.invoke([("system", system), ("user", json.dumps(user_prompt, ensure_ascii=False))])
        raw = getattr(resp, "content", str(resp))
    except Exception as e:
        debug_lines.append(f"LLM error: {e}")
        return JSONResponse(status_code=500, content={"error": "LLM 호출 실패", "debugLog": "\n".join(debug_lines)})

    parsed = _extract_json_object(raw) or {}
    title = str(parsed.get("title") or "").strip()
    body = str(parsed.get("body") or "").strip()

    # Non-Korean length enforcement (marketing copy; channel-aware)
    lang = (req.language or "ko").strip().lower()
    ch = (req.channel or "").strip().lower()
    if lang != "ko":
        # Sentence cap first (prevents verbose explanations)
        max_sents = CHANNEL_SENTENCE_CAP.get(ch, LANG_SENTENCE_LIMITS.get(lang, 3))
        body = _trim_sentences_by_lang(body, lang, max_sents)

        # Word caps for space-delimited languages (channel-aware)
        if lang in LANG_WORD_LIMITS:
            title_w0, body_w0 = LANG_WORD_LIMITS[lang]
            title_w1, body_w1 = CHANNEL_WORD_CAP.get(ch, (title_w0, body_w0))
            title = _truncate_words(title, min(title_w0, title_w1))
            body = _truncate_words(body, min(body_w0, body_w1))

    # If parse failed, attempt one repair pass
    if not title or not body:
        repair_sys = (
            "너는 JSON 포맷을 엄격히 지키는 편집자다.\n"
            f"반드시 JSON만 출력: {{\"title\": string, \"body\": string}}\n"
        )
        repair_user = {
            "raw": raw,
            "requirements": {"title_max_chars": title_max, "body_max_chars": body_max},
        }
        try:
            resp2 = llm.invoke([("system", repair_sys), ("user", json.dumps(repair_user, ensure_ascii=False))])
            raw2 = getattr(resp2, "content", str(resp2))
            parsed2 = _extract_json_object(raw2) or {}
            title = str(parsed2.get("title") or title).strip()
            body = str(parsed2.get("body") or body).strip()
            raw = raw2
        except Exception as e:
            debug_lines.append(f"repair error: {e}")

    # Length enforcement (try LLM shorten, then hard truncate)
    if len(title) > title_max or len(body) > body_max:
        shorten_sys = (
            "너는 마케팅 카피를 글자수에 맞게 다듬는 편집자다.\n"
            "의미/톤은 유지하고, 반드시 JSON만 출력해라.\n"
        )
        shorten_user = {
            "title": title,
            "body": body,
            "title_max": title_max,
            "body_max": body_max,
        }
        try:
            resp3 = llm.invoke([("system", shorten_sys), ("user", json.dumps(shorten_user, ensure_ascii=False))])
            raw3 = getattr(resp3, "content", str(resp3))
            parsed3 = _extract_json_object(raw3) or {}
            title = str(parsed3.get("title") or title).strip()
            body = str(parsed3.get("body") or body).strip()
        except Exception as e:
            debug_lines.append(f"shorten error: {e}")

    # Final hard truncation
    if (req.language or "ko").strip().lower() == "ko":
        title = _truncate_hard(title, title_max)
        body = _truncate_hard(body, body_max)
    else:
        # Keep non-Korean concise: apply char cap as a safety net only
        title = _truncate_hard(title, title_max)
        body = _truncate_hard(body, body_max)
        # Re-apply sentence/word caps after any shorten attempts (channel-aware)
        lang = (req.language or "ko").strip().lower()
        ch = (req.channel or "").strip().lower()
        max_sents = CHANNEL_SENTENCE_CAP.get(ch, LANG_SENTENCE_LIMITS.get(lang, 3))
        body = _trim_sentences_by_lang(body, lang, max_sents)
        if lang in LANG_WORD_LIMITS:
            title_w0, body_w0 = LANG_WORD_LIMITS[lang]
            title_w1, body_w1 = CHANNEL_WORD_CAP.get(ch, (title_w0, body_w0))
            title = _truncate_words(title, min(title_w0, title_w1))
            body = _truncate_words(body, min(body_w0, body_w1))

    debug_lines.append("\n--- persona_docs ---")
    for d in persona_docs:
        debug_lines.append(f"- {d.metadata.get('persona_id')} score_ctx_len={len(d.page_content)}")
    debug_lines.append(f"(pool) brands_top12=" + ", ".join([(p.get('brandName') or '-') for p in pool[:12]]))
    debug_lines.append(f"(pool) unique_brands_top12=" + str(len(set([(p.get('brandName') or '-') for p in pool[:12]]))))
    debug_lines.append("\n--- product_candidates ---")
    for p in cand_products[:8]:
        debug_lines.append(
            f"- [{p.get('brandName')}] {p.get('name')} tier={p.get('brandTier')} priceTier={p.get('priceTier')} giftFit={p.get('giftFit')} fit={p.get('_fit_score', 0):.1f} dist={p.get('_rag_distance'):.3f} price={p.get('price_discounted')}"
        )
    debug_lines.append("\n--- raw_model_output_excerpt ---")
    debug_lines.append((raw or "")[:800])

    return {
        "title": title,
        "body": body,
        "pickedProducts": picked,
        "ragSummary": rag_summary,
        "debugLog": "\n".join(debug_lines),
    }


if __name__ == "__main__":
    # Allow running `python app.py` directly as described in the README.
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
