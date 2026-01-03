import os
import json
import sqlite3
from typing import Any, Dict, List
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

SQLITE_PATH = os.getenv("SQLITE_PATH", "./amoremall.sqlite")
PERSONA_PATH = os.getenv("PERSONA_PATH", "./persona_logic_v2.json")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

FAISS_PRODUCTS_DIR = os.getenv("FAISS_PRODUCTS_DIR", "./data/faiss_products")
FAISS_PERSONAS_DIR = os.getenv("FAISS_PERSONAS_DIR", "./data/faiss_personas")

def load_persona_logic(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def connect_sqlite(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn

def build_product_docs(conn: sqlite3.Connection) -> List[Document]:
    rows = conn.execute(
        """SELECT onlineProdSn, onlineProdCode, name, brandName, categoryTitle,
                  categoryLarge, categoryMiddle, categorySmall,
                  price_before, price_discounted, discount_rate, url,
                  disclosures_json, detail_text
           FROM products"""
    ).fetchall()

    # Chunking을 과도하게 하면(특히 상세설명이 긴 브랜드/상품) 검색 상위 후보를 독식하는 문제가 생김.
    # 따라서 chunk 크기를 키우고, product당 chunk 수를 제한해 브랜드 쏠림을 완화한다.
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=120)

    # 아모레 브랜드/라인업을 간단히 티어링(임베딩용 시그널). 필요시 확장 가능.
    brand_tier_map = {
        # Luxury / Prestige
        "설화수": "luxury",
        "헤라": "prestige",
        "프리메라": "premium",
        "바이탈뷰티": "premium",
        # Daily / Mass
        "라네즈": "daily",
        "이니스프리": "daily",
        "에뛰드": "daily",
        "마몽드": "daily",
        "한율": "daily",
        # Derma / Sensitive
        "에스트라": "derma",
        "일리윤": "derma",
        "라보에이치": "derma",
        # Tools / Accessories (예: 브러시)
        "피카소": "tools",
    }

    def price_tier(price: Any) -> str:
        try:
            p = float(price)
        except Exception:
            return "unknown"
        # 단순 휴리스틱: 운영 환경에 맞춰 조정 가능
        if p >= 100000:
            return "high"
        if p >= 50000:
            return "mid"
        return "low"

    def gift_fit(name: str, cat: str, price: Any, disc: Any) -> str:
        text = f"{name} {cat}".lower()
        # 세트/기프트/스페셜 키워드 우선
        if any(k in text for k in ["세트", "기프트", "선물", "홀리데이", "리미티드", "스페셜", "키트"]):
            return "high"
        # 아주 저가(휴리스틱)는 선물 추천에 신중
        pt = price_tier(price)
        if pt == "low":
            return "low"
        return "medium"

    def compact(text: str, limit: int) -> str:
        t = (text or "").strip().replace("\r", "")
        # 공백 정리
        while "\n\n\n" in t:
            t = t.replace("\n\n\n", "\n\n")
        if len(t) <= limit:
            return t
        return t[:limit] + "…"

    docs: List[Document] = []

    for r in rows:
        meta = {
            "doc_type": "product",
            "onlineProdSn": int(r["onlineProdSn"]) if r["onlineProdSn"] is not None else None,
            "onlineProdCode": r["onlineProdCode"],
            "name": r["name"],
            "brandName": r["brandName"],
            "categoryTitle": r["categoryTitle"],
            "categoryLarge": r["categoryLarge"],
            "categoryMiddle": r["categoryMiddle"],
            "categorySmall": r["categorySmall"],
            "price_before": r["price_before"],
            "price_discounted": r["price_discounted"],
            "discount_rate": r["discount_rate"],
            "url": r["url"],
            "brandTier": None,
            "priceTier": None,
            "giftFit": None,
        }

        disclosures = compact((r["disclosures_json"] or ""), 700)
        detail = compact((r["detail_text"] or ""), 1600)

        bname = (r["brandName"] or "").strip()
        tier = brand_tier_map.get(bname, "unknown")
        ptier = price_tier(r["price_discounted"])
        gfit = gift_fit(r["name"] or "", r["categoryTitle"] or "", r["price_discounted"], r["discount_rate"])

        meta["brandTier"] = tier
        meta["priceTier"] = ptier
        meta["giftFit"] = gfit

        base = (
            f"[doc_type] product\n"
            f"[브랜드] {bname}\n"
            f"[브랜드티어] {tier}\n"
            f"[가격대] {ptier}\n"
            f"[선물적합도(휴리스틱)] {gfit}\n"
            f"[상품명] {r['name']}\n"
            f"[카테고리] {r['categoryTitle']} ({r['categoryLarge']} > {r['categoryMiddle']} > {r['categorySmall']})\n"
            f"[가격] 할인가 {r['price_discounted']} / 정가 {r['price_before']} / 할인율 {r['discount_rate']}\n"
            f"[핵심요약] 고객 니즈(건조/민감/탄력/미백/모공/피지/메이크업 등)와 상황(선물/입문/리필/세트)에 맞춰 추천 근거를 만들 것\n"
            f"[고시/주의(요약)] {disclosures}\n"
            f"[상세설명(요약)] {detail}\n"
            f"[URL] {r['url']}\n"
        )

        chunks = splitter.split_text(base)
        # product당 chunk 수를 제한해 특정 브랜드/상품이 후보군을 독식하는 현상을 완화
        max_chunks_per_product = 2
        chunks = chunks[:max_chunks_per_product]

        for i, ch in enumerate(chunks):
            m = dict(meta)
            m["chunk"] = i
            docs.append(Document(page_content=ch, metadata=m))

    return docs

def summarize_persona_for_embedding(pid: str, p: Dict[str, Any], persona_logic: Dict[str, Any]) -> str:
    # persona_logic에는 점수 규칙(weights_by_persona)이 포함되어 있어,
    # embedding용 문서로 '핵심 시그널'을 텍스트화해서 저장한다.
    weights = (persona_logic.get("scoring_model_full_variables", {}) or {}).get("weights_by_persona", {}) or {}
    rules = weights.get(pid, []) or []
    # 상위 positive 규칙을 뽑아 텍스트화
    pos = []
    neg = []
    for r in rules:
        sc = r.get("score", 0)
        if sc >= 3:
            pos.append(r)
        if sc <= -1:
            neg.append(r)
    pos = sorted(pos, key=lambda x: x.get("score", 0), reverse=True)[:10]
    neg = sorted(neg, key=lambda x: x.get("score", 0))[:5]

    def rule_to_text(r):
        field = r.get("field")
        if "equals" in r:
            cond = f"=={r['equals']}"
        elif "in" in r:
            cond = f"in({', '.join(map(str, r['in']))})"
        elif "contains_any" in r:
            cond = f"contains_any({', '.join(map(str, r['contains_any']))})"
        elif "range" in r:
            cond = f"range({r['range'][0]}~{r['range'][1]})"
        elif r.get("exists") is True:
            cond = "exists"
        else:
            cond = "(condition)"
        return f"- {field} {cond} (+{r.get('score')})"

    lines = [
        f"[persona_id] {pid}",
        f"[name_ko] {p.get('name_ko','')}",
        f"[name_en] {p.get('name_en','')}",
        "",
        "[추천 정책 힌트]",
        "- 선물 목적이면: giftFit=high/medium 우선, 너무 저가(priceTier=low)는 신중",
        "- 럭셔리/프레스티지 성향이면: brandTier=luxury/prestige/premium 우선",
        "- 민감/피부장벽 성향이면: brandTier=derma 우선",
        "",
        "[핵심 시그널(상위 가중치)]",
    ]
    lines += [rule_to_text(r) for r in pos]
    if neg:
        lines += ["", "[비선호/패널티 시그널]"] + [rule_to_text(r) for r in neg]
    return "\n".join(lines)

def build_persona_docs(persona_logic: Dict[str, Any]) -> List[Document]:
    personas = persona_logic.get("personas") or {}
    docs: List[Document] = []
    for pid, p in personas.items():
        text = summarize_persona_for_embedding(pid, p, persona_logic)
        docs.append(Document(page_content=text, metadata={"doc_type": "persona", "persona_id": pid, "name_ko": p.get("name_ko",""), "name_en": p.get("name_en","")}))
    return docs

def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY 환경변수를 설정해야 인덱싱이 가능합니다.")

    if not os.path.exists(SQLITE_PATH):
        raise SystemExit(f"SQLite 파일을 찾을 수 없습니다: {SQLITE_PATH}")
    if not os.path.exists(PERSONA_PATH):
        raise SystemExit(f"Persona JSON 파일을 찾을 수 없습니다: {PERSONA_PATH}")

    persona_logic = load_persona_logic(PERSONA_PATH)
    conn = connect_sqlite(SQLITE_PATH)

    print("[1/4] 상품 문서 생성 중...")
    product_docs = build_product_docs(conn)
    print(f"  - product chunks: {len(product_docs)}")

    print("[2/4] 페르소나 문서 생성 중...")
    persona_docs = build_persona_docs(persona_logic)
    print(f"  - persona docs: {len(persona_docs)}")

    conn.close()

    embeddings = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)

    print("[3/4] FAISS 상품 인덱스 생성/저장...")
    import time
    if not product_docs:
        print("  - 생성할 상품 문서가 없어 스킵합니다.")
    else:
        batch_size = 80  # 100~300 사이 권장
        vs_prod = None
        for i in range(0, len(product_docs), batch_size):
            batch = product_docs[i : i + batch_size]
            if vs_prod is None:
                vs_prod = FAISS.from_documents(batch, embeddings)
            else:
                vs_prod.add_documents(batch)
            time.sleep(1.2)
        os.makedirs(FAISS_PRODUCTS_DIR, exist_ok=True)
        vs_prod.save_local(FAISS_PRODUCTS_DIR)
        print(f"  - 저장 완료 ({FAISS_PRODUCTS_DIR})")

    print("[4/4] FAISS 페르소나 인덱스 생성/저장...")
    if not persona_docs:
        print("  - 생성할 페르소나 문서가 없어 스킵합니다.")
    else:
        vs_pers = FAISS.from_documents(persona_docs, embeddings)
        os.makedirs(FAISS_PERSONAS_DIR, exist_ok=True)
        vs_pers.save_local(FAISS_PERSONAS_DIR)
        print(f"  - 저장 완료 ({FAISS_PERSONAS_DIR})")

    print("완료.")
    print(f"- products index: {FAISS_PRODUCTS_DIR}")
    print(f"- personas index: {FAISS_PERSONAS_DIR}")

if __name__ == "__main__":
    main()
