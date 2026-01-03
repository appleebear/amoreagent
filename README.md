# Amoremall RAG Marketing Message Agent (LangChain + SQLite)

이 프로젝트는 다음 2개 로컬 파일을 **RAG 방식**으로 참조해,
- 페르소나(규칙/톤/스코어) JSON
- 아모레몰 상품 SQLite DB

개인화 마케팅 메시지(제목 ≤ 40자, 본문 ≤ 350자)를 생성하는 **웹 실행형** 프로토타입입니다.

## 0) 준비물
- Python 3.10+
- OpenAI API Key

## 1) 파일 배치
아래 두 파일을 프로젝트 루트에 그대로 두거나, 환경변수로 경로를 지정하세요.

- `persona_logic_v2.json`
- `amoremall.sqlite`

(이 ChatGPT 대화에서는 이미 업로드된 파일을 사용하면 됩니다.)

## 2) 설치
```bash
pip install -r requirements.txt
```

## 3) 환경변수 설정
권장: 서버에서만 키를 보관
```bash
export OPENAI_API_KEY="..."
# 선택
export OPENAI_MODEL="gpt-5.2"
export OPENAI_EMBED_MODEL="text-embedding-3-small"
export SQLITE_PATH="./amoremall.sqlite"
export PERSONA_PATH="./persona_logic_v2.json"
```

## 4) 벡터 인덱스 생성(최초 1회)
```bash
python build_index.py
```

- 상품 수(약 1천개)와 상세 텍스트 길이에 따라 시간이 걸릴 수 있습니다.
- 생성물: `./data/faiss_products/`, `./data/faiss_personas/`

## 5) 서버 실행
```bash
python app.py
```
브라우저에서:
- http://localhost:8000

## 6) 운영 팁(권장)
- **브라우저에 API Key를 넣는 방식은 데모용**입니다. 운영은 반드시 서버-side secret으로.
- 프롬프트 내 의료적 효능/치료 표현은 금지하도록 가드레일이 들어있습니다(완전 차단은 별도 정책/필터 권장).

## 7) 구조
- `build_index.py`: SQLite → Document → (LangChain + OpenAI Embeddings) → FAISS 저장
- `app.py`: FastAPI 서버. RAG(FAISS) → 3개 상품 선정 → LLM 생성 → 길이/JSON 검증
- `static/index.html`: 기존 프로토타입 UI를 유지한 프론트엔드

