import os, pickle, re
from rank_bm25 import BM25Okapi

BASE_DIR   = os.path.dirname(__file__)
INDICES_DIR= os.path.join(BASE_DIR, "indices")
DOCS_TXT   = os.path.join(INDICES_DIR, "documents.txt")
BM25_PKL   = os.path.join(INDICES_DIR, "bm25_index.pkl")

def tok(x: str):
    return re.sub(r"\s+", " ", x.strip().lower()).split()

def main():
    assert os.path.exists(DOCS_TXT), f"Không thấy {DOCS_TXT}"
    with open(DOCS_TXT, "r", encoding="utf-8") as f:
        docs = [ln.strip() for ln in f if ln.strip()]
    assert len(docs) > 0, "documents.txt rỗng!"
    corpus = [tok(d) for d in docs]
    bm25 = BM25Okapi(corpus)
    with open(BM25_PKL, "wb") as f:
        pickle.dump(bm25, f)
    print(f"✅ Built BM25 with {len(docs)} docs -> {BM25_PKL}")

if __name__ == "__main__":
    main()
