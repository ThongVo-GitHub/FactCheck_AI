import os, re, math, pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

# ========================== CẤU HÌNH ==========================
TRAIN_CSV = r"D:/Code_cuaThong/GPPM_2025/train_data.csv"
DEV_CSV   = r"D:/Code_cuaThong/GPPM_2025/dev_data.csv"
TEST_CSV  = r"D:/Code_cuaThong/GPPM_2025/test_data.csv"
OUT_DIR   = r"D:/Code_cuaThong/GPPM_2025/indices"

MIN_WORDS = 4
ADD_STATEMENTS = True  # gộp thêm Statement vào corpus
SAVE_RERANK_READY = True  # lưu thêm file cho reranking

# ========================== HÀM HỖ TRỢ ==========================
def convert_to_list(value):
    import ast
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value

def read_dataset(path):
    df = pd.read_csv(path)
    if "evidence_top5" in df.columns:
        df = df.drop(columns=["evidence_top5"])
    for col in ["splited_sentences", "Evidence_List"]:
        if col in df.columns:
            df[col] = df[col].apply(convert_to_list)
    return df

# ========================== BM25 ==========================
class BM25Okapi:
    def __init__(self, corpus, k1=1.5, b=0.75, epsilon=0.25):
        self.k1, self.b, self.epsilon = k1, b, epsilon
        self.corpus_size = len(corpus)
        if self.corpus_size == 0:
            raise ValueError("Corpus rỗng. Không thể build BM25.")
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_len) / self.corpus_size
        self._initialize(corpus)

    def _initialize(self, corpus):
        nd = {}
        for document in corpus:
            freqs = {}
            for w in document:
                freqs[w] = freqs.get(w, 0) + 1
            self.doc_freqs.append(freqs)
            for w in freqs:
                nd[w] = nd.get(w, 0) + 1
        self._calc_idf(nd)

    def _calc_idf(self, nd):
        idf_sum = 0
        neg = []
        for w, df in nd.items():
            idf = math.log(self.corpus_size - df + 0.5) - math.log(df + 0.5)
            self.idf[w] = idf
            idf_sum += idf
            if idf < 0:
                neg.append(w)
        self.average_idf = idf_sum / max(len(self.idf), 1)
        eps = self.epsilon * self.average_idf
        for w in neg:
            self.idf[w] = eps

    def get_scores(self, query_tokens):
        score = np.zeros(self.corpus_size, dtype=np.float32)
        doc_len = np.array(self.doc_len, dtype=np.float32)
        for q in query_tokens:
            q_freq = np.array([doc.get(q, 0) for doc in self.doc_freqs], dtype=np.float32)
            denom = q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += (self.idf.get(q, 0.0)) * (q_freq * (self.k1 + 1) / np.where(denom == 0, 1, denom))
        return score

# ========================== TIỀN XỬ LÝ TV ==========================
VI_CHARS = "a-zA-Z0-9àáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ"
VI_KEEP  = f"[^{VI_CHARS} ]"

def clean_text(s: str) -> str:
    s = str(s).replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def tokenize_vi(s: str):
    s = s.lower()
    s = re.sub(VI_KEEP, " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.split()

# ========================== MAIN ==========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Kiểm tra file
    for p in [TRAIN_CSV, DEV_CSV, TEST_CSV]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Không thấy file: {p}")

    print("📖 Đọc dữ liệu…")
    train = read_dataset(TRAIN_CSV)
    dev   = read_dataset(DEV_CSV)
    test  = read_dataset(TEST_CSV)
    all_df = pd.concat([train, dev, test], ignore_index=True)

    # Gom documents
    print("🧩 Gom documents từ 'splited_sentences'…")
    documents = []
    if "splited_sentences" in all_df.columns:
        for lst in tqdm(all_df["splited_sentences"], total=len(all_df)):
            if isinstance(lst, list):
                for s in lst:
                    if isinstance(s, str):
                        s2 = clean_text(s)
                        if len(s2.split()) >= MIN_WORDS:
                            documents.append(s2)

    # Thêm Statement vào corpus
    if ADD_STATEMENTS and "Statement" in all_df.columns:
        print("➕ Thêm Statement vào corpus…")
        for s in tqdm(all_df["Statement"].fillna("").astype(str).tolist()):
            s2 = clean_text(s)
            if len(s2.split()) >= MIN_WORDS:
                documents.append(s2)

    # Loại trùng
    documents = list(dict.fromkeys(documents))
    print(f"📊 Tổng số documents: {len(documents):,}")

    # Ghi file documents.txt
    doc_path = os.path.join(OUT_DIR, "documents.txt")
    with open(doc_path, "w", encoding="utf-8") as f:
        for d in documents:
            f.write(d + "\n")
    print(f"✅ Ghi xong {doc_path}")

    # Tokenize corpus
    print("🔤 Tokenize corpus…")
    tokenized_corpus = [tokenize_vi(d) for d in tqdm(documents, total=len(documents))]

    # Build BM25
    print("⚙️ Build BM25 Okapi…")
    bm25 = BM25Okapi(tokenized_corpus)

    pkl_path = os.path.join(OUT_DIR, "bm25_index.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(bm25, f)
    print(f"✅ Lưu xong {pkl_path}")

    # (Tùy chọn) Chuẩn bị file cho bước Semantic Reranking
    if SAVE_RERANK_READY:
        rerank_ready = {
            "documents": documents,
            "tokenized_corpus": tokenized_corpus
        }
        pkl_rerank = os.path.join(OUT_DIR, "rerank_ready.pkl")
        with open(pkl_rerank, "wb") as f:
            pickle.dump(rerank_ready, f)
        txt_rerank = os.path.join(OUT_DIR, "rerank_ready.txt")
        with open(txt_rerank, "w", encoding="utf-8") as f:
            for d in documents:
                f.write(d + "\n")
        print(f"💾 Lưu dữ liệu sẵn cho Semantic Reranking: {pkl_rerank}, {txt_rerank}")

    print("🎉 Hoàn tất xây dựng BM25 và dữ liệu reranking!")

# ==========================================================
if __name__ == "__main__":
    main()
