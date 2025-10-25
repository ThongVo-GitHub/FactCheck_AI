

import os
import re
import json
import pickle
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from transformers import AutoModel, AutoTokenizer, AutoConfig

# BM25
from rank_bm25 import BM25Okapi

# ================== CONFIG ==================
BASE_DIR = os.path.dirname(__file__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# PhoBERT classifier (fine-tuned)
CKPT_DIR      = os.path.join(BASE_DIR, "checkpoints")
MODEL_PTH     = os.path.join(CKPT_DIR, "phobert_best.pth")
TOKENIZER_DIR = os.path.join(CKPT_DIR, "tokenizer")

# BM25 index paths (can be overridden via ENV for portability)
INDICES_DIR = os.path.join(BASE_DIR, "indices")
BM25_PKL    = os.getenv("BM25_PKL", os.path.join(INDICES_DIR, "bm25_index.pkl"))
DOCS_TXT    = os.getenv("DOCS_TXT", os.path.join(INDICES_DIR, "documents.txt"))

# SBERT model (reranker)
SBERT_NAME  = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"

# LoRA (optional, via PEFT) — set HF_TOKEN in your environment if the base model is gated
LORA_DIR        = os.path.join(BASE_DIR, "outputs", "lora_adapter")     # where your adapter is saved
BASE_LLM        = os.getenv("BASE_LLM", "mistralai/Mistral-7B-Instruct-v0.2")
HF_TOKEN        = os.getenv("HF_TOKEN", None)                            # setx HF_TOKEN "hf_xxx" on Windows
USE_4BIT        = True                                                   # requires CUDA + bitsandbytes
LLM             = None
LLM_TOK         = None

MAX_LEN          = 256
BM25_CANDIDATES  = 50
MAX_EVID         = 5
DEFAULT_TOP_EVID = 5
LABEL_MAP        = {0: "Supported", 1: "Refuted", 2: "Not Enough Information"}

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"


# ================== PhoBERT CLASSIFIER ==================
class PhoBERTClassifier(nn.Module):
    def __init__(self, phobert, num_classes=3, dropout=0.3):
        super().__init__()
        self.phobert = phobert
        self.dropout = nn.Dropout(dropout)
        self.linear  = nn.Linear(self.phobert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.phobert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        cls = out.last_hidden_state[:, 0, :]    # [CLS]
        logits = self.linear(self.dropout(cls))
        return logits


def load_phobert_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    cfg  = AutoConfig.from_pretrained("vinai/phobert-large")
    base = AutoModel.from_pretrained("vinai/phobert-large", config=cfg)
    model = PhoBERTClassifier(base, num_classes=3)
    state = torch.load(MODEL_PTH, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model, tokenizer


# ================== BM25 LOADING ==================
def _simple_tokenize(s: str):
    s = re.sub(r"\s+", " ", s.strip().lower())
    return s.split()

def _build_bm25_from_docs(docs: List[str]):
    tokenized_corpus = [_simple_tokenize(d) for d in docs]
    return BM25Okapi(tokenized_corpus)

def load_bm25():
    try:
        if not os.path.isfile(DOCS_TXT):
            print(f"⚠️ Không thấy documents: {DOCS_TXT}")
            return None, None

        with open(DOCS_TXT, "r", encoding="utf-8") as f:
            documents = [ln.strip() for ln in f if ln.strip()]

        bm25 = None
        if os.path.isfile(BM25_PKL):
            try:
                with open(BM25_PKL, "rb") as f:
                    bm25 = pickle.load(f)
                _ = bm25.get_scores(["test"])
                print(f"✅ BM25 loaded from {BM25_PKL}. num_docs={len(documents)}")
            except Exception as e:
                print(f"❌ BM25 pickle lỗi ({BM25_PKL}) → sẽ build lại: {e}")

        if bm25 is None:
            print("⏳ Building BM25 from documents.txt ...")
            bm25 = _build_bm25_from_docs(documents)
            try:
                os.makedirs(os.path.dirname(BM25_PKL), exist_ok=True)
                with open(BM25_PKL, "wb") as f:
                    pickle.dump(bm25, f)
                print(f"💾 Saved BM25 to {BM25_PKL}")
            except Exception as e:
                print("⚠️ Không thể lưu BM25 pickle:", e)

        return bm25, documents
    except Exception as e:
        print("⚠️ BM25 load error:", e)
        return None, None


def retrieve_bm25_candidates(bm25, documents: List[str], query: str, k=BM25_CANDIDATES):
    if bm25 is None or not documents:
        return []
    tokens = _simple_tokenize(query)
    try:
        scores = bm25.get_scores(tokens)
    except Exception as e:
        print(f"[BM25] get_scores error: {e}")
        return []
    idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    out = [{"text": documents[i], "bm25_score": float(scores[i])} for i in idx if float(scores[i]) > 0.0]
    if not out:
        print(f"[BM25] No positive hits for: '{query}'. Top raw:", [float(scores[i]) for i in idx])
    return out


# ================== SBERT Reranker ==================
tokenizer_sbert = AutoTokenizer.from_pretrained(SBERT_NAME)
model_sbert = AutoModel.from_pretrained(SBERT_NAME).to(device)
model_sbert.eval()
SBERT_READY = True

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * input_mask_expanded).sum(1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

@torch.inference_mode()
def rerank_with_sbert(claim: str, candidates: List[dict], top_k=MAX_EVID):
    if not candidates:
        return []
    texts = [claim] + [c["text"] for c in candidates]
    enc = tokenizer_sbert(texts, padding=True, truncation=True, return_tensors="pt", max_length=MAX_LEN)
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model_sbert(**enc)
    emb = mean_pooling(out, enc["attention_mask"])
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)

    claim_vec = emb[0:1]
    cand_vecs = emb[1:]
    cos = torch.matmul(claim_vec, cand_vecs.T).squeeze(0).cpu().tolist()

    ranked = []
    for i, c in enumerate(candidates):
        ranked.append({
            "text": c["text"],
            "sbert_score": float(cos[i]),
            "bm25_score": float(c.get("bm25_score", 0.0)),
        })
    ranked.sort(key=lambda x: x["sbert_score"], reverse=True)
    return ranked[:top_k]


# ================== FASTAPI APP ==================
app = FastAPI(title="VietFactAI_THL API", version="1.8")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # limit this when deploying
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL, TOKENIZER = load_phobert_model_and_tokenizer()
BM25, DOCUMENTS  = load_bm25()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(device),
        "bm25_loaded": BM25 is not None,
        "sbert_loaded": SBERT_READY,
        "num_docs": len(DOCUMENTS) if DOCUMENTS else 0,
        "lora_loaded": LLM is not None,
        "lora_dir": LORA_DIR,
        "base_llm": BASE_LLM,
        "bm25_pkl": BM25_PKL,
        "docs_txt": DOCS_TXT,
    }


# --------- DEBUG (giúp soi vì sao không có evidence) ----------
@app.get("/debug/bm25_status")
def debug_bm25_status():
    exists_pkl = os.path.isfile(BM25_PKL)
    exists_txt = os.path.isfile(DOCS_TXT)
    num_docs   = len(DOCUMENTS) if DOCUMENTS else 0
    size_pkl   = os.path.getsize(BM25_PKL) if exists_pkl else 0
    size_txt   = os.path.getsize(DOCS_TXT) if exists_txt else 0
    return {
        "BM25_PKL": BM25_PKL, "exists_pkl": exists_pkl, "size_pkl": size_pkl,
        "DOCS_TXT": DOCS_TXT, "exists_txt": exists_txt, "size_txt": size_txt,
        "num_docs": num_docs,
    }

@app.get("/debug/bm25")
def debug_bm25(q: str, k: int = 5):
    if BM25 is None or not DOCUMENTS:
        return {"ok": False, "reason": "BM25 chưa load hoặc documents rỗng."}
    toks = _simple_tokenize(q)
    scores = BM25.get_scores(toks)
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return {
        "ok": True,
        "query": q,
        "top": [
            {"rank": i+1, "score": float(scores[j]), "text": DOCUMENTS[j]}
            for i, j in enumerate(idxs)
        ],
    }

@app.get("/debug/retrieve")
def debug_retrieve(q: str, k: int = 5):
    if BM25 is None or not DOCUMENTS:
        return {"ok": False, "reason": "BM25 chưa load hoặc documents rỗng."}
    bm25_cands = retrieve_bm25_candidates(BM25, DOCUMENTS, q, k=BM25_CANDIDATES)
    if not bm25_cands:
        return {"ok": True, "bm25_top": [], "sbert_top": [], "note": "BM25 không có hit dương (score>0)."}
    sbert_top = rerank_with_sbert(q, bm25_cands, top_k=k) if SBERT_READY else []
    return {"ok": True, "bm25_top": bm25_cands[:k], "sbert_top": sbert_top}


# ================== LLM (PEFT LoRA) ==================
def load_llm_peft():
    """Load base LLM + gắn LoRA adapter bằng PEFT (ổn định trên Windows)."""
    global LLM, LLM_TOK
    if not os.path.isdir(LORA_DIR):
        print(f"ℹ️ LoRA adapter folder không tồn tại: {LORA_DIR}")
        return None, None

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        quantization_config = None
        if torch.cuda.is_available() and USE_4BIT:
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                print("✅ Đang dùng 4bit quantization (bitsandbytes).")
            except Exception as e:
                print("⚠️ Không thể khởi tạo BitsAndBytesConfig, sẽ load full precision:", e)

        # --- Load tokenizer ---
        tok = AutoTokenizer.from_pretrained(BASE_LLM, use_fast=True, token=HF_TOKEN, trust_remote_code=True)

        # --- Load base model ---
        base_kwargs = {
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            "device_map": "auto" if torch.cuda.is_available() else None,
            "trust_remote_code": True,
            "token": HF_TOKEN,
        }
        if quantization_config is not None:
            base_kwargs["quantization_config"] = quantization_config

        base = AutoModelForCausalLM.from_pretrained(BASE_LLM, **base_kwargs)

        # --- Attach LoRA adapter ---
        model = PeftModel.from_pretrained(base, LORA_DIR, token=HF_TOKEN)
        model.eval()
        LLM, LLM_TOK = model, tok
        print(f"✅ LoRA (PEFT) loaded thành công. Base={BASE_LLM} | Adapter={LORA_DIR}")
        return LLM, LLM_TOK

    except Exception as e:
        print("⚠️ Không thể load LoRA qua PEFT:", e)
        return None, None

def ensure_llm_loaded():
    global LLM, LLM_TOK
    if LLM is None:
        LLM, LLM_TOK = load_llm_peft()
    return LLM is not None

def build_prompt_vi(claim: str, evid_sentences: list):
    evid_block = "\n".join([f"- {s}" for s in evid_sentences])
    return f"""Bạn là trợ lý kiểm chứng. Dưới đây là các đoạn bằng chứng đã truy xuất và một phát biểu cần kiểm chứng.

### Evidence:
{evid_block}

### Claim:
{claim}

Hãy chọn 1–3 câu bằng chứng phù hợp nhất, viết ngắn gọn lý do,
và trả về JSON:
{{
    "label": 0|1|2,
    "evidence": ["câu 1","câu 2"],
    "rationale": "giải thích ngắn"
}}
""".strip()

@torch.inference_mode()
def llm_generate_json(prompt: str):
    if LLM is None:
        return None
    try:
        # letting PEFT model place tensors where it wants (device_map="auto")
        inputs = LLM_TOK(prompt, return_tensors="pt")
        out = LLM.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=128,
            eos_token_id=LLM_TOK.eos_token_id,
        )
        text = LLM_TOK.decode(out[0], skip_special_tokens=True)
        start = text.index("{")
        return json.loads(text[start:])
    except Exception as e:
        print("LLM generate error:", e)
        return None


# ================== API Schemas ==================
class PredictIn(BaseModel):
    text: str
    k: Optional[int] = DEFAULT_TOP_EVID
    use_explain: Optional[bool] = False

class PredictOut(BaseModel):
    label_id: int
    label: str
    probs: List[float]
    evidences: List[dict]            # [{text, sbert_score, bm25_score}]
    explanation: Optional[dict] = None  # {"evidence":[...], "rationale":"...", "label_id":int, "label":str}

class EvidenceIn(BaseModel):
    text: str
    k: int = 5
    use_explain: Optional[bool] = False

class EvidenceOut(BaseModel):
    candidates: List[dict]
    explanation: Optional[dict] = None


# ================== PREDICT (full pipeline) ==================
@torch.inference_mode()
@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn):
    text = (payload.text or "").strip()
    top_k = payload.k or DEFAULT_TOP_EVID

    if not text:
        return PredictOut(label_id=2, label=LABEL_MAP[2], probs=[0.0, 0.0, 1.0], evidences=[], explanation=None)

    # [2] PhoBERT classification
    enc = TOKENIZER.encode_plus(text, max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    logits = MODEL(input_ids=input_ids, attention_mask=attention_mask)
    probs  = F.softmax(logits, dim=-1).squeeze(0).tolist()
    label_id = int(torch.argmax(logits, dim=-1).item())

    # [1] Retrieve evidence: BM25 -> SBERT rerank
    bm25_candidates = retrieve_bm25_candidates(BM25, DOCUMENTS, text, k=BM25_CANDIDATES)
    evidences = rerank_with_sbert(text, bm25_candidates, top_k=top_k) if (bm25_candidates and SBERT_READY) else []

    # [3] Optional LoRA explanation (PEFT)
    explanation = None
    if payload.use_explain and evidences and ensure_llm_loaded():
        prompt = build_prompt_vi(text, [e["text"] for e in evidences])
        res = llm_generate_json(prompt)
        if res:
            expl_label_id = res.get("label")
            explanation = {
                "evidence": res.get("evidence", []),
                "rationale": res.get("rationale", ""),
                "label_id": expl_label_id if isinstance(expl_label_id, int) else None,
                "label": LABEL_MAP.get(expl_label_id) if isinstance(expl_label_id, int) else None,
            }

    return PredictOut(
        label_id   = label_id,
        label      = LABEL_MAP[label_id],
        probs      = [float(p) for p in probs],
        evidences  = evidences,
        explanation= explanation,
    )


# ================== EVIDENCE ONLY ==================
@torch.inference_mode()
@app.post("/evidence", response_model=EvidenceOut)
def evidence_api(payload: EvidenceIn):
    claim = (payload.text or "").strip()
    top_k = payload.k or DEFAULT_TOP_EVID
    if not claim:
        return EvidenceOut(candidates=[], explanation=None)

    bm25_list = retrieve_bm25_candidates(BM25, DOCUMENTS, claim, k=BM25_CANDIDATES)
    if not bm25_list or not SBERT_READY:
        return EvidenceOut(candidates=[], explanation=None)

    candidates = rerank_with_sbert(claim, bm25_list, top_k=top_k)

    explanation = None
    if payload.use_explain and candidates and ensure_llm_loaded():
        prompt = build_prompt_vi(claim, [c["text"] for c in candidates])
        res = llm_generate_json(prompt)
        if res:
            expl_label_id = res.get("label")
            explanation = {
                "evidence": res.get("evidence", []),
                "rationale": res.get("rationale", ""),
                "label_id": expl_label_id if isinstance(expl_label_id, int) else None,
                "label": LABEL_MAP.get(expl_label_id) if isinstance(expl_label_id, int) else None,
            }

    return EvidenceOut(candidates=candidates, explanation=explanation)


def _extract_json_block(text: str) -> dict | None:
    """
    Tìm khối JSON đầu tiên trong output, parse an toàn.
    """
    try:
        start = text.index("{")
        # tìm ngoặc đóng phù hợp
        stack = 0
        end = None
        for i, ch in enumerate(text[start:], start):
            if ch == "{": stack += 1
            elif ch == "}":
                stack -= 1
                if stack == 0:
                    end = i + 1
                    break
        if end:
            return json.loads(text[start:end])
    except Exception:
        pass
    # fallback nhẹ: tìm mảng "evidence": [...]
    try:
        m = re.search(r'"evidence"\s*:\s*\[(.*?)\]', text, flags=re.S)
        if m:
            arr = m.group(1)
            # cắt các chuỗi giữa dấu ngoặc kép
            items = re.findall(r'"(.*?)"', arr)
            return {"evidence": items}
    except Exception:
        pass
    return None

def _place_on_model_device(inputs: dict):
    """
    Đặt input tensors lên device của model nếu có (hỗ trợ device_map='auto').
    """
    try:
        dev = next(LLM.parameters()).device
        return {k: v.to(dev) for k, v in inputs.items()}
    except Exception:
        return inputs

@torch.inference_mode()
def llm_select_evidence(claim: str, candidate_sentences: list[str]) -> dict | None:
    """
    Cho LLM (LoRA-PEFT) chọn evidence + rationale từ danh sách đã rerank (BM25+SBERT).
    Trả về dict dạng: {"label": int?, "evidence": [...], "rationale": "..."} hoặc None.
    """
    if LLM is None or LLM_TOK is None:
        return None

    prompt = build_prompt_vi(claim, candidate_sentences)
    try:
        inputs = LLM_TOK(prompt, return_tensors="pt")
        inputs = _place_on_model_device(inputs)

        gen = LLM.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=256,
            eos_token_id=getattr(LLM_TOK, "eos_token_id", None),
        )
        text = LLM_TOK.decode(gen[0], skip_special_tokens=True)
        parsed = _extract_json_block(text)
        return parsed
    except Exception as e:
        print("LLM generate error:", e)
        return None
    
# ====== Chỉ sinh bằng chứng bằng LoRA (tùy chọn) ======
class LoraEvidenceIn(BaseModel):
    text: str
    k: int = 5                  # số câu candidate chuyển vào LLM

class LoraEvidenceOut(BaseModel):
    evidences: List[str] = []   # chỉ trả về câu bằng chứng
    rationale: str | None = None
    label_id: int | None = None
    label: str | None = None
    # debug thêm nếu cần
    bm25_top: List[str] = []
    sbert_top: List[str] = []

@torch.inference_mode()
@app.post("/evidence_only_lora", response_model=LoraEvidenceOut)
def evidence_only_lora(payload: LoraEvidenceIn):
    claim = (payload.text or "").strip()
    top_k = max(1, int(payload.k or 5))

    # guard
    if not claim:
        return LoraEvidenceOut(evidences=[])

    if BM25 is None or not DOCUMENTS:
        return LoraEvidenceOut(evidences=[])

    # Lấy ứng viên bằng BM25
    bm25_candidates = retrieve_bm25_candidates(BM25, DOCUMENTS, claim, k=BM25_CANDIDATES)
    if not bm25_candidates:
        return LoraEvidenceOut(evidences=[])

    # SBERT rerank → danh sách câu
    if SBERT_READY:
        reranked = rerank_with_sbert(claim, bm25_candidates, top_k=top_k)
        cand_texts = [r["text"] for r in reranked]
    else:
        cand_texts = [c["text"] for c in bm25_candidates[:top_k]]

    # Đảm bảo LoRA đã load
    if not ensure_llm_loaded():
        return LoraEvidenceOut(evidences=cand_texts[: min(2, len(cand_texts))])

    # Cho LLM chọn evidence
    res = llm_select_evidence(claim, cand_texts)
    if not res:
        # fallback: trả thẳng top-k candidate
        return LoraEvidenceOut(evidences=cand_texts)

    # Chuẩn hoá output
    out = LoraEvidenceOut(
        evidences=[str(x).strip() for x in res.get("evidence", []) if str(x).strip()],
        rationale=res.get("rationale", None),
        bm25_top=[c["text"] for c in bm25_candidates[:min(5, len(bm25_candidates))]],
        sbert_top=cand_texts[:min(5, len(cand_texts))],
    )
    if isinstance(res.get("label"), int):
        out.label_id = res["label"]
        out.label = LABEL_MAP.get(res["label"])

    # nếu LLM không trả evidence thì rơi về top-k
    if not out.evidences:
        out.evidences = cand_texts[:min(top_k, len(cand_texts))]

    return out
