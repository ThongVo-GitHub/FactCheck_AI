# app.py
import os
import json
import requests
import streamlit as st

# ================== CẤU HÌNH CHUNG ==================
st.set_page_config(page_title="VietFactAI_THL", page_icon="🕵️", layout="wide")
DEFAULT_API_URL = os.getenv("VIETFACT_API_URL", "http://127.0.0.1:8000")

# ========= TIỆN ÍCH GIAO DIỆN =========
def get_page():
    # Streamlit mới: st.query_params là dict-like
    try:
        return st.query_params.get("page", "home")
    except Exception:
        # fallback nếu bản Streamlit cũ
        return "home"

def go_page(name: str):
    try:
        st.query_params["page"] = name
    except Exception:
        pass

def ui_css(theme="green"):
    if theme == "blue":
        bg = "radial-gradient(1000px 800px at 50% 50%, #90CAF9 0%, #E3F2FD 50%, #BBDEFB 100%)"
        header_color = "#1565C0"
        accent = "#0D47A1"
    elif theme == "red":
        bg = "radial-gradient(1000px 800px at 50% 50%, #EF5350 0%, #FFCDD2 50%, #E57373 100%)"
        header_color = "#B71C1C"
        accent = "#880E4F"
    else:
        bg = "radial-gradient(1000px 800px at 50% 50%, #A5D6A7 0%, #E8F5E9 50%, #C8E6C9 100%)"
        header_color = "#2E7D32"
        accent = "#33691E"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [data-testid="stAppViewContainer"] {{
        font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
    }}
    [data-testid="stAppViewContainer"] {{
        background: {bg};
        color: #1B1B1B;
        transition: background 1s ease-in-out;
    }}
    .vx-header {{ text-align: center; margin: 8px 0 18px 0; }}
    .vx-header h1 {{ font-size: 36px; margin: 0; letter-spacing: .3px; color: {header_color}; }}
    .vx-sub {{ opacity: .8; margin-top: 6px; color: {accent}; }}

    .vx-card {{
        background: linear-gradient(180deg, rgba(255,255,255,0.6), rgba(255,255,255,0.3));
        border: 1px solid rgba(76,175,80,0.25);
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
        backdrop-filter: blur(8px);
        border-radius: 16px;
        padding: 18px 18px;
        color: #1B1B1B;
    }}
    .vx-card + .vx-card {{ margin-top: 14px; }}

    div[data-testid="stButton"] > button {{
        background: linear-gradient(90deg, #43A047, #66BB6A);
        color: #ffffff; font-weight: 700; border: none; border-radius: 12px; padding: 10px 16px;
        box-shadow: 0 6px 16px rgba(76,175,80,0.3); transition: all 0.2s ease-in-out;
    }}
    div[data-testid="stButton"] > button:hover {{
        transform: scale(1.05); background: linear-gradient(90deg, #388E3C, #43A047);
    }}
    div[data-testid="stButton"] > button:active {{ transform: translateY(1px); }}

    .vx-pill {{ display: inline-block; padding: 6px 10px; border-radius: 999px; font-weight: 700; font-size: 12px; letter-spacing: .3px; }}
    .vx-pill.ok {{ background: #E8F5E9; border: 1px solid #43A047; color: #2E7D32; }}
    .vx-pill.warn {{ background: #FFF8E1; border: 1px solid #FFB300; color: #F57C00; }}
    .vx-pill.bad {{ background: #FFEBEE; border: 1px solid #E53935; color: #C62828; }}

    .vx-title {{ font-size: 20px; font-weight: 700; margin-bottom: 10px; color: {header_color}; }}
    .vx-hr {{ height: 1px; background: linear-gradient(90deg, transparent, rgba(76,175,80,0.4), transparent); margin: 14px 0; }}
    </style>
    """, unsafe_allow_html=True)

def render_header():
    st.markdown("""
    <div class="vx-header">
        <h1>🕵️ VietFactAI_THL</h1>
        <div class="vx-sub">Phát hiện & cảnh báo thông tin sai lệch (tiếng Việt)</div>
    </div>
    """, unsafe_allow_html=True)

# ========= GỌI API =========
def call_api_health(api_url: str):
    url = api_url.rstrip("/") + "/health"
    return requests.get(url, timeout=10).json()

def call_api_predict(api_url: str, text: str, k: int, use_explain: bool):
    url = api_url.rstrip("/") + "/predict"
    payload = {"text": text, "k": int(k), "use_explain": bool(use_explain)}
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()

def call_api_debug_retrieve(api_url: str, q: str, k: int):
    url = api_url.rstrip("/") + "/debug/retrieve"
    resp = requests.get(url, params={"q": q, "k": int(k)}, timeout=30)
    resp.raise_for_status()
    return resp.json()

# (tùy chọn) gọi endpoint chỉ sinh bằng chứng bằng LoRA nếu bạn đã thêm /evidence_only_lora trong server
def call_api_evidence_only_lora(api_url: str, text: str, k: int):
    url = api_url.rstrip("/") + "/evidence_only_lora"
    payload = {"text": text, "k": int(k)}
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()

# ========= TRANG HOME =========
def page_home():
    render_header()
    left, right = st.columns([1,1])

    with left:
        st.markdown('<div class="vx-card">', unsafe_allow_html=True)
        st.markdown('<div class="vx-title">🚀 Bắt đầu nhanh</div>', unsafe_allow_html=True)
        st.write("- Mở **Trình kiểm chứng** để dán nội dung cần kiểm tra.")
        st.write("- Hệ thống chỉ **cảnh báo**, không chặn/xóa nội dung.")
        st.write("- Nếu thiếu bằng chứng → trả về **CHƯA ĐỦ BẰNG CHỨNG**.")
        st.markdown('<div class="vx-hr"></div>', unsafe_allow_html=True)
        if st.button("🔍 Mở trình kiểm chứng (Checker)"):
            go_page("checker"); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="vx-card">', unsafe_allow_html=True)
        st.markdown('<div class="vx-title">⚙️ Trạng thái hệ thống</div>', unsafe_allow_html=True)
        api_url = st.session_state.get("api_url", DEFAULT_API_URL)
        try:
            h = call_api_health(api_url)
            st.success(
                f"API OK · device={h.get('device')} · bm25={h.get('bm25_loaded')} "
                f"· sbert={h.get('sbert_loaded')} · docs={h.get('num_docs')} · lora={h.get('lora_loaded')}"
            )
            # Nếu server có trả về đường dẫn các file, hiển thị phụ đề
            if "bm25_pkl" in h or "docs_txt" in h:
                st.caption(f"BM25_PKL: {h.get('bm25_pkl')} | DOCS_TXT: {h.get('docs_txt')}")
            st.markdown('<span class="vx-pill ok">Sẵn sàng thử nghiệm</span>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Không gọi được API tại {api_url}: {e}")
            st.markdown('<span class="vx-pill bad">API lỗi</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ========= TRANG CHECKER =========
def page_checker():
    render_header()
    top_left, _ = st.columns([1,1])
    with top_left:
        if st.button("← Về trang chủ", key="back_home"):
            go_page("home")
            st.session_state["theme"] = "green"
            st.rerun()

    st.markdown('<div class="vx-card">', unsafe_allow_html=True)
    st.markdown('<div class="vx-title">🔎 Phân tích nhanh</div>', unsafe_allow_html=True)

    # Form nhập
    with st.form("check_form"):
        text = st.text_area(
            "Dán nội dung/bài viết ở đây:",
            height=180,
            placeholder="Ví dụ: Uống gừng có thể chữa được COVID-19..."
        )
        submitted = st.form_submit_button("Phân tích ngay")

    if submitted:
        if not (text or "").strip():
            st.warning("⚠️ Hãy nhập nội dung để phân tích.")
        else:
            api_url = st.session_state.get("api_url", DEFAULT_API_URL)
            k = st.session_state.get("k_top", 5)
            use_explain = st.session_state.get("use_explain", False)
            try:
                result = call_api_predict(api_url, text, k=k, use_explain=use_explain)
                label = result.get("label", "Not Enough Information")
                probs = result.get("probs", [])
                evidences = result.get("evidences", [])
                explanation = result.get("explanation", None)

                # Đổi theme theo nhãn
                lablow = str(label).lower()
                if lablow.startswith("refut"):
                    st.session_state["theme"] = "red"
                    st.error("❌ Phát hiện thông tin có khả năng sai lệch!")
                elif lablow.startswith("support"):
                    st.session_state["theme"] = "blue"
                    st.success("✅ Nội dung có vẻ được hỗ trợ bởi bằng chứng.")
                else:
                    st.session_state["theme"] = "green"
                    st.info("ℹ️ Chưa đủ bằng chứng xác thực.")

                # Xác suất
                st.markdown("**Xác suất dự đoán:**")
                lbls = ["Supported", "Refuted", "Not Enough Information"]
                for i, p in enumerate(probs[:3]):
                    st.progress(min(max(float(p), 0.0), 1.0), text=f"{lbls[i]}: {float(p):.3f}")

                # Evidence
                st.markdown("---")
                st.markdown("**Top evidences (SBERT rerank từ BM25):**")
                if evidences:
                    for i, ev in enumerate(evidences, 1):
                        t = ev.get("text", "")
                        st.markdown(f"**{i}.** {t}")
                else:
                    st.caption("Không có evidence (có thể BM25 rỗng hoặc query không match).")

                # LoRA explanation (nếu bật)
                if use_explain:
                    st.markdown("---")
                    st.markdown("**🧠 Giải thích (LoRA PEFT)**")
                    if explanation:
                        st.json(explanation)
                    else:
                        st.caption("Không có giải thích (LLM chưa sẵn sàng hoặc không parse được JSON).")

            except Exception as e:
                st.session_state["theme"] = "red"
                st.error(f"Không gọi được API: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

# ========= SIDEBAR =========
with st.sidebar:
    st.header("⚙️ Cấu hình")
    api_in = st.text_input("API URL", value=st.session_state.get("api_url", DEFAULT_API_URL))
    k_top = st.number_input("Top-K evidences", min_value=1, max_value=10, value=int(st.session_state.get("k_top", 5)))
    use_explain = st.checkbox("Sinh giải thích bằng LoRA (nếu có)", value=bool(st.session_state.get("use_explain", False)))

    colA, colB = st.columns(2)
    with colA:
        if st.button("Lưu cấu hình"):
            st.session_state["api_url"] = api_in
            st.session_state["k_top"] = int(k_top)
            st.session_state["use_explain"] = bool(use_explain)
            st.success("Đã cập nhật cấu hình!")
    with colB:
        if st.button("Kiểm tra /health"):
            try:
                h = call_api_health(api_in)
                st.success("OK"); st.json(h)
            except Exception as e:
                st.error(e)

# ========= ROUTING =========
if "theme" not in st.session_state:
    st.session_state["theme"] = "green"
ui_css(theme=st.session_state["theme"])
page = get_page()
page_checker() if page == "checker" else page_home()
