# 🧠 VietFactAI_THL  
### Phát hiện và cảnh báo thông tin sai lệch tiếng Việt (Open Source AI Fact-checker)

![VietFactAI Banner](https://img.shields.io/badge/PhoBERT-FactCheck-blue)  
![RAG](https://img.shields.io/badge/RAG-Retrieval-green)  
![License](https://img.shields.io/badge/license-MIT-orange)

---

## 🚀 Giới thiệu

**VietFactAI_THL** là hệ thống AI mã nguồn mở giúp **phát hiện, phân loại và cung cấp bằng chứng xác thực cho các tin tức tiếng Việt**.  
Dự án ứng dụng:

- **PhoBERT (VinAI)** – mô hình ngôn ngữ tiếng Việt đã fine-tuned để **phân loại độ xác thực** của phát biểu.  
- **RAG (Retrieval-Augmented Generation)** – kết hợp **BM25** và **SBERT** để **truy xuất bằng chứng chính thống** từ nguồn dữ liệu đã chỉ mục.  
- **Mistral LoRA (tùy chọn)** – tạo phần **giải thích tự nhiên** cho kết quả kiểm chứng.  

---

## ⚙️ Cấu trúc hệ thống

```bash
GPPM_2025/
│
├── checkpoints/              # PhoBERT model (fine-tuned)
│   ├── phobert_best.pth
│   └── tokenizer/
│
├── indices/                  # BM25 + corpus text
│   ├── bm25_index.pkl
│   └── documents.txt
│
├── outputs/                  # Lưu LoRA adapter (nếu có)
│
├── retrieval.py              # SBERT retriever module (BM25 + SBERT)
├── server.py                 # API server (FastAPI)
├── app.py                    # Giao diện người dùng (Streamlit)
│
└── requirements.txt          # Thư viện cần cài đặt
