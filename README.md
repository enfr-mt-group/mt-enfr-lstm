DỊCH MÁY ANH → PHÁP

## **Mô hình Encoder–Decoder LSTM (Seq2Seq)**

### **Đồ án Xử lý Ngôn ngữ Tự nhiên – HK1 / 2025–2026**

---

**1. Giới thiệu**

Đồ án triển khai hệ thống **Dịch máy Anh → Pháp (English → French Machine Translation)** sử dụng kiến trúc **Seq2Seq Encoder–Decoder với LSTM**, theo mô hình kinh điển trong *Sutskever et al., 2014*.

Mô hình sử dụng **context vector cố định (không dùng attention)** theo đúng yêu cầu đề bài.

Toàn bộ pipeline được **tự xây dựng từ đầu bằng PyTorch**, bao gồm:

* Tokenization bằng spaCy
* Xây dựng Vocabulary (giới hạn 10.000 từ phổ biến nhất)
* Numericalization & Padding
* Encoder–Decoder LSTM
* Teacher Forcing
* Greedy Decoding
* Tính BLEU Score
* Phân tích lỗi và đề xuất cải tiến

---

**2. Cấu trúc dự án**

```
NMT-English-French/
│
├── data/                # Dữ liệu Multi30K EN–FR
├── src/
│   ├── dataset.py       # Xử lý dữ liệu, vocab, dataloader
│   ├── model.py         # Encoder, Decoder, Seq2Seq LSTM
│   ├── train.py         # Huấn luyện + lưu checkpoint
│   ├── evaluate.py      # Dịch câu + BLEU score
│   └── utils.py         # Hàm tiện ích
│
├── checkpoints/         # Trọng số mô hình (.pth)
├── notebooks/           # Notebook phân tích
└── README.md
```

---

**3. Dữ liệu**

Sử dụng bộ **Multi30K** (English–French):

| Tập        | Số lượng câu |
| ---------- | ------------ |
| Train      | 29,000       |
| Validation | 1,014        |
| Test       | 1,000        |

### **Xây dựng Vocabulary**

* Lấy **10.000 từ phổ biến nhất** cho mỗi ngôn ngữ
* Thêm token đặc biệt: `<pad>, <sos>, <eos>, <unk>`

Kích thước vocab cuối:

* **Tiếng Anh:** 9.797 từ
* **Tiếng Pháp:** 10.004 từ

---

**4. Kiến trúc mô hình**

### **Encoder**

* Embedding
* LSTM 2 lớp
* Xuất ra hidden + cell cuối → **context vector**

### **Decoder**

* Embedding
* LSTM 2 lớp
* Khởi tạo bằng context vector
* Linear → Softmax

### **Cơ chế huấn luyện**

* **Teacher Forcing = 0.5**
* Chiến lược Greedy Decoding

---

 **5. Huấn luyện mô hình**

### **Chạy huấn luyện**

```
python src/train.py
```

### Hyperparameters chính

* Embedding size: 256
* Hidden size: 512
* LSTM layers: 2
* Dropout: 0.5
* Batch size: 32
* Optimizer: Adam
* Loss: CrossEntropy (ignore `<pad>`)

Checkpoint lưu tại:

```
checkpoints/best_model.pth
```

---

 **6. Đánh giá**

### Chạy:

```
python src/evaluate.py
```

Tính:

* BLEU score (NLTK corpus BLEU)
* 5–10 ví dụ dịch mẫu để phân tích chất lượng

### **BLEU kỳ vọng (không attention)**

| Mô hình                               | BLEU      |
| ------------------------------------- | --------- |
| Seq2Seq LSTM (context vector cố định) | **15–20** |

---

**7. Ví dụ dịch**

```
EN:  A man is running on the beach .
FR:  un homme court sur la plage .
PRED: un homme court sur la plage .
```

```
EN:  A dog is catching a frisbee in the park .
FR:  un chien attrape un frisbee dans le parc .
PRED: un chien joue dans le parc .
```

---
 **8. Phân tích lỗi**

### **1. Từ hiếm (OOV) → `<unk>`**

≈ 40% từ chỉ xuất hiện 1 lần → mất thông tin.

### **2. Mất thông tin ở câu dài**

Context vector cố định ⇒ bottleneck.

### **3. Thiếu từ hoặc sai nghĩa**

Không có attention nên khó căn chỉnh từng vị trí.

### **4. Lỗi ngữ pháp tiếng Pháp**

Ví dụ: “un/une”, thì động từ, giới từ.

---

**9. Đề xuất cải tiến (không bắt buộc nhưng nên trình bày trong báo cáo)**

* Thêm **Attention (Bahdanau / Luong)**
* **Beam Search** thay vì Greedy
* BiLSTM Encoder
* Pretrained embeddings (FastText EN/FR)
* LayerNorm hoặc Dropout nâng cao
* Subword (BPE) để giảm OOV

Các cải tiến này thường tăng BLEU lên **24–30+**.

---


**Thành viên thực hiện**

| Họ và tên                  | MSSV       | Vai trò                                                 |
| -------------------------- | ---------- | ------------------------------------------------------- |
| **Đỗ Minh Quân**           | 3122411166 | Xử lý dữ liệu, xây dựng vocab, cài đặt inference & BLEU |
| **Lê Thị Mỹ Hương**        | 3122411077 | Xây dựng mô hình, huấn luyện, phân tích lỗi             |

GVHD: TS. Nguyễn Tuấn Đăng

**Khoa Công nghệ Thông tin – ĐH Sài Gòn**

