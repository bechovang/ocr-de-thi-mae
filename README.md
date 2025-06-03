
## 🧩 `mathpix-batch-ocr`

**Chuyển đổi hàng loạt ảnh chụp đề thi hoặc tài liệu toán học thành mã LaTeX bằng Mathpix API.**

---

### 🧠 Tính năng nổi bật

* 🖼️ **OCR công thức toán học**: Trích xuất LaTeX từ ảnh có chứa biểu thức toán học (in hoặc viết tay).
* 🗂️ **Xử lý hàng loạt**: Tự động quét thư mục ảnh và xử lý toàn bộ.
* 📝 **Lưu kết quả dễ dàng**: Lưu LaTeX của từng ảnh vào thư mục `output/`, theo tên file tương ứng.
* 🔑 **Sử dụng Mathpix API chính xác cao**: Nhận diện được cả văn bản và công thức.

---

### 🧾 Yêu cầu hệ thống (Requirements)

| Thành phần        | Phiên bản đề xuất           |
| ----------------- | --------------------------- |
| Python            | 3.7 trở lên                 |
| Tài khoản Mathpix | Có App ID và App Key        |
| Gói Python        | `requests`, `python-dotenv` |

Cài đặt bằng lệnh:

```bash
pip install -r requirements.txt
```

---

### 📋 Kế hoạch triển khai (Planning & Cách làm)

#### Bước 1️⃣: Tạo tài khoản và lấy API

1. Truy cập: [https://mathpix.com](https://mathpix.com)
2. Đăng ký tài khoản (miễn phí hoặc nâng cấp nếu cần)
3. Vào Dashboard → Copy `APP ID` và `APP KEY`

#### Bước 2️⃣: Tạo `.env` chứa thông tin API

Tạo file `.env` trong thư mục gốc:

```env
MATHPIX_APP_ID=your_app_id
MATHPIX_APP_KEY=your_app_key
```

#### Bước 3️⃣: Chuẩn bị ảnh đề thi

Tạo thư mục `images/` và đặt các ảnh `.jpg`, `.png`, `.jpeg` vào đó.

#### Bước 4️⃣: Chạy script

Chạy script bằng:

```bash
python main.py
```

Script sẽ:

* Duyệt tất cả ảnh trong `images/`
* Gửi ảnh tới Mathpix API
* Nhận lại kết quả LaTeX
* Ghi ra file `output/filename.txt`

---

### 📂 Cấu trúc thư mục

```
mathpix-batch-ocr/
│
├── images/               # Ảnh đề thi cần OCR
├── output/               # Kết quả đầu ra LaTeX
├── main.py               # Script chính
├── .env                  # Chứa API key
├── requirements.txt      # Thư viện cần thiết
└── README.md             # Tài liệu hướng dẫn
```

---

### ▶️ Ví dụ kết quả

Giả sử bạn có ảnh `images/de1.jpg`, kết quả sau khi chạy sẽ là:

```
output/de1.txt:
\[
\int_0^\infty \frac{x^2}{e^x - 1} dx = 2\zeta(3)
\]
```

---

### ❓ Câu hỏi thường gặp

#### ❓ Dùng được cho ảnh viết tay không?

✅ Có! Mathpix hỗ trợ công thức viết tay nếu rõ nét.

#### ❓ Có thể xuất sang Word không?

Không trực tiếp, nhưng bạn có thể:

* Dùng LaTeX → PDF → Copy vào Word
* Hoặc dùng bản Mathpix Snip Desktop có hỗ trợ export sang Word

