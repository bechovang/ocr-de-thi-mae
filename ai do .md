Tuyệt vời! Đây là một mô tả dự án rất chi tiết và tham vọng. Để triển khai toàn bộ các tính năng này trong một notebook Colab duy nhất sẽ rất phức tạp và có thể vượt quá giới hạn của một môi trường tương tác. Tuy nhiên, tôi sẽ cung cấp cho bạn một bộ khung mã Python cho Colab, tập trung vào các chức năng cốt lõi bạn đã mô tả, đặc biệt là quy trình OCR đa tầng và tích hợp AI cơ bản.

**Lưu ý quan trọng trước khi bắt đầu:**

1.  **API Keys:** Bạn sẽ cần API key cho Google Gemini (và tùy chọn cho OpenAI/Claude nếu muốn tích hợp).
2.  **Thư viện:** Một số thư viện OCR toán học có thể yêu cầu cài đặt phức tạp hoặc có model lớn. Tôi sẽ cố gắng sử dụng các thư viện dễ cài đặt trên Colab.
3.  **Thời gian xử lý:** Xử lý 50 ảnh với nhiều tầng OCR và AI có thể tốn thời gian đáng kể, đặc biệt nếu không có GPU.
4.  **Độ chính xác:** Độ chính xác thực tế phụ thuộc rất nhiều vào chất lượng ảnh đầu vào và sự phức tạp của công thức.
5.  **Tính năng "2025":** Một số tính năng được mô tả là "2025 updated" hoặc "mới 2025" có thể là giả định về công nghệ tương lai. Tôi sẽ triển khai dựa trên các công cụ hiện có tốt nhất. Texify (VikParuchuri) thường được biết đến với tên `pix2tex` hoặc `LaTeX-OCR`.
6.  **Giới hạn Colab:** Xử lý song song mạnh mẽ và auto-scaling phức tạp có thể bị hạn chế bởi tài nguyên Colab.
7.  **Pix2Text:** Thư viện này khá mạnh, nhưng có thể cần một số bước cài đặt phụ thuộc.
8.  **TrOCR:** Thường dùng cho chữ viết tay hoặc văn bản in, không chuyên cho LaTeX. Tôi sẽ đưa vào như một fallback.

Đây là bộ khung mã, bạn có thể chạy từng khối trên Google Colab:

```python
#@title 1. Cài đặt các thư viện cần thiết
# Thời gian cài đặt có thể mất vài phút

# OCR Engines
!pip install pix2tex -q # Cho Texify (LaTeX-OCR)
!pip install transformers sentencepiece Pillow -q # Cho TrOCR và các tác vụ ảnh
!pip install opencv-python-headless numpy -q # Cho tiền xử lý ảnh
!pip install "unstructured[all-docs]" -q # Có thể dùng cho layout detection hoặc OCR, nhưng ở đây sẽ tập trung vào Pix2Text riêng
!pip install cnocr[onnx-opt] -q # Pix2Text phụ thuộc vào cnocr cho phần text
!pip install pix2text -q # Cho Pix2Text

# AI Enhancement
!pip install google-generativeai -q # Cho Gemini
!pip install python-Levenshtein -q # Để so sánh chuỗi (ví dụ: đánh giá confidence)
!pip install tqdm -q # Để theo dõi tiến trình

# LaTeX
!pip install pylatexenc -q # Để kiểm tra cú pháp LaTeX (cơ bản)

# Hiển thị ảnh trong Colab
from IPython.display import display, Image as IPImage, Markdown
import os
import shutil # Để xoá thư mục tạm
print("✅ Cài đặt hoàn tất!")
```

```python
#@title 2. Import các thư viện và thiết lập API Key
import os
import glob
import shutil
import time
import json
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image as PILImage
from tqdm.notebook import tqdm # Progress bar cho notebook
from google.colab import files
import getpass

# OCR specific imports
from pix2tex.cli import LatexOCR # Texify / LaTeX-OCR
from pix2text import Pix2Text # Pix2Text
from transformers import TrOCRProcessor, VisionEncoderDecoderModel # TrOCR

# AI Enhancement
import google.generativeai as genai

# LaTeX validation
from pylatexenc.latexwalker import LatexWalkerError
from pylatexenc.latex2text import LatexNodes2Text

# --- Cấu hình API Key ---
# Bạn có thể đặt API key trực tiếp vào code nếu chỉ dùng cá nhân,
# nhưng dùng getpass an toàn hơn khi chia sẻ notebook.
try:
    from google.colab import userdata
    GEMINI_API_KEY = userdata.get('GEMINI_API_KEY')
    if not GEMINI_API_KEY:
        print("⚠️ Gemini API Key chưa được thiết lập trong Colab Secrets.")
        print("Vui lòng tạo Secret có tên 'GEMINI_API_KEY' và dán API Key của bạn vào đó.")
        GEMINI_API_KEY = getpass.getpass('Hoặc nhập Google AI Studio API Key của bạn: ')
except ImportError: # Fallback cho môi trường không phải Colab hoặc Colab cũ
    GEMINI_API_KEY = getpass.getpass('Nhập Google AI Studio API Key của bạn: ')

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini API Key đã được cấu hình.")
else:
    print("⛔ Gemini API Key chưa được cung cấp. Chức năng AI Enhancement sẽ bị vô hiệu hóa.")

# --- Khởi tạo các model OCR (có thể tốn thời gian) ---
# Bạn có thể di chuyển việc khởi tạo này vào hàm xử lý để tiết kiệm bộ nhớ nếu cần
# nhưng sẽ làm chậm mỗi lần gọi.

# Tầng 1: Texify (LaTeX-OCR)
try:
    latex_ocr_model = LatexOCR()
    print("✅ LaTeX-OCR (Texify) model loaded.")
except Exception as e:
    latex_ocr_model = None
    print(f"⚠️ Không thể tải LaTeX-OCR model: {e}")

# Tầng 2: Pix2Text
# Pix2Text có thể cần tải model lần đầu chạy
try:
    p2t_model = Pix2Text() # Mặc định dùng model cho tiếng Anh và công thức
    print("✅ Pix2Text model loaded.")
except Exception as e:
    p2t_model = None
    print(f"⚠️ Không thể tải Pix2Text model: {e}")


# Tầng 3: TrOCR (fallback) - Nên dùng model 'printed' cho đề thi
# Sử dụng model nhỏ hơn để tiết kiệm tài nguyên trên Colab nếu cần
# Ví dụ: 'microsoft/trocr-small-printed'
try:
    trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
    trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
    print("✅ TrOCR model loaded.")
except Exception as e:
    trocr_processor = None
    trocr_model = None
    print(f"⚠️ Không thể tải TrOCR model: {e}")

# Thư mục làm việc
BASE_DIR = "math_ocr_processor_2025"
INPUT_DIR = os.path.join(BASE_DIR, "input_images")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_latex")
PREVIEW_DIR = os.path.join(BASE_DIR, "output_previews") # Thư mục cho ảnh render LaTeX (nếu có)
TEMP_PREPROC_DIR = os.path.join(BASE_DIR, "temp_preprocessed") # Thư mục ảnh đã tiền xử lý

# Tạo các thư mục nếu chưa có
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)
os.makedirs(TEMP_PREPROC_DIR, exist_ok=True)

print(f"🗂️ Các thư mục làm việc đã được tạo tại: {BASE_DIR}")
```

```python
#@title 3. Các hàm Tiền Xử Lý Ảnh Thông Minh
def preprocess_image(image_path, output_path, target_size_ocr=None):
    """
    Tiền xử lý ảnh để cải thiện chất lượng cho OCR.
    target_size_ocr: (width, height) tuple for resizing for specific OCR, or None.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Lỗi: Không thể đọc ảnh {image_path}")
            return None

        # 1. Chuyển đổi sang grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Adaptive thresholding để nhị phân hóa ảnh
        # Có thể thử các giá trị blockSize và C khác nhau
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

        # 3. Khử nhiễu (ví dụ: median blur)
        denoised = cv2.medianBlur(binary, 3) # Kích thước kernel có thể điều chỉnh

        # 4. Tăng độ tương phản (không cần thiết lắm sau thresholding, nhưng có thể thử trên ảnh gốc)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # contrasted = clahe.apply(gray) # áp dụng trên ảnh xám

        # 5. Crop thông minh: Loại bỏ viền trắng (đơn giản)
        # Tìm contours, lấy bounding box của contour lớn nhất hoặc tất cả các contours
        # Đây là một phiên bản đơn giản hóa
        contours, _ = cv2.findContours(cv2.bitwise_not(denoised), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Giả sử nội dung chính là một khối lớn
            # Hoặc có thể kết hợp bounding box của nhiều contours
            x_coords = []
            y_coords = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # Lọc các contour quá nhỏ
                if w*h > 100: # Ngưỡng diện tích, cần điều chỉnh
                    x_coords.extend([x, x+w])
                    y_coords.extend([y, y+h])

            if x_coords and y_coords:
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                # Thêm một chút padding
                padding = 10
                min_x = max(0, min_x - padding)
                min_y = max(0, min_y - padding)
                max_x = min(img.shape[1], max_x + padding)
                max_y = min(img.shape[0], max_y + padding)

                cropped_img = denoised[min_y:max_y, min_x:max_x]
                if cropped_img.size == 0: # Nếu crop lỗi, dùng ảnh đã khử nhiễu
                    final_image = denoised
                else:
                    final_image = cropped_img
            else: # Không tìm thấy contour phù hợp
                final_image = denoised
        else:
            final_image = denoised # Nếu không có contour, dùng ảnh đã khử nhiễu

        # 6. Resize (tùy chọn, một số OCR engine có yêu cầu riêng)
        if target_size_ocr:
            final_image = cv2.resize(final_image, target_size_ocr, interpolation=cv2.INTER_LANCZOS4)

        cv2.imwrite(output_path, final_image)
        return output_path
    except Exception as e:
        print(f"Lỗi tiền xử lý ảnh {image_path}: {e}")
        # Nếu lỗi, thử lưu ảnh gốc vào output_path để OCR engine vẫn có thể thử
        try:
            shutil.copy(image_path, output_path)
            return output_path
        except:
            return None


# Hàm phát hiện và xoay ảnh (đơn giản, có thể cần cải thiện)
def deskew_image(image_path, output_path):
    try:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray) # Đảo ngược màu để text thành trắng, nền đen
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1] # Góc trả về trong khoảng [-90, 0)

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # Có thể cần crop lại sau khi xoay để bỏ viền đen
        # (Để đơn giản, tạm thời bỏ qua bước crop phức tạp này)
        cv2.imwrite(output_path, rotated)
        return output_path
    except Exception as e:
        print(f"Lỗi xoay ảnh {image_path}: {e}")
        try:
            shutil.copy(image_path, output_path) # Trả về ảnh gốc nếu lỗi
            return output_path
        except:
            return None

print("✅ Các hàm tiền xử lý ảnh đã sẵn sàng.")
```

```python
#@title 4. Các hàm OCR Engine
def ocr_with_texify(image_path):
    """Sử dụng LaTeX-OCR (Texify)"""
    if not latex_ocr_model:
        return None, 0.0
    try:
        pil_img = PILImage.open(image_path)
        # Không có confidence score trực tiếp từ thư viện này một cách dễ dàng
        # Ta có thể giả định là cao nếu có kết quả
        latex_code = latex_ocr_model(pil_img)
        confidence = 0.95 if latex_code and latex_code.strip() else 0.0
        return latex_code, confidence
    except Exception as e:
        print(f"Lỗi Texify OCR: {e}")
        return None, 0.0

def ocr_with_pix2text(image_path):
    """Sử dụng Pix2Text"""
    if not p2t_model:
        return None, 0.0
    try:
        # Pix2Text trả về list các dict, mỗi dict là 1 phần tử (text, formula)
        # Ta cần ghép các phần tử formula lại
        results = p2t_model.recognize(image_path)
        latex_formulas = [item['text'] for item in results if item['type'] == 'formula']

        if not latex_formulas:
            # Nếu không có formula, thử lấy text thường và coi đó là LaTeX (ít chính xác)
            all_text = " ".join([item['text'] for item in results])
            if all_text:
                return all_text, 0.6 # Confidence thấp hơn
            return None, 0.0

        # Nối các công thức nếu có nhiều (hiếm khi với 1 câu hỏi là 1 ảnh)
        full_latex = " ".join(latex_formulas)
        # Pix2Text không trả confidence trực tiếp, dựa vào kết quả
        confidence = 0.85 if full_latex and full_latex.strip() else 0.0
        return full_latex, confidence
    except Exception as e:
        print(f"Lỗi Pix2Text OCR: {e}")
        return None, 0.0

def ocr_with_trocr(image_path):
    """Sử dụng TrOCR (chủ yếu cho text, fallback)"""
    if not trocr_processor or not trocr_model:
        return None, 0.0
    try:
        pil_img = PILImage.open(image_path).convert("RGB")
        pixel_values = trocr_processor(images=pil_img, return_tensors="pt").pixel_values
        generated_ids = trocr_model.generate(pixel_values)
        generated_text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # TrOCR không chuyên cho LaTeX, confidence thấp hơn
        confidence = 0.50 if generated_text and generated_text.strip() else 0.0
        return generated_text, confidence
    except Exception as e:
        print(f"Lỗi TrOCR: {e}")
        return None, 0.0

OCR_ENGINES = {
    "Texify (LaTeX-OCR)": ocr_with_texify,
    "Pix2Text": ocr_with_pix2text,
    "TrOCR (Fallback)": ocr_with_trocr,
}
OCR_ENGINE_PRIORITY = ["Texify (LaTeX-OCR)", "Pix2Text", "TrOCR (Fallback)"]

print("✅ Các hàm OCR đã sẵn sàng.")
```

```python
#@title 5. Hàm AI Enhancement (Google Gemini)
def enhance_with_gemini(latex_code, question_context=""):
    """
    Sử dụng Gemini để sửa lỗi cú pháp LaTeX và chuẩn hóa.
    question_context: có thể là loại câu hỏi hoặc thông tin thêm.
    """
    if not GEMINI_API_KEY or not genai:
        print("Gemini API không khả dụng. Bỏ qua enhancement.")
        return latex_code, "Gemini API not configured"

    if not latex_code or not latex_code.strip():
        return "", "Input LaTeX is empty"

    model = genai.GenerativeModel('gemini-pro')
    prompt = f"""
    Given the following OCR'd LaTeX string, please correct any syntax errors and ensure it's a valid mathematical expression.
    Return ONLY the corrected LaTeX code. Do not add any explanation or surrounding text.
    If the input is already perfect or very good, return it as is.
    If the input seems to be mostly text and not a math formula, try to format it as plain text within a LaTeX environment if appropriate, or return the original text.

    Context (if any): {question_context}

    Original LaTeX:
    ```latex
    {latex_code}
    ```

    Corrected LaTeX:
    """
    try:
        response = model.generate_content(prompt)
        # Loại bỏ ```latex và ``` nếu có
        enhanced_text = response.text.strip()
        if enhanced_text.startswith("```latex"):
            enhanced_text = enhanced_text[len("```latex"):].strip()
        if enhanced_text.endswith("```"):
            enhanced_text = enhanced_text[:-len("```")].strip()
        return enhanced_text, "Successfully enhanced by Gemini"
    except Exception as e:
        print(f"Lỗi Gemini API: {e}")
        return latex_code, f"Gemini API error: {e}"

print("✅ Hàm AI Enhancement đã sẵn sàng.")
```

```python
#@title 6. Hàm Quality Control & Validation
def validate_latex_syntax(latex_code):
    """
    Kiểm tra cú pháp LaTeX cơ bản.
    Trả về True nếu hợp lệ (hoặc không thể kiểm tra), False nếu có lỗi rõ ràng.
    """
    if not latex_code or not latex_code.strip():
        return False, "Empty LaTeX"
    try:
        # Thử parse LaTeX, nếu không có lỗi nghiêm trọng thì coi là ổn
        # Đây là kiểm tra rất cơ bản
        LatexNodes2Text().latex_to_text(latex_code)
        return True, "Syntax appears valid (basic check)"
    except LatexWalkerError as e:
        # print(f"Lỗi cú pháp LaTeX: {e}") # Gỡ comment nếu muốn debug
        return False, f"Syntax error: {e}"
    except Exception as e: # Các lỗi khác không lường trước
        # print(f"Lỗi không xác định khi kiểm tra LaTeX: {e}")
        return True, f"Unknown validation error (assumed valid): {e}" # Tạm coi là True để không chặn quá trình

print("✅ Hàm Quality Control đã sẵn sàng.")
```

```python
#@title 7. Hàm Xử Lý Chính (Single Image) và Batch Processing
# Cấu hình
ENABLE_DESKEW = False # Đặt True để thử deskew, có thể làm chậm và không luôn hiệu quả
ENABLE_AI_ENHANCEMENT = True # Đặt True để sử dụng Gemini

def process_single_image(image_path, image_idx, total_images, temp_dir):
    """
    Quy trình xử lý cho một ảnh đơn lẻ.
    """
    filename = os.path.basename(image_path)
    unique_id = str(uuid4()) # Để tránh trùng tên file tạm
    print(f"Processing [{image_idx+1}/{total_images}]: {filename}")

    result = {
        "original_filename": filename,
        "output_filename_pattern": f"question_{image_idx+1:02d}",
        "processed_image_path": None,
        "raw_latex": None,
        "ocr_engine_used": None,
        "ocr_confidence": 0.0,
        "enhanced_latex": None,
        "ai_enhancement_status": None,
        "latex_syntax_valid": False,
        "validation_message": "",
        "processing_time": 0.0,
        "error": None
    }
    start_time = time.time()

    try:
        # 1. Tiền xử lý ảnh
        # Tạo tên file tạm duy nhất cho ảnh đã tiền xử lý
        preprocessed_image_name = f"preprocessed_{result['output_filename_pattern']}_{unique_id}.png"
        temp_preprocessed_path = os.path.join(temp_dir, preprocessed_image_name)

        if ENABLE_DESKEW:
            deskewed_path_name = f"deskewed_{result['output_filename_pattern']}_{unique_id}.png"
            deskewed_path = os.path.join(temp_dir, deskewed_path_name)
            deskewed_image_path = deskew_image(image_path, deskewed_path)
            if not deskewed_image_path: # Nếu deskew lỗi, dùng ảnh gốc
                deskewed_image_path = image_path
        else:
            deskewed_image_path = image_path # Bỏ qua deskew

        processed_path = preprocess_image(deskewed_image_path, temp_preprocessed_path)
        if not processed_path:
            raise ValueError("Image preprocessing failed.")
        result["processed_image_path"] = processed_path

        # 2. OCR đa tầng
        raw_latex_output = None
        selected_engine = None
        highest_confidence = 0.0

        for engine_name in OCR_ENGINE_PRIORITY:
            if engine_name in OCR_ENGINES:
                print(f"  Trying OCR Engine: {engine_name}...")
                ocr_func = OCR_ENGINES[engine_name]
                try:
                    latex, confidence = ocr_func(processed_path)
                    if latex and latex.strip(): # Ưu tiên engine nào có kết quả
                        # Nếu có kết quả và confidence cao hơn, hoặc là engine đầu tiên có kết quả
                        if latex and (confidence > highest_confidence or not raw_latex_output):
                            raw_latex_output = latex
                            selected_engine = engine_name
                            highest_confidence = confidence
                            print(f"    ✨ Got result from {engine_name} (Conf: {confidence:.2f})")
                            # Nếu là engine ưu tiên cao (Texify) và có kết quả tốt, có thể dừng sớm
                            if engine_name == "Texify (LaTeX-OCR)" and confidence > 0.9: # Ngưỡng tự tin
                                break
                except Exception as e:
                    print(f"    ⚠️ Error with {engine_name}: {e}")
            else:
                print(f"  Engine {engine_name} not available or not loaded.")

        result["raw_latex"] = raw_latex_output
        result["ocr_engine_used"] = selected_engine
        result["ocr_confidence"] = highest_confidence

        if not raw_latex_output:
            print(f"  ⚠️ No OCR engine could extract LaTeX for {filename}.")
            result["error"] = "All OCR engines failed or returned empty."
        else:
            # 3. AI Enhancement (Tùy chọn)
            if ENABLE_AI_ENHANCEMENT and GEMINI_API_KEY:
                print(f"  Enhancing with Gemini...")
                enhanced_l, enhance_status = enhance_with_gemini(raw_latex_output)
                result["enhanced_latex"] = enhanced_l
                result["ai_enhancement_status"] = enhance_status
            else:
                result["enhanced_latex"] = raw_latex_output # Dùng raw nếu AI không bật/lỗi
                result["ai_enhancement_status"] = "Skipped or API not available"

            # 4. Quality Control & Validation
            final_latex_to_validate = result["enhanced_latex"] if result["enhanced_latex"] else result["raw_latex"]
            is_valid, val_msg = validate_latex_syntax(final_latex_to_validate)
            result["latex_syntax_valid"] = is_valid
            result["validation_message"] = val_msg
            if not is_valid:
                 print(f"  ⚠️ LaTeX syntax validation failed for {filename}: {val_msg}")


    except Exception as e:
        print(f"Critical error processing {filename}: {e}")
        result["error"] = str(e)

    result["processing_time"] = time.time() - start_time
    return result

def batch_process_images(image_paths, output_dir, temp_preproc_dir, max_workers=4):
    """
    Xử lý hàng loạt ảnh sử dụng ThreadPoolExecutor.
    """
    all_results = []
    # Giới hạn max_workers để tránh quá tải Colab, đặc biệt với các model nặng
    # max_workers = min(max_workers, os.cpu_count() or 1)
    # Với Colab, 2-4 workers thường là hợp lý, tùy thuộc vào model có dùng GPU không.

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Tạo future cho mỗi ảnh
        futures = {
            executor.submit(process_single_image, img_path, idx, len(image_paths), temp_preproc_dir): img_path
            for idx, img_path in enumerate(image_paths)
        }

        # Thu thập kết quả khi hoàn thành, có progress bar
        for future in tqdm(as_completed(futures), total=len(image_paths), desc="Batch Processing Images"):
            img_path = futures[future]
            try:
                single_result = future.result()
                all_results.append(single_result)

                # Lưu file .tex cá nhân
                if single_result.get("raw_latex"): # Chỉ lưu nếu có kết quả OCR
                    final_latex = single_result.get("enhanced_latex") or single_result.get("raw_latex")
                    output_tex_filename = f"{single_result['output_filename_pattern']}.tex"
                    output_tex_path = os.path.join(output_dir, output_tex_filename)
                    with open(output_tex_path, "w", encoding="utf-8") as f:
                        f.write(final_latex if final_latex else "% No LaTeX extracted")
                    # print(f"  Saved: {output_tex_path}")

            except Exception as e:
                print(f"Error processing future for {img_path}: {e}")
                all_results.append({
                    "original_filename": os.path.basename(img_path),
                    "error": str(e),
                    "processing_time": 0.0 # Hoặc đo thời gian tới lúc lỗi
                })
    # Sắp xếp lại kết quả theo thứ tự ban đầu (nếu cần, dựa trên output_filename_pattern)
    all_results.sort(key=lambda r: r.get("output_filename_pattern", ""))
    return all_results

print("✅ Hàm xử lý chính và batch đã sẵn sàng.")
```

```python
#@title 8. Nhập Liệu và Chạy Xử Lý

# Dọn dẹp thư mục input và output cũ (nếu cần)
# Thận trọng khi dùng các lệnh này!
# if os.path.exists(INPUT_DIR): shutil.rmtree(INPUT_DIR)
# if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
# if os.path.exists(PREVIEW_DIR): shutil.rmtree(PREVIEW_DIR)
# if os.path.exists(TEMP_PREPROC_DIR): shutil.rmtree(TEMP_PREPROC_DIR)

# os.makedirs(INPUT_DIR, exist_ok=True)
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# os.makedirs(PREVIEW_DIR, exist_ok=True)
# os.makedirs(TEMP_PREPROC_DIR, exist_ok=True) # Tạo lại thư mục tạm

print(f"Vui lòng upload các file ảnh câu hỏi (JPG, PNG, BMP, TIFF, WebP) vào thư mục '{INPUT_DIR}'.")
print(f"Bạn có thể dùng panel 'Files' bên trái của Colab để kéo thả hoặc upload.")
print(f"Sau khi upload xong, chạy cell này một lần nữa (hoặc cell tiếp theo).")

# Cách 1: Upload trực tiếp qua Colab UI (chạy cell này để hiện nút Upload)
# Lưu ý: cách này không tự động quét thư mục, bạn phải upload vào đúng INPUT_DIR
# Hoặc dùng cách upload này rồi copy file vào INPUT_DIR
# uploaded_files = files.upload()
# for filename, content in uploaded_files.items():
#     with open(os.path.join(INPUT_DIR, filename), 'wb') as f:
#         f.write(content)
#     print(f'Đã lưu file {filename} vào {INPUT_DIR}')

# Ví dụ: Thêm một vài ảnh mẫu để test nếu không có ảnh thật
# Bạn có thể comment đoạn này nếu đã upload ảnh thật.
# Tạo ảnh mẫu (nếu không có)
def create_dummy_image(filepath, text="Sample Math: $x^2 + y^2 = z^2$"):
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new('RGB', (400, 100), color = (255, 255, 255))
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
        d.text((10,30), text, fill=(0,0,0), font=font)
        img.save(filepath)
        return True
    except ImportError:
        print("Pillow.ImageFont không tìm thấy, không tạo được ảnh mẫu.")
        return False
    except Exception as e:
        print(f"Lỗi tạo ảnh mẫu: {e}")
        return False

# Kiểm tra nếu INPUT_DIR rỗng thì tạo ảnh mẫu
if not os.listdir(INPUT_DIR):
    print("Thư mục INPUT_DIR rỗng. Tạo ảnh mẫu...")
    create_dummy_image(os.path.join(INPUT_DIR, "dummy_question_01.png"), "$\\sum_{i=1}^n i = \\frac{n(n+1)}{2}$")
    create_dummy_image(os.path.join(INPUT_DIR, "dummy_question_02.jpg"), "$E=mc^2$ (Einstein)")
    create_dummy_image(os.path.join(INPUT_DIR, "dummy_question_03.png"), "Text with math $\\alpha + \\beta$")
else:
    print(f"Tìm thấy {len(os.listdir(INPUT_DIR))} file(s) trong {INPUT_DIR}.")


# --- Bắt đầu xử lý ---
# Lấy danh sách file ảnh từ INPUT_DIR
image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp")
input_image_paths = []
for ext in image_extensions:
    input_image_paths.extend(glob.glob(os.path.join(INPUT_DIR, ext)))

if not input_image_paths:
    print("⛔ Không tìm thấy file ảnh nào trong thư mục input. Vui lòng upload ảnh và chạy lại.")
else:
    print(f"Tìm thấy {len(input_image_paths)} ảnh để xử lý.")
    # Hiển thị một vài ảnh đầu vào (tùy chọn)
    # for i, img_path in enumerate(input_image_paths[:3]):
    #     print(f"Input image {i+1}: {os.path.basename(img_path)}")
    #     display(IPImage(filename=img_path, width=300))

    # Thiết lập số worker cho xử lý song song
    # Trên Colab free tier, 2-4 là hợp lý. Nếu có GPU và model dùng GPU, có thể khác.
    num_workers = 2 # Điều chỉnh nếu cần

    print(f"\n🚀 Bắt đầu xử lý hàng loạt với {num_workers} worker(s)...")
    processing_results = batch_process_images(input_image_paths, OUTPUT_DIR, TEMP_PREPROC_DIR, max_workers=num_workers)
    print("\n✅ Xử lý hàng loạt hoàn tất!")

    # Dọn dẹp thư mục tạm chứa ảnh đã tiền xử lý
    if os.path.exists(TEMP_PREPROC_DIR):
        try:
            shutil.rmtree(TEMP_PREPROC_DIR)
            print(f"🗑️ Đã xóa thư mục tạm: {TEMP_PREPROC_DIR}")
            os.makedirs(TEMP_PREPROC_DIR, exist_ok=True) # Tạo lại cho lần chạy sau
        except Exception as e:
            print(f"Lỗi khi xóa thư mục tạm: {e}")
```

```python
#@title 9. Export & Output Options + Báo Cáo
def generate_report(results, output_dir):
    if not results:
        print("Không có kết quả để tạo báo cáo.")
        return

    # --- 1. Compiled LaTeX Document ---
    compiled_tex_path = os.path.join(output_dir, "compiled_document.tex")
    # Tạo một file .tex hoàn chỉnh chứa tất cả các câu hỏi
    # Bạn cần một trình biên dịch LaTeX (như MiKTeX, TeX Live) để biên dịch file này thành PDF.
    # Colab không có sẵn trình biên dịch LaTeX đầy đủ.
    # Chúng ta chỉ tạo file .tex nguồn.
    with open(compiled_tex_path, "w", encoding="utf-8") as f:
        f.write("\\documentclass{article}\n")
        f.write("\\usepackage{amsmath, amssymb,amsfonts}\n")
        f.write("\\usepackage[utf8]{inputenc}\n")
        f.write("\\usepackage{geometry}\n")
        f.write("\\geometry{a4paper, margin=1in}\n")
        f.write("\\title{Tổng hợp Câu hỏi Toán OCR}\n")
        f.write("\\author{Free Math OCR Batch Processor 2025}\n")
        f.write("\\date{\\today}\n")
        f.write("\\begin{document}\n")
        f.write("\\maketitle\n")
        f.write("\\begin{enumerate}\n\n")

        for idx, res in enumerate(results):
            latex_content = res.get("enhanced_latex") or res.get("raw_latex") or "% Lỗi: Không có LaTeX"
            f.write(f"\\item % Câu hỏi {idx+1} từ file: {res.get('original_filename', 'N/A')}\n")
            f.write(latex_content + "\n\n")

        f.write("\\end{enumerate}\n")
        f.write("\\end{document}\n")
    print(f"📄 Document LaTeX tổng hợp đã lưu tại: {compiled_tex_path}")

    # --- 2. Excel/CSV Export ---
    csv_path = os.path.join(output_dir, "processing_summary.csv")
    fieldnames = [
        "Original Filename", "Output Pattern", "OCR Engine", "OCR Confidence",
        "Raw LaTeX", "Enhanced LaTeX", "AI Status",
        "Syntax Valid", "Validation Message", "Processing Time (s)", "Error"
    ]
    with open(csv_path, "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow({
                "Original Filename": res.get("original_filename"),
                "Output Pattern": res.get("output_filename_pattern"),
                "OCR Engine": res.get("ocr_engine_used"),
                "OCR Confidence": f"{res.get('ocr_confidence', 0.0):.2f}",
                "Raw LaTeX": res.get("raw_latex"),
                "Enhanced LaTeX": res.get("enhanced_latex"),
                "AI Status": res.get("ai_enhancement_status"),
                "Syntax Valid": res.get("latex_syntax_valid"),
                "Validation Message": res.get("validation_message"),
                "Processing Time (s)": f"{res.get('processing_time', 0.0):.2f}",
                "Error": res.get("error")
            })
    print(f"📊 Báo cáo CSV đã lưu tại: {csv_path}")

    # --- 3. JSON Export ---
    json_path = os.path.join(output_dir, "processing_summary.json")
    with open(json_path, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"🔗 Dữ liệu JSON đã lưu tại: {json_path}")

    # --- 4. Summary Report (Console) ---
    total_questions = len(results)
    successful_ocr = sum(1 for r in results if r.get("raw_latex"))
    successful_enhancement = sum(1 for r in results if r.get("enhanced_latex") and "Success" in str(r.get("ai_enhancement_status","")))
    syntax_ok = sum(1 for r in results if r.get("latex_syntax_valid"))
    total_processing_time = sum(r.get("processing_time", 0.0) for r in results)
    avg_confidence = 0
    if successful_ocr > 0:
         avg_confidence = sum(r.get('ocr_confidence',0.0) for r in results if r.get("raw_latex")) / successful_ocr

    manual_review_count = total_questions - syntax_ok # Đơn giản hóa: cần review nếu syntax không ok
                                                    # Hoặc confidence thấp (ví dụ < 0.7)
    low_confidence_threshold = 0.7
    manual_review_confidence = sum(1 for r in results if r.get("raw_latex") and (r.get('ocr_confidence', 0.0) < low_confidence_threshold or not r.get("latex_syntax_valid")))


    report_md = f"""
    📊 **PROCESSING SUMMARY**
    ├─ Total Questions Processed: {total_questions}
    ├─ Successfully OCR'd: {successful_ocr} ({successful_ocr/total_questions:.2%} if total_questions > 0 else 0)
    ├─ AI Enhancement Applied: {successful_enhancement} (trên {sum(1 for r in results if r.get("raw_latex") and ENABLE_AI_ENHANCEMENT and GEMINI_API_KEY)} câu có OCR và AI bật)
    ├─ Average OCR Confidence (cho câu thành công): {avg_confidence:.3f}
    ├─ Total Processing Time: {total_processing_time:.2f}s
    └─ Manual Review Recommended (syntax error or low confidence < {low_confidence_threshold}): {manual_review_confidence} questions

    📈 **QUALITY METRICS**
    ├─ Valid LaTeX Syntax (basic check): {syntax_ok} questions
    └─ Potentially Needing Corrections: {total_questions - syntax_ok} questions
    """
    print("\n--- BÁO CÁO TÓM TẮT ---")
    # print(report_md) # Dạng Markdown đẹp hơn
    display(Markdown(report_md))

    # Hiển thị một vài kết quả đầu ra
    print("\n--- VÍ DỤ KẾT QUẢ ---")
    for i, res in enumerate(results[:min(5, len(results))]): # Hiển thị 5 kết quả đầu
        print(f"\nCâu hỏi: {res.get('original_filename', 'N/A')} ({res.get('output_filename_pattern')})")
        print(f"  OCR Engine: {res.get('ocr_engine_used', 'N/A')}, Confidence: {res.get('ocr_confidence', 0.0):.2f}")
        print(f"  Raw LaTeX: {res.get('raw_latex', 'N/A')[:100]}...") # Cắt bớt cho gọn
        if ENABLE_AI_ENHANCEMENT:
            print(f"  Enhanced LaTeX: {res.get('enhanced_latex', 'N/A')[:100]}...")
            print(f"  AI Status: {res.get('ai_enhancement_status', 'N/A')}")
        print(f"  Syntax Valid: {res.get('latex_syntax_valid', False)} ({res.get('validation_message', '')})")
        if res.get('error'):
            print(f"  Lỗi: {res.get('error')}")

    # Hướng dẫn tải file
    print(f"\n\n📂 Các file kết quả đã được lưu trong thư mục '{OUTPUT_DIR}'.")
    print("Bạn có thể tải về dưới dạng ZIP bằng cách click chuột phải vào thư mục trong panel 'Files' bên trái và chọn 'Download'.")
    # Hoặc tạo file zip để tải
    # shutil.make_archive(BASE_DIR, 'zip', BASE_DIR)
    # print(f"\n🗜️ Đã tạo file {BASE_DIR}.zip. Bạn có thể tải về từ panel Files.")


if 'processing_results' in locals() and processing_results:
    generate_report(processing_results, OUTPUT_DIR)
else:
    print("Chưa có kết quả xử lý. Vui lòng chạy bước 8 (Nhập liệu và Xử Lý).")

```

**Cách sử dụng Notebook này:**

1.  **Mở trên Google Colab:** File > Open notebook > GitHub > Dán URL của Gist (nếu bạn lưu nó thành Gist) hoặc upload file .ipynb.
2.  **Thiết lập API Key (Cell 2):**
    *   Cách tốt nhất: Vào menu "Secrets" (biểu tượng chìa khóa bên trái Colab), tạo một secret mới tên là `GEMINI_API_KEY` và dán API key của Google AI Studio vào đó.
    *   Hoặc: Chạy cell 2 và nhập API key khi được hỏi.
3.  **Chạy từng cell theo thứ tự:**
    *   **Cell 1:** Cài đặt thư viện (chỉ cần chạy lần đầu hoặc khi session Colab khởi động lại).
    *   **Cell 2:** Import và cấu hình API.
    *   **Cell 3-7:** Định nghĩa các hàm tiện ích.
    *   **Cell 8:**
        *   Upload ảnh của bạn vào thư mục `math_ocr_processor_2025/input_images` bằng cách sử dụng trình quản lý file bên trái của Colab (kéo thả hoặc click chuột phải vào thư mục > Upload).
        *   Sau khi upload xong, chạy cell 8 để bắt đầu quá trình xử lý. Nếu thư mục `input_images` rỗng, nó sẽ tạo vài ảnh mẫu.
    *   **Cell 9:** Xem báo cáo, xuất file và hiển thị ví dụ kết quả.

**Các cải tiến và lưu ý thêm:**

*   **Render LaTeX Preview:** Để tạo ảnh preview PNG/SVG từ LaTeX trên Colab, bạn cần cài đặt một bản phân phối LaTeX đầy đủ (ví dụ: TeX Live) và các công cụ như `pdflatex`, `dvisvgm`. Điều này có thể khá nặng và mất thời gian cài đặt. Bạn có thể thêm một bước dùng `matplotlib.mathtext` để render công thức đơn giản, nhưng nó không phải là một trình biên dịch LaTeX đầy đủ.
    ```python
    # Ví dụ render LaTeX dùng Matplotlib (cho công thức đơn giản)
    # import matplotlib.pyplot as plt
    # def render_latex_to_image(latex_str, output_path):
    #     try:
    #         fig, ax = plt.subplots(figsize=(max(1, len(latex_str)/10), 2)) # Điều chỉnh kích thước
    #         ax.text(0, 0.5, f"${latex_str}$", size=15, va='center')
    #         ax.axis('off')
    #         plt.savefig(output_path, bbox_inches='tight', dpi=200)
    #         plt.close(fig)
    #         return output_path
    #     except Exception as e:
    #         print(f"Lỗi render LaTeX: {e}")
    #         return None
    ```
*   **Smart Question Detection & LaTeX Templating:** Đây là một tác vụ NLP phức tạp. Bạn có thể sử dụng các mô hình phân loại văn bản hoặc LLM để thử phát hiện loại câu hỏi, sau đó áp dụng template.
*   **Advanced Error Recovery:** Có thể bao gồm việc thử lại với các tham số tiền xử lý khác nhau hoặc crop lại vùng ảnh nếu confidence thấp.
*   **Collaborative Features (Overleaf):** Việc xuất ra file `.tex` tổng hợp đã là bước đầu. Tích hợp API Overleaf (nếu có) sẽ phức tạp hơn.
*   **Tối ưu hóa hiệu suất:**
    *   **Model Quantization:** Một số model (như của Hugging Face Transformers) có thể được lượng tử hóa để giảm kích thước và tăng tốc độ trên CPU, có thể đánh đổi một chút độ chính xác.
    *   **Intelligent Caching:** Lưu kết quả của các bước tốn thời gian (ví dụ, kết quả OCR cho một ảnh không đổi) để không phải xử lý lại.
*   **Độ chính xác của "Confidence Score":** Các thư viện OCR không phải lúc nào cũng cung cấp confidence score đáng tin cậy hoặc dễ truy cập. Các giá trị confidence tôi dùng ở trên phần lớn là ước lượng.
*   **Hạn chế tài nguyên Colab:** Các session Colab có giới hạn về thời gian chạy, RAM và GPU (nếu dùng). Các tác vụ nặng có thể bị ngắt.

Bộ khung mã này cung cấp một nền tảng vững chắc. Bạn có thể mở rộng và tinh chỉnh từng phần dựa trên yêu cầu cụ thể và các công cụ/API mới xuất hiện. Chúc bạn thành công với dự án!
