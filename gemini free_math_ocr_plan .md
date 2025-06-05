Okay, I will complete the `free_error_recovery.py` file and then flesh out the rest of Phase 3 (Optimization + UI) for your plan.

First, let's complete the `free_error_recovery.py` file:

```python
# free_error_recovery.py
import hashlib
import pickle
import os
from typing import Optional, Dict, Any
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FreeErrorRecovery:
    def __init__(self, cache_dir: str = "./ocr_cache"): # Changed to ocr_cache for clarity
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.max_retries = 3
        self.retry_delay_seconds = 5 # Wait 5 seconds between retries

    def get_image_hash(self, image_path: str) -> str:
        """Generate MD5 hash for image file content to use as cache key."""
        hasher = hashlib.md5()
        try:
            with open(image_path, 'rb') as f:
                buf = f.read()
                hasher.update(buf)
            return hasher.hexdigest()
        except FileNotFoundError:
            logging.error(f"Cache: Image file not found at {image_path}")
            return None
        except Exception as e:
            logging.error(f"Cache: Error hashing image {image_path}: {e}")
            return None

    def load_from_cache(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Load processed result from cache using image hash."""
        image_hash = self.get_image_hash(image_path)
        if not image_hash:
            return None

        cache_file = os.path.join(self.cache_dir, f"{image_hash}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                logging.info(f"Cache: Loaded result for {os.path.basename(image_path)} from cache.")
                return cached_data
            except Exception as e:
                logging.warning(f"Cache: Error loading cache file {cache_file}: {e}. Will reprocess.")
                return None
        return None

    def save_to_cache(self, image_path: str, data: Dict[str, Any]):
        """Save processed result to cache using image hash."""
        image_hash = self.get_image_hash(image_path)
        if not image_hash:
            logging.warning(f"Cache: Could not save to cache, failed to hash {os.path.basename(image_path)}")
            return

        cache_file = os.path.join(self.cache_dir, f"{image_hash}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            logging.info(f"Cache: Saved result for {os.path.basename(image_path)} to cache.")
        except Exception as e:
            logging.error(f"Cache: Error saving cache file {cache_file}: {e}")

    async def retry_operation(self, operation, *args, **kwargs):
        """
        Retry an asynchronous operation with a specified delay.
        `operation` should be an async function (coroutine).
        """
        for attempt in range(self.max_retries):
            try:
                result = await operation(*args, **kwargs)
                return result
            except Exception as e:
                logging.warning(f"Operation failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt + 1 == self.max_retries:
                    logging.error(f"Operation failed after {self.max_retries} retries.")
                    raise # Re-raise the last exception
                await asyncio.sleep(self.retry_delay_seconds * (attempt + 1)) # Exponential backoff might be better
        return None # Should not be reached if raise is used

    def handle_component_failure(self, component_name: str, error: Exception, image_path: Optional[str] = None):
        """
        Generic handler for component failures.
        Logs the error and can be extended for more specific actions.
        """
        error_message = f"Component '{component_name}' failed"
        if image_path:
            error_message += f" while processing '{os.path.basename(image_path)}'"
        error_message += f". Error: {error}"
        logging.error(error_message)
        # Potentially, we could mark this image as problematic,
        # or try a fallback mechanism if this was a critical component.
        # For now, just logging.

# Example usage (conceptual, would be integrated into FreeBatchProcessor)
# async def some_critical_async_task(param):
#     # Simulates a task that might fail
#     if random.random() < 0.5:
#         raise ValueError("Simulated task failure")
#     return f"Task completed with {param}"

# async def main_example():
#     recovery_system = FreeErrorRecovery()
#     try:
#         # Example of retrying an operation
#         result = await recovery_system.retry_operation(some_critical_async_task, "test_param")
#         if result:
#             print(f"Operation successful: {result}")
#     except Exception as e:
#         print(f"Operation ultimately failed: {e}")
#         recovery_system.handle_component_failure("ExampleTask", e, "some_image.png")

#     # Example of caching
#     dummy_image_path = "dummy_image.png" # Create a dummy file for testing
#     if not os.path.exists(dummy_image_path):
#         with open(dummy_image_path, "w") as f: f.write("dummy content")

#     cached_result = recovery_system.load_from_cache(dummy_image_path)
#     if cached_result:
#         print(f"Loaded from cache: {cached_result}")
#     else:
#         print("Not in cache, processing...")
#         processed_data = {"latex": "\\frac{1}{2}", "confidence": 90} # Simulate processing
#         recovery_system.save_to_cache(dummy_image_path, processed_data)
#         print("Processed and saved to cache.")
#     if os.path.exists(dummy_image_path): os.remove(dummy_image_path)

# if __name__ == "__main__":
#     import asyncio
#     import random
#     # asyncio.run(main_example()) # Uncomment to run example
```

Now, I will complete the plan document by adding Phase 3 details.

--- START OF FILE free_math_ocr_plan.md ---

# 🆓 MATH OCR - GIẢI PHÁP HOÀN TOÀN MIỄN PHÍ

## 🎯 TỔNG QUAN GIẢI PHÁP FREE
**Mục tiêu**: Xây dựng hệ thống OCR toán học hoàn toàn miễn phí với độ chính xác 85%+, xử lý 50 câu hỏi/batch

**💡 Triết lý**: Thay thế hoàn toàn các API trả phí bằng các mô hình mã nguồn mở và công cụ offline

---

## 🔄 THAY ĐỔI CHÍNH SO VỚI PHIÊN BẢN TRƯA PHÍ

| Component | Phiên bản trả phí | ➜ | Phiên bản FREE |
|-----------|-------------------|---|----------------|
| **AI Enhancement** | Gemini + Claude API | ➜ | **Ollama (Llama 3.1, Qwen2.5-Math)** |
| **Vision AI** | Gemini Vision | ➜ | **LLaVA, MiniCPM-V2.6** |
| **Cloud Processing** | Cloud APIs | ➜ | **100% Local Processing** |
| **Storage** | Cloud Storage | ➜ | **Local File System** |
| **Dependencies** | Paid services | ➜ | **Open Source Only** |

---

## 🛠️ TECH STACK HOÀN TOÀN MIỄN PHÍ

### **Core OCR Engines (Miễn phí)**
```bash
# Primary OCR Engines - Tất cả đều miễn phí
- pix2tex: LaTeX OCR (Open Source)
- TrOCR: Microsoft's Transformer OCR (Free)
- PaddleOCR: Baidu's OCR (Open Source)  
- EasyOCR: Open source OCR
- Tesseract: Google's OCR (Free)

# Backup/Specialized
- LaTeX-OCR: Specialized for math (Open Source)
- TexTeller: Academic math OCR (Free)
```

### **Local AI Models (Chạy offline)**
```bash
# Language Models via Ollama (100% Free)
- llama3.1:8b (General purpose)
- qwen2.5-math:7b (Specialized for math)
- codellama:7b (Code generation)

# Vision Models (Open Source)
- llava:7b (Vision-language model)
- minicpm-v:8b (Efficient vision model) # Note: Ollama might not have 8B, check available sizes e.g., minicpm:v2.5
- moondream2 (Lightweight vision)

# Specialized Models
# mathpix-alternative (Open source math OCR) -> This is more a concept/project, pix2tex is a concrete example
- nomeroff-net (Text detection, number plates, may not be directly useful for math)
```

### **Supporting Libraries (Tất cả miễn phí)**
```python
# Image Processing
opencv-python          # Computer vision
pillow                # Image handling  
scikit-image          # Advanced image processing
imageio               # Image I/O

# Machine Learning
torch                 # PyTorch
transformers          # Hugging Face models
sentence-transformers # Text embeddings
datasets              # Dataset handling

# OCR & Text Processing
pytesseract          # Tesseract wrapper
easyocr              # Easy OCR
paddlepaddle         # Paddle OCR # Requires paddlepaddle-gpu or paddlepaddle for CPU
nltk                 # Natural language processing

# Utilities
numpy, scipy         # Numerical computing
pandas               # Data processing
matplotlib           # Plotting
tqdm                 # Progress bars
asyncio              # Async processing
psutil               # System monitoring
```

---

## 📅 TIMELINE MIỄN PHÍ: 6 TUẦN

| Phase | Thời gian | Mục tiêu chính |
|-------|-----------|----------------|
| **Phase 1** | Tuần 1-2 | Setup Local AI + Core OCR |
| **Phase 2** | Tuần 3-4 | Local AI Integration & Advanced Features |
| **Phase 3** | Tuần 5-6 | Optimization, Evaluation + UI |

---

## 🚀 PHASE 1: LOCAL AI SETUP (Tuần 1-2)

### **TUẦN 1: Environment & Local AI Setup**

#### **Ngày 1-2: Local AI Environment**
```bash
# 1. Install Ollama (Local AI Runtime)
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Pull free math-specialized models
ollama pull llama3.1:8b
ollama pull qwen2.5-math:7b # Check exact model name on Ollama Hub, might be qwen2:7b-instruct-q8_0 etc.
ollama pull llava:7b
ollama pull codellama:7b
ollama pull moondream # Lightweight vision for quick checks
# ollama pull minicpm # Check available MiniCPM tags

# 3. Install Python dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # Or CPU version
pip install transformers sentence-transformers datasets
pip install opencv-python pillow scikit-image imageio
pip install easyocr paddlepaddle # or paddlepaddle-gpu
pip install pytesseract
pip install ollama psutil numpy scipy pandas matplotlib tqdm asyncio
```

**Local AI Server Setup:**
```python
# local_ai_server.py
import ollama
from typing import Dict, Any, List
import asyncio
import json # For parsing JSON response

class LocalAIServer:
    def __init__(self):
        self.models = {
            'math': 'qwen2.5-math:7b',      # Specialized for math (confirm exact tag on Ollama)
            'general': 'llama3.1:8b',       # General purpose  
            'vision': 'llava:7b',           # Vision + text
            'code': 'codellama:7b'          # LaTeX generation
        }
    
    async def enhance_latex(self, raw_latex: str, image_path: str = None, context_prompt: str = None):
        base_prompt = f"""
        Correct and enhance the following LaTeX OCR output.
        Return *only* the corrected LaTeX code, with no other text or explanations.
        Ensure mathematical correctness and standard notation. Add missing brackets or operators where appropriate.
        
        Raw LaTeX:
        {raw_latex}
        """
        
        if context_prompt: # For more complex scenarios like consensus
            prompt = context_prompt
        else:
            prompt = base_prompt
        
        model_to_use = self.models['math']
        images_param = []

        if image_path:
            model_to_use = self.models['vision'] # LLaVA can use image context
            images_param = [image_path]
        
        response = ollama.generate(
            model=model_to_use,
            prompt=prompt,
            images=images_param,
            format="text" # Explicitly ask for text, though default
        )
        
        return response['response'].strip()
    
    async def validate_math(self, latex_content: str):
        prompt = f"""
        Validate this mathematical expression: {latex_content}
        Check for:
        1. LaTeX syntax correctness.
        2. Mathematical plausibility (e.g., does it look like a valid math snippet?).
        3. Obvious incompleteness (e.g. unclosed brackets if not part of a larger system).
        
        Return JSON with keys: "valid" (true/false), "confidence" (0-100 integer), "issues" (list of strings).
        Example: {{"valid": true, "confidence": 85, "issues": ["Possible missing closing parenthesis"]}}
        Return ONLY the JSON object.
        """
        
        response = ollama.generate(
            model=self.models['math'], # Math model for validation
            prompt=prompt,
            format="json" # Request JSON output
        )
        
        try:
            # Ollama might still wrap JSON in text, try to extract
            # A more robust JSON extraction might be needed if models are inconsistent
            json_response_str = response['response'].strip()
            # Handle cases where the model might add markdown ```json ... ```
            if json_response_str.startswith("```json"):
                json_response_str = json_response_str[7:]
            if json_response_str.endswith("```"):
                json_response_str = json_response_str[:-3]
            
            return json.loads(json_response_str)
        except json.JSONDecodeError:
            return {"valid": False, "confidence": 20, "issues": ["Failed to parse AI validation response."]}
        except Exception:
            return {"valid": False, "confidence": 10, "issues": ["Unknown error in AI validation."]}

```

#### **Ngày 3-7: Multi-Engine OCR Setup**
```python
# free_ocr_engines.py
import easyocr
import cv2
from PIL import Image
import pytesseract
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Pix2TexProcessor, VisionEncoderDecoderModel as Pix2TexModel
import asyncio # For async operations later
import logging # For better error reporting
from typing import Dict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FreeOCRManager:
    def __init__(self):
        self.easy_ocr = easyocr.Reader(['en', 'vi'], gpu=torch.cuda.is_available()) # Use GPU if available
        
        self.trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten') # Base is smaller
        self.trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
        
        # pix2tex (LaTeX-OCR) - Ensure you have the model downloaded or specify a local path if needed
        # Using a general config, specific math model might perform better if available via transformers
        try:
            self.pix2tex_processor = Pix2TexProcessor.from_pretrained("vikasdotsh/pix2tex-base") # Check for math specific version if available
            self.pix2tex_model = Pix2TexModel.from_pretrained("vikasdotsh/pix2tex-base")
            logging.info("Pix2Tex model loaded successfully.")
        except Exception as e:
            logging.warning(f"Could not load Pix2Tex model: {e}. It will be unavailable.")
            self.pix2tex_processor = None
            self.pix2tex_model = None

        self.tesseract_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789+-=()[]{}ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\\^_{}<>≤≥∫∑∏√∞αβγδεζηθικλμνξοπρστυφχψω.,;:/'
    
    async def _run_ocr_engine(self, engine_func, *args):
        """Helper to run OCR engine and catch exceptions."""
        try:
            return await asyncio.to_thread(engine_func, *args) # Run synchronous OCR in a thread
        except Exception as e:
            logging.error(f"Error in OCR engine {engine_func.__name__}: {str(e)}")
            return f"Error: {str(e)}"

    async def process_with_all_engines(self, image_path: str) -> Dict[str, str]:
        """Process image with all available free OCR engines asynchronously."""
        try:
            pil_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logging.error(f"Failed to open or convert image {image_path}: {e}")
            return {engine: f"Error: Could not load image - {e}" for engine in ['easyocr', 'tesseract', 'trocr', 'pix2tex']}

        results = {}
        
        # EasyOCR
        async def run_easyocr(img_path):
            # EasyOCR readtext is not directly async, so run in thread
            # Also, easy_ocr.readtext expects a file path or numpy array, not PIL image for best performance
            img_cv2 = cv2.imread(img_path)
            if img_cv2 is None: return "Error: OpenCV could not read image"
            easy_result = self.easy_ocr.readtext(img_cv2)
            return ' '.join([item[1] for item in easy_result])
        results['easyocr'] = await self._run_ocr_engine(run_easyocr, image_path)
        
        # Tesseract  
        async def run_tesseract(img_pil):
            return pytesseract.image_to_string(img_pil, config=self.tesseract_config).strip()
        results['tesseract'] = await self._run_ocr_engine(run_tesseract, pil_image.copy())
        
        # TrOCR
        async def run_trocr(img_pil):
            pixel_values = self.trocr_processor(images=img_pil, return_tensors="pt").pixel_values
            generated_ids = self.trocr_model.generate(pixel_values)
            return self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        results['trocr'] = await self._run_ocr_engine(run_trocr, pil_image.copy())

        # Pix2Tex
        if self.pix2tex_model and self.pix2tex_processor:
            async def run_pix2tex(img_pil):
                inputs = self.pix2tex_processor(images=img_pil, return_tensors="pt")
                generated_ids = self.pix2tex_model.generate(**inputs)
                return self.pix2tex_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            results['pix2tex'] = await self._run_ocr_engine(run_pix2tex, pil_image.copy())
        else:
            results['pix2tex'] = "Error: Model not loaded"
        
        return results
    
    async def consensus_processing(self, results: Dict[str, str], image_path: str):
        """Use local AI to find consensus among OCR results, using image for context."""
        ai_server = LocalAIServer() # Assuming LocalAIServer is accessible
        
        # Filter out error results for the prompt
        valid_results = {k: v for k, v in results.items() if not v.startswith("Error:")}
        
        if not valid_results:
            return "Error: No valid OCR results to form consensus."
            
        # Construct prompt with available results
        ocr_inputs_str = "\n".join([f"{engine.upper()}: {text}" for engine, text in valid_results.items()])

        consensus_prompt_template = f"""
        Analyze the following OCR outputs from an image of a mathematical expression.
        The goal is to produce the most accurate and complete LaTeX representation of the math content.
        Consider that some OCR engines might be better at text, others at symbols, and some specialized for LaTeX.
        
        OCR Outputs:
        {ocr_inputs_str}
        
        Prioritize results from engines like Pix2Tex if available and seemingly correct for math.
        Combine strengths, correct errors, and ensure valid LaTeX syntax.
        If the outputs are wildly different, try to infer the most plausible mathematical expression.
        
        Return *only* the consolidated and corrected LaTeX code. No explanations.
        """
        
        # Use enhance_latex, which can take an image_path for vision model context
        consensus_latex = await ai_server.enhance_latex(
            raw_latex="", # raw_latex is not strictly needed due to custom prompt
            image_path=image_path, 
            context_prompt=consensus_prompt_template
        )
        return consensus_latex

```

### **TUẦN 2: Advanced Free Processing**

#### **Ngày 8-10: Local Vision AI Integration**
```python
# local_vision_ai.py  
from PIL import Image
import ollama # For LLaVA via Ollama
import asyncio
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LocalVisionAI:
    def __init__(self):
        # Using Ollama for Vision Models (LLaVA, Moondream)
        self.llava_model_tag = "llava:7b"  # Default LLaVA, can be configured
        self.moondream_model_tag = "moondream" # Default Moondream
        # BLIP/other transformers models can be added here if preferred over Ollama for certain tasks
        # self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        # self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    async def _query_ollama_vision(self, model_tag: str, prompt: str, image_path: str, format_type: str = "text"):
        try:
            response = await asyncio.to_thread(
                ollama.generate,
                model=model_tag,
                prompt=prompt,
                images=[image_path],
                format=format_type # Ollama can sometimes output JSON if prompted correctly
            )
            return response['response'].strip()
        except Exception as e:
            logging.error(f"Error querying Ollama vision model {model_tag} for {image_path}: {e}")
            return f"Error: Ollama vision query failed - {e}"

    async def get_image_caption(self, image_path: str):
        """Generate a basic caption for the image using a lightweight vision model."""
        prompt = "Provide a brief, one-sentence caption for this image."
        # Moondream is smaller and faster for simple captioning
        caption = await self._query_ollama_vision(self.moondream_model_tag, prompt, image_path)
        return caption

    async def analyze_image_layout(self, image_path: str):
        """Analyze mathematical content layout using LLaVA."""
        caption = await self.get_image_caption(image_path) # Get a quick caption first

        analysis_prompt = f"""
        Image Caption: {caption}
        
        Analyze this image, which likely contains mathematical content. Describe:
        1. Type of content (e.g., equation, matrix, graph with labels, text problem with inline math, table of values).
        2. Estimated complexity (e.g., simple algebra, multi-line integral, complex diagram).
        3. Key visual elements relevant for OCR (e.g., handwritten parts, small subscripts, unusual symbols, noise, skew).
        4. Suggest one or two preprocessing steps that might improve OCR quality (e.g., binarization, deskewing, noise removal, contrast enhancement).
        
        Return a concise analysis.
        """
        
        detailed_analysis = await self._query_ollama_vision(self.llava_model_tag, analysis_prompt, image_path)
        
        return {
            'basic_caption': caption,
            'detailed_analysis': detailed_analysis
        }
    
    async def smart_preprocessing(self, image_path: str, output_dir: str = "./processed_images"):
        """AI-guided image preprocessing using OpenCV based on LLaVA's analysis."""
        import cv2
        import numpy as np
        import os

        os.makedirs(output_dir, exist_ok=True)
        
        try:
            image = cv2.imread(image_path)
            if image is None:
                logging.error(f"Smart Preprocessing: Could not read image {image_path}")
                return image_path # Return original if load fails
        except Exception as e:
            logging.error(f"Smart Preprocessing: Error loading {image_path} with OpenCV: {e}")
            return image_path

        try:
            # Get AI recommendations (could be simplified for speed if LLaVA is slow)
            # For performance, a simpler model or heuristic could be used here.
            # analysis_data = await self.analyze_image_layout(image_path)
            # recommendations = analysis_data.get('detailed_analysis', '').lower()
            
            # Using a faster, heuristic-based approach for now to avoid slow LLaVA call per image just for preprocessing
            # A more advanced approach would parse LLaVA's output for keywords.
            processed_image = image.copy()

            # 0. Convert to grayscale - standard for many OCR
            if len(processed_image.shape) == 3 and processed_image.shape[2] == 3: # Check if it's a color image
                 processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)

            # 1. Denoising
            processed_image = cv2.medianBlur(processed_image, 3) # Good for salt-and-pepper noise

            # 2. Thresholding (Adaptive is often good for varying lighting)
            # Only apply if not already binary-like. Check std dev of pixel values.
            if np.std(processed_image) > 20: # Heuristic: image is not already mostly flat
                processed_image = cv2.adaptiveThreshold(
                    processed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, 2 # Block size, C constant
                )
            
            # 3. Optional: Contrast enhancement (if 'low contrast' was detected by AI)
            # if 'low contrast' in recommendations or 'dim' in recommendations:
            #    alpha = 1.5 # Contrast control
            #    beta = 10   # Brightness control
            #    processed_image = cv2.convertScaleAbs(processed_image, alpha=alpha, beta=beta)

            # 4. Optional: Sharpening (if 'blurry' was detected)
            # if 'blurry' in recommendations:
            #    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            #    processed_image = cv2.filter2D(processed_image, -1, kernel)

            base, ext = os.path.splitext(os.path.basename(image_path))
            processed_filename = f"{base}_processed{ext}"
            processed_path = os.path.join(output_dir, processed_filename)
            
            cv2.imwrite(processed_path, processed_image)
            logging.info(f"Smart Preprocessing: Saved processed image to {processed_path}")
            return processed_path

        except Exception as e:
            logging.error(f"Smart Preprocessing: Error processing {image_path}: {e}")
            return image_path # Return original path if preprocessing fails
```

#### **Ngày 11-14: Batch Processing System**
```python
# free_batch_processor.py
import asyncio
# from concurrent.futures import ThreadPoolExecutor # asyncio.to_thread is preferred for async/await style
import json
from datetime import datetime
import os
import logging
# Assuming other class definitions (FreeOCRManager, LocalVisionAI, LocalAIServer, FreeErrorRecovery) are in scope
# or imported appropriately. For this example, let's assume they are available.
# from free_ocr_engines import FreeOCRManager
# from local_vision_ai import LocalVisionAI
# from local_ai_server import LocalAIServer
# from free_error_recovery import FreeErrorRecovery # For caching

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FreeBatchProcessor:
    def __init__(self, output_base_dir: str = "./ocr_output"):
        self.ocr_manager = FreeOCRManager()
        self.vision_ai = LocalVisionAI()
        self.ai_server = LocalAIServer()
        self.error_recovery = FreeErrorRecovery(cache_dir="./ocr_cache") # Initialize caching
        # self.max_concurrent_tasks = 4 # Limit concurrency to avoid overwhelming system or Ollama
        self.output_base_dir = output_base_dir
        os.makedirs(self.output_base_dir, exist_ok=True)
        self.processed_images_dir = os.path.join(self.output_base_dir, "processed_images")
        os.makedirs(self.processed_images_dir, exist_ok=True)

    async def _process_single_image(self, image_path: str, index: int, total: int):
        logger.info(f"Processing {index+1}/{total}: {os.path.basename(image_path)}")

        # 1. Check cache first
        cached_result = self.error_recovery.load_from_cache(image_path)
        if cached_result:
            logger.info(f"Cache hit for {os.path.basename(image_path)}. Skipping processing.")
            return cached_result

        # 2. Smart Preprocessing (AI-guided or heuristic)
        # This creates a new image, use its path for subsequent steps
        try:
            processed_image_path = await self.vision_ai.smart_preprocessing(image_path, self.processed_images_dir)
        except Exception as e:
            logger.error(f"Error during smart_preprocessing for {image_path}: {e}")
            processed_image_path = image_path # Fallback to original image

        # 3. Multi-Engine OCR
        try:
            raw_ocr_results = await self.ocr_manager.process_with_all_engines(processed_image_path)
        except Exception as e:
            logger.error(f"Error during process_with_all_engines for {processed_image_path}: {e}")
            raw_ocr_results = {"error": str(e)}
        
        # 4. Consensus from OCR results (using AI, with image context)
        consensus_latex = "Error: Consensus failed"
        if "error" not in raw_ocr_results:
            try:
                # Pass original image for better context if preprocessed one is too different or distorted
                consensus_latex = await self.ocr_manager.consensus_processing(raw_ocr_results, image_path) 
            except Exception as e:
                logger.error(f"Error during consensus_processing for {image_path}: {e}")
                consensus_latex = f"Error in consensus: {e}"
        
        # 5. Local AI Enhancement (using image context)
        enhanced_latex = "Error: Enhancement failed"
        if not consensus_latex.startswith("Error:"):
            try:
                # Pass original image for context
                enhanced_latex = await self.ai_server.enhance_latex(consensus_latex, image_path=image_path)
            except Exception as e:
                logger.error(f"Error during enhance_latex for {image_path}: {e}")
                enhanced_latex = f"Error in enhancement: {e}"
        
        # 6. Validation
        validation_output = {"valid": False, "confidence": 0, "issues": ["Validation failed or not run"]}
        if not enhanced_latex.startswith("Error:"):
            try:
                validation_output = await self.ai_server.validate_math(enhanced_latex)
            except Exception as e:
                logger.error(f"Error during validate_math for {enhanced_latex}: {e}")
                validation_output["issues"] = [f"Error in validation: {e}"]
        
        result = {
            'original_image_path': image_path,
            'processed_image_path': processed_image_path,
            'raw_ocr': raw_ocr_results,
            'consensus_latex': consensus_latex,
            'enhanced_latex': enhanced_latex,
            'validation': validation_output,
            'timestamp': datetime.now().isoformat()
        }
        
        # 7. Save to cache
        self.error_recovery.save_to_cache(image_path, result)
        logger.info(f"✅ Finished {index+1}/{total}: {os.path.basename(image_path)}")
        return result

    async def process_batch_free(self, image_paths: list):
        """Process batch of images using only free tools with limited concurrency."""
        # tasks = [self._process_single_image(img_path, i, len(image_paths)) for i, img_path in enumerate(image_paths)]
        # results = await asyncio.gather(*tasks, return_exceptions=True) # return_exceptions=True allows some to fail

        # To limit concurrency if asyncio.gather spawns too many OS threads for sync tasks or Ollama gets overwhelmed:
        # semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        results = []
        # async def bounded_process(img_path, i, total):
        #    async with semaphore:
        #        return await self._process_single_image(img_path, i, total)
        
        # tasks = [bounded_process(img_path, i, len(image_paths)) for i, img_path in enumerate(image_paths)]
        # results = await asyncio.gather(*tasks, return_exceptions=True)

        # Simpler sequential processing for now, can be parallelized with semaphore later if needed
        # Or rely on asyncio.to_thread within sub-components to manage threading
        for i, img_path in enumerate(image_paths):
            try:
                result = await self._process_single_image(img_path, i, len(image_paths))
                results.append(result)
            except Exception as e:
                logger.error(f"FATAL ERROR processing image {img_path}: {e}")
                results.append({
                    'original_image_path': img_path,
                    'error': f"Fatal processing error: {e}",
                    'timestamp': datetime.now().isoformat()
                })
        return results

    def export_results_free(self, results: list, batch_name: str = "batch_results"):
        """Export results using free tools only. Results are saved in a batch-specific subdirectory."""
        
        batch_output_dir = os.path.join(self.output_base_dir, batch_name)
        os.makedirs(batch_output_dir, exist_ok=True)
        
        json_path = os.path.join(batch_output_dir, 'math_ocr_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        latex_path = os.path.join(batch_output_dir, 'compiled_questions.tex')
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write("\\documentclass[12pt]{article}\n")
            f.write("\\usepackage{amsmath, amsfonts, amssymb, graphicx}\n") # Added graphicx
            f.write("\\usepackage[utf8]{inputenc}\n")
            f.write("\\usepackage[a4paper, margin=1in]{geometry}\n") # Better page layout
            f.write("\\title{Math OCR Processed Questions}\n")
            f.write(f"\\date{{{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}}\n")
            f.write("\\begin{document}\n")
            f.write("\\maketitle\n\n")
            
            for i, result in enumerate(results):
                if 'error' in result: # Handle fatal error for an image
                    f.write(f"\\section*{{Question {i+1} (Failed)}}\n")
                    f.write(f"Original Image: {os.path.basename(result.get('original_image_path', 'N/A'))}\\\\\n")
                    f.write(f"Error: {result['error']}\n\n")
                    f.write("\\hrulefill\n\n")
                    continue

                f.write(f"\\section*{{Question {i+1}: {os.path.basename(result['original_image_path'])}}}\n\n")
                # Include original image if path exists
                # For LaTeX, paths might need to be relative or absolute depending on compilation context
                # For simplicity, we'll just list the path. To include, ensure pdflatex can find it.
                # f.write(f"\\subsection*{{Original Image: {result['original_image_path']}}}")
                # f.write(f"\\includegraphics[width=0.8\\textwidth]{{{result['original_image_path']}}}\n\n") # This requires image to be accessible

                f.write("\\subsection*{Recognized LaTeX}\n")
                if result.get('enhanced_latex', '').startswith("Error:"):
                    f.write("Error in LaTeX generation or enhancement.\n")
                    f.write(f"Details: {result['enhanced_latex']}\n\n")
                    if result.get('consensus_latex') and not result['consensus_latex'].startswith("Error:"):
                         f.write("\\textbf{Consensus Fallback:} \\begin{verbatim}\n" + result['consensus_latex'] + "\n\\end{verbatim}\n\n")
                else:
                    f.write(f"$$\n{result['enhanced_latex']}\n$$\n\n")

                f.write("\\subsection*{Validation}\n")
                validation_info = result.get('validation', {})
                f.write(f"Valid: {validation_info.get('valid', 'N/A')}, ")
                f.write(f"Confidence: {validation_info.get('confidence', 'N/A')}\%\\\\\n")
                issues = validation_info.get('issues', [])
                if issues:
                    f.write("Issues: \\begin{itemize}\n")
                    for issue in issues:
                        f.write(f"  \\item {issue}\n")
                    f.write("\\end{itemize}\n")
                f.write("\\hrulefill\n\n")
            
            f.write("\\end{document}\n")
        
        summary_path = os.path.join(batch_output_dir, 'processing_summary.txt')
        successful_results = [r for r in results if 'error' not in r and not r.get('enhanced_latex','').startswith("Error:")]
        total_processed = len(results)
        success_count = len(successful_results)
        
        with open(summary_path, 'w', encoding='utf-utf-8') as f:
            f.write("MATH OCR PROCESSING SUMMARY (FREE VERSION)\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Batch Name: {batch_name}\n")
            f.write(f"Total images submitted: {total_processed}\n")
            f.write(f"Successfully processed (enhanced LaTeX generated): {success_count}\n")
            f.write(f"Processing date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if total_processed > 0:
                f.write(f"Success rate: {success_count/total_processed*100:.1f}%\n\n")
            else:
                f.write("Success rate: N/A (no images processed)\n\n")

            for i, result in enumerate(results):
                is_fatal_error = 'error' in result
                is_enhancement_error = not is_fatal_error and result.get('enhanced_latex','').startswith("Error:")
                
                f.write(f"Image {i+1}: {os.path.basename(result.get('original_image_path', 'N/A'))}\n")
                if is_fatal_error:
                    f.write(f"  Status: ❌ FATAL ERROR ({result['error']})\n")
                elif is_enhancement_error:
                    f.write(f"  Status: ⚠️ Error in Enhancement/Validation\n")
                    f.write(f"  LaTeX (Enhanced): {result['enhanced_latex'][:100]}...\n")
                    f.write(f"  LaTeX (Consensus): {result.get('consensus_latex', 'N/A')[:100]}...\n")
                else:
                    f.write(f"  Status: ✅ Success\n")
                    f.write(f"  LaTeX: {result['enhanced_latex'][:100]}...\n")
                    validation = result.get('validation', {})
                    f.write(f"  Validation: Valid={validation.get('valid')}, Conf={validation.get('confidence')}%\n")
                f.write("\n")
        
        logger.info(f"Results exported to: {batch_output_dir}")
        return {
            'json_export': json_path,
            'latex_export': latex_path,
            'summary_report': summary_path,
            'output_directory': batch_output_dir
        }

```

---

## 🧠 PHASE 2: ADVANCED FREE AI (Tuần 3-4)

### **TUẦN 3: Multi-Model Consensus System**

#### **Ngày 15-17: Local Multi-AI Consensus**
```python
# free_multi_ai.py
import ollama
import asyncio
from typing import List, Dict, Tuple
import logging
import json

# from local_ai_server import LocalAIServer # If used for _query_model wrapper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FreeMultiAI:
    def __init__(self):
        self.models = {
            'math_specialist': 'qwen2.5-math:7b', # Confirm exact Ollama tag
            'general_llm': 'llama3.1:8b', 
            'code_specialist': 'codellama:7b', # Good for LaTeX syntax
            'vision_llm': 'llava:7b' # For image context if needed
        }
        # self.ai_server = LocalAIServer() # Optional: if LocalAIServer has a generic query method
    
    async def _query_model(self, model_key: str, prompt: str, image_path: str = None, is_json_format: bool = False) -> str:
        """Query a specific local model via Ollama."""
        model_tag = self.models.get(model_key)
        if not model_tag:
            raise ValueError(f"Model key {model_key} not found in configuration.")
        
        images_param = [image_path] if image_path and model_key == 'vision_llm' else [] # Only pass images to vision model
        
        try:
            # Use asyncio.to_thread for Ollama's blocking generate call
            response = await asyncio.to_thread(
                ollama.generate,
                model=model_tag,
                prompt=prompt,
                images=images_param,
                format='json' if is_json_format else 'text'
            )
            return response['response'].strip()
        except Exception as e:
            logging.error(f"Model {model_tag} query failed: {str(e)}")
            # raise # Re-raise to be caught by gather
            return f"ERROR: Model {model_tag} query failed: {str(e)}" # Return error string

    async def multi_model_enhancement(self, latex_input: str, image_path: str = None) -> str:
        """
        Get enhanced LaTeX from multiple local AI models and then pick the best one.
        This is an alternative to the single-model enhance_latex if more robustness is needed.
        """
        
        candidate_prompts = {
            'math_specialist': f"Correct and refine this LaTeX code for mathematical accuracy and standard notation. Input: {latex_input}\nReturn only the corrected LaTeX.",
            'general_llm': f"Review this LaTeX for overall clarity, syntax, and logical flow. Input: {latex_input}\nReturn only the improved LaTeX.",
            'code_specialist': f"Strictly check and fix any LaTeX syntax errors in the following code. Ensure it's compilable. Input: {latex_input}\nReturn only the valid LaTeX code."
        }
        
        tasks = []
        for model_key, prompt_text in candidate_prompts.items():
            # Vision model (LLaVA) could also be used here if image_path is provided, 
            # potentially with a prompt asking it to OCR/transcribe the math from the image directly
            # or to use the image as strong context for correcting `latex_input`.
            # For now, focusing on text-based refinement models.
            tasks.append(self._query_model(model_key, prompt_text))
        
        results_with_errors = await asyncio.gather(*tasks)
        
        # Filter out results that are error strings
        successful_results = [res for res in results_with_errors if not res.startswith("ERROR:")]
        
        if not successful_results:
            logging.warning("All models failed in multi_model_enhancement. Returning original input.")
            return latex_input
        
        if len(successful_results) == 1:
            return successful_results[0]
        
        # If multiple successful results, use a voting mechanism
        return await self._consensus_vote_on_text(successful_results, "LaTeX expression")

    async def _consensus_vote_on_text(self, texts: List[str], item_description: str = "item") -> str:
        """Use a designated AI model to vote on the best text among candidates."""
        if not texts:
            return "" # Or raise error
        if len(texts) == 1:
            return texts[0]

        options_str = ""
        option_map = {}
        for i, text_item in enumerate(texts):
            option_char = chr(ord('A') + i)
            option_map[option_char] = text_item
            options_str += f"Option {option_char}: {text_item}\n\n"
            if i >= 25: break # Limit to 26 options (A-Z)

        vote_prompt = f"""
        Compare the following {item_description}s and choose the single best one.
        Consider accuracy, completeness, and correctness based on the item type.
        
        {options_str}
        
        Respond with only the letter (e.g., A, B, C) of the best option. Do not explain.
        """
        
        # Use a capable model for voting, e.g., the math specialist or general LLM
        # Using math_specialist if it's about LaTeX
        vote_response_str = await self._query_model('math_specialist', vote_prompt)
        
        # Parse vote
        # Take the first uppercase letter found in the response.
        parsed_vote = ""
        for char_in_response in vote_response_str.upper():
            if 'A' <= char_in_response <= 'Z':
                parsed_vote = char_in_response
                break
        
        if parsed_vote and parsed_vote in option_map:
            return option_map[parsed_vote]
        else:
            logging.warning(f"Consensus vote failed to pick a valid option (response: '{vote_response_str}'). Defaulting to first option.")
            return texts[0] # Default to the first option if vote parsing fails
```

#### **Ngày 18-21: Free Question Intelligence**
```python
# free_question_intelligence.py
import json
import logging
import os
from typing import List, Dict, Any
import asyncio

# from local_ai_server import LocalAIServer # Not directly needed if FreeMultiAI is used
from free_multi_ai import FreeMultiAI # For querying specific models

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FreeQuestionIntelligence:
    def __init__(self):
        self.multi_ai = FreeMultiAI() # Uses Ollama models defined in FreeMultiAI
    
    async def analyze_question_free(self, latex_content: str, image_path: str = None) -> Dict[str, Any]:
        """Comprehensive question analysis using free local AI."""
        
        # This is the target JSON structure. Make sure the prompt guides the LLM to fill this.
        json_format_description = """
        {
            "question_type": "string (e.g., multiple-choice, calculation, proof, word problem, equation solving)",
            "subject_area": "string (e.g., algebra, geometry, calculus, statistics, trigonometry, linear algebra)", 
            "difficulty_level_estimate": "integer (1-10, 1=very easy, 10=very difficult)",
            "key_mathematical_concepts": ["list of strings (e.g., 'differentiation', 'quadratic equations', 'matrix inversion')"],
            "solution_approach_summary": "string (brief description of how one might solve it, e.g., 'apply Pythagorean theorem', 'integrate by parts')",
            "estimated_time_to_solve_minutes": "integer (e.g., 2, 5, 15)",
            "required_prerequisites": ["list of strings (e.g., 'basic algebra', 'understanding of derivatives')"],
            "latex_quality_assessment": "string (e.g., excellent, good, fair, poor - based on syntax and completeness of the provided LaTeX)",
            "image_quality_assessment": "string (if image provided: e.g., clear, blurry, noisy, well-lit, dark - N/A otherwise)"
        }
        """

        analysis_prompt = f"""
        Analyze the following mathematical question, provided as LaTeX.
        If an image was used to generate this LaTeX, its visual characteristics might also be relevant.
        
        LaTeX Content:
        {latex_content}
        
        {"Image context is also available for this question." if image_path else "No direct image context for this analysis, rely on LaTeX."}

        Provide a comprehensive analysis. Your response MUST be a single, valid JSON object adhering to this structure:
        {json_format_description}
        
        Do NOT include any text before or after the JSON object.
        For fields like "difficulty_level_estimate", provide your best guess.
        If some information cannot be reliably determined, use "N/A" for string fields or appropriate defaults for lists/numbers.
        """
        
        # Determine which model to use. LLaVA if image_path is present, otherwise a math/general LLM.
        model_to_use_key = 'vision_llm' if image_path else 'math_specialist'
        
        analysis_json_str = await self.multi_ai._query_model(
            model_key=model_to_use_key,
            prompt=analysis_prompt,
            image_path=image_path, # Will be ignored if model_to_use_key is not 'vision_llm'
            is_json_format=True # Hint to Ollama to try and produce JSON
        )
        
        try:
            # Try to clean common LLM mistakes with JSON (like markdown code blocks)
            if analysis_json_str.startswith("```json"):
                analysis_json_str = analysis_json_str[7:]
            if analysis_json_str.endswith("```"):
                analysis_json_str = analysis_json_str[:-3]
            analysis_json_str = analysis_json_str.strip()
            
            return json.loads(analysis_json_str)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON from AI analysis: {e}\nRaw response: {analysis_json_str}")
            # Fallback if JSON parsing fails
            return self._get_fallback_analysis(error_message=f"JSON parsing error: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred during question analysis: {e}")
            return self._get_fallback_analysis(error_message=f"Unexpected error: {e}")

    def _get_fallback_analysis(self, error_message: str = "Analysis unavailable") -> Dict[str, Any]:
        """Provides a default/error structure if AI analysis fails."""
        return {
            "question_type": "N/A",
            "subject_area": "N/A", 
            "difficulty_level_estimate": 0,
            "key_mathematical_concepts": [error_message],
            "solution_approach_summary": "N/A",
            "estimated_time_to_solve_minutes": 0,
            "required_prerequisites": [],
            "latex_quality_assessment": "N/A",
            "image_quality_assessment": "N/A",
            "error": error_message # Add an error field to the fallback
        }
    
    async def generate_metadata_for_batch(self, processed_results: List[Dict]) -> List[Dict]:
        """Generate metadata, including AI question analysis, for a batch of processed OCR results."""
        all_metadata = []
        
        for i, result in enumerate(processed_results):
            logger.info(f"Generating metadata for result {i+1}/{len(processed_results)}...")
            if 'error' in result or result.get('enhanced_latex', '').startswith("Error:"):
                analysis = self._get_fallback_analysis(
                    error_message=result.get('error', result.get('enhanced_latex', "Unknown processing error"))
                )
                processing_quality = 'error'
                confidence = 0
            else:
                try:
                    # Pass original image path for LLaVA context
                    analysis = await self.analyze_question_free(
                        result['enhanced_latex'], 
                        result['original_image_path'] 
                    )
                    processing_quality = self._assess_processing_quality(result)
                    confidence = self._calculate_overall_confidence(result, analysis)
                except Exception as e:
                    logging.error(f"Error generating intelligence for {result.get('original_image_path', 'N/A')}: {e}")
                    analysis = self._get_fallback_analysis(error_message=str(e))
                    processing_quality = 'error'
                    confidence = 10 # Low confidence if analysis fails

            item_metadata = {
                'question_id': result.get('original_image_path', f'unknown_image_{i+1}'), # Use image path as a unique ID
                'filename': os.path.basename(result.get('original_image_path', 'N/A')),
                'ai_analysis': analysis,
                'processing_quality_heuristic': processing_quality,
                'overall_confidence_score': confidence, # Combined score
                'timestamps': {'ocr_processing': result.get('timestamp')}
            }
            all_metadata.append(item_metadata)
        
        return all_metadata
    
    def _assess_processing_quality(self, result: Dict) -> str:
        """Heuristic assessment of OCR processing quality based on output."""
        latex = result.get('enhanced_latex', '')
        validation = result.get('validation', {})
        
        if not latex or latex.startswith("Error:"): return 'error'
        if not validation.get('valid', False) and validation.get('confidence', 0) < 30: return 'poor'
        if len(latex) < 10 and not any(cmd in latex for cmd in ['\\frac', '\\sum', '\\int', '\\sqrt']): return 'low' # Very short, no common math commands
        if validation.get('valid', True) and validation.get('confidence', 0) > 70: return 'good'
        if any(cmd in latex for cmd in ['\\frac', '\\sum', '\\int', '\\sqrt', 'matrix', 'align']): return 'good' # Contains complex structures
        return 'fair'
    
    def _calculate_overall_confidence(self, result: Dict, analysis: Dict) -> int:
        """Calculate an overall confidence score (0-100) for the processed question."""
        latex = result.get('enhanced_latex', '')
        validation_conf = result.get('validation', {}).get('confidence', 0)
        
        if not latex or latex.startswith("Error:"): return 0
        
        score = 0
        # From validation
        score += validation_conf * 0.6  # Validation confidence is a major factor (60%)
        
        # From LaTeX quality (heuristic)
        if any(cmd in latex for cmd in ['\\frac', '\\sum', '\\int', '\\sqrt', 'matrix', 'align']):
            score += 20 # Bonus for complex LaTeX structures present
        elif '\\' in latex:
            score += 10 # Basic LaTeX syntax present
        
        # From AI analysis quality (if difficulty makes sense)
        # Ensure 'difficulty_level_estimate' exists and is a number.
        difficulty = analysis.get('difficulty_level_estimate')
        if isinstance(difficulty, (int, float)) and difficulty > 0:
            score += 10
        
        # Normalize to 0-100
        final_score = max(0, min(100, int(score)))
        return final_score

```

### **TUẦN 4: Optimization & Advanced Features**

#### **Ngày 22-24: Performance Optimization**
```python
# free_performance_optimizer.py
import psutil
import time
import asyncio
import logging
from typing import List, Dict, Any
# from free_batch_processor import FreeBatchProcessor # Assumed to be imported

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SystemMonitor:
    def __init__(self, cpu_threshold=85.0, mem_threshold=85.0):
        self.cpu_threshold = cpu_threshold
        self.mem_threshold = mem_threshold

    def get_system_status(self) -> Dict[str, float]:
        """Returns current CPU and Memory usage percentages."""
        cpu_percent = psutil.cpu_percent(interval=0.1) # Short interval for responsiveness
        memory_info = psutil.virtual_memory()
        memory_percent = memory_info.percent
        return {"cpu_percent": cpu_percent, "memory_percent": memory_percent}

    def log_system_status(self):
        status = self.get_system_status()
        logging.info(f"💻 System Status - CPU: {status['cpu_percent']:.1f}%, RAM: {status['memory_percent']:.1f}% used")
    
    def is_system_stressed(self) -> bool:
        """Check if system is under high load based on thresholds."""
        status = self.get_system_status()
        stressed = status['cpu_percent'] > self.cpu_threshold or status['memory_percent'] > self.mem_threshold
        if stressed:
            logging.warning(f"System STRESSED: CPU {status['cpu_percent']:.1f}% (Threshold {self.cpu_threshold}%), RAM {status['memory_percent']:.1f}% (Threshold {self.mem_threshold}%)")
        return stressed

class FreePerformanceOptimizer:
    def __init__(self, batch_processor_instance): # Takes an instance of FreeBatchProcessor
        self.system_monitor = SystemMonitor()
        self.batch_processor = batch_processor_instance # Use the passed instance
        self.min_batch_size = 1
        self.max_batch_size_ideal = 5 # Max items to process concurrently or in a sub-batch
                                      # if FreeBatchProcessor itself doesn't manage internal concurrency well.
                                      # This is more about chunking the input list.
        self.cooldown_period_seconds = 10 # Seconds to wait if system is stressed

    def _determine_adaptive_batch_size(self, total_images: int) -> int:
        """Determine optimal sub-batch size based on system resources and total images."""
        # This is for chunking the input list if `process_batch_free` processes its inputs sequentially
        # or if we want to feed smaller chunks to a parallel `process_batch_free`.
        available_ram_gb = psutil.virtual_memory().available / (1024**3)
        
        if available_ram_gb > 12: # e.g., >12GB free RAM
            batch_size = self.max_batch_size_ideal 
        elif available_ram_gb > 6: # e.g., >6GB free RAM
            batch_size = max(self.min_batch_size, self.max_batch_size_ideal // 2, 2)
        else: # Low RAM
            batch_size = self.min_batch_size
            
        return min(batch_size, total_images) # Don't exceed total images

    async def adaptive_run_processing(self, all_image_paths: List[str], batch_name: str = "optimized_batch") -> List[Dict]:
        """
        Processes all images by breaking them into adaptive sub-batches.
        Manages system load between these sub-batches.
        The `FreeBatchProcessor.process_batch_free` itself might internally use asyncio.gather for parallelism.
        This function primarily chunks the overall list of images and introduces cooldowns.
        """
        total_images = len(all_image_paths)
        if total_images == 0:
            logging.info("No images to process.")
            return []

        overall_results = []
        
        # Determine how to chunk the main list of images
        # This is useful if FreeBatchProcessor's own parallelism needs to be managed
        # or if it processes images sequentially within its call.
        # If FreeBatchProcessor is fully async and manages its own concurrency well,
        # this outer chunking might be less critical, but cooldowns are still useful.
        
        # For this implementation, let's assume process_batch_free takes a list and processes it.
        # We will call it once with all images, but monitor/cooldown *before* the main call if needed.
        # A more complex optimizer might break all_image_paths into smaller lists.

        logging.info(f"🔧 Preparing to process {total_images} images for batch '{batch_name}'.")
        self.system_monitor.log_system_status()
        
        if self.system_monitor.is_system_stressed():
            logging.warning(f"System is initially stressed. Cooling down for {self.cooldown_period_seconds}s before starting.")
            await asyncio.sleep(self.cooldown_period_seconds)

        start_time = time.time()
        
        # Process all images in one go by FreeBatchProcessor, which handles its own internal logic
        # (caching, potential internal parallelism if implemented with asyncio.gather, etc.)
        # The `process_batch_free` method in the provided plan processes sequentially per image after an initial gather,
        # so this top-level optimizer ensures system health checks before large batches.
        
        # If `process_batch_free` itself were to be chunked:
        # current_idx = 0
        # while current_idx < total_images:
        #     sub_batch_size = self._determine_adaptive_batch_size(total_images - current_idx)
        #     image_sub_batch = all_image_paths[current_idx : current_idx + sub_batch_size]
        #     logging.info(f"📦 Processing sub-batch of {len(image_sub_batch)} images (Start index: {current_idx}).")
        #
        #     # Monitor system before processing this sub-batch
        #     self.system_monitor.log_system_status()
        #     if self.system_monitor.is_system_stressed():
        #         logging.warning(f"System stressed. Cooling down for {self.cooldown_period_seconds}s.")
        #         await asyncio.sleep(self.cooldown_period_seconds)
        #
        #     sub_batch_results = await self.batch_processor.process_batch_free(image_sub_batch) # This should be the async call
        #     overall_results.extend(sub_batch_results)
        #     current_idx += sub_batch_size
        # else: # No chunking, process all at once by FreeBatchProcessor

        overall_results = await self.batch_processor.process_batch_free(all_image_paths)
            
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_image = total_time / total_images if total_images > 0 else 0
        
        logging.info(f"🏁 Batch '{batch_name}' completed processing {total_images} images in {total_time:.2f}s.")
        logging.info(f"⏱️ Average time per image: {avg_time_per_image:.2f}s.")
        self.system_monitor.log_system_status() # Final status
        
        # Export results (this is a synchronous operation in the current plan)
        # Could be made async if I/O is heavy.
        if overall_results:
            # The batch_processor's export method already handles subdirectories.
            export_paths = self.batch_processor.export_results_free(overall_results, batch_name=batch_name)
            logging.info(f"📊 Results for batch '{batch_name}' exported to {export_paths.get('output_directory')}")
        else:
            logging.info(f"No results to export for batch '{batch_name}'.")
            
        return overall_results
```

#### **Ngày 25-28: Error Recovery & Caching**
*(The `free_error_recovery.py` file was completed at the beginning of this response. It includes image hashing, loading from cache, saving to cache, a retry_operation helper, and a basic handle_component_failure. This class is instantiated and used within `FreeBatchProcessor` for caching.)*

**Key Integration Points for `FreeErrorRecovery`:**
1.  **Instantiation:** `FreeBatchProcessor` creates an instance of `FreeErrorRecovery` in its `__init__`.
2.  **Cache Check:** In `FreeBatchProcessor._process_single_image`, before any processing, `self.error_recovery.load_from_cache(image_path)` is called. If it returns data, processing is skipped.
3.  **Cache Save:** After successful processing of an image in `_process_single_image`, `self.error_recovery.save_to_cache(image_path, result)` is called.
4.  **Retry (Conceptual):** The `retry_operation` method in `FreeErrorRecovery` can be used to wrap specific critical async calls within the `FreeBatchProcessor` or its sub-components if they are prone to transient failures (e.g., network calls to a self-hosted Ollama if it's temporarily busy).

    ```python
    # Example of using retry_operation within a component:
    # In LocalAIServer or FreeOCRManager for a specific model call:
    # async def some_flaky_ollama_call(self, model, prompt):
    #     # ... ollama.generate logic ...
    #     pass 
    #
    # async def robust_ollama_call(self, model, prompt):
    #     error_recovery = FreeErrorRecovery() # Or get from class instance
    #     try:
    #         return await error_recovery.retry_operation(self.some_flaky_ollama_call, model, prompt)
    #     except Exception as e:
    #         # Handle ultimate failure after retries
    #         logging.error(f"Ollama call for model {model} failed definitively: {e}")
    #         return {"error": str(e)} # Or appropriate error structure
    ```

---

## 🎨 PHASE 3: OPTIMIZATION, EVALUATION & UI (Tuần 5-6)

### **TUẦN 5: System Optimization & Evaluation**

#### **Ngày 29-31: End-to-End Testing & Bottleneck Identification**
*   **Goal**: Run the entire pipeline with a diverse set of 50-100 test images (various complexities, handwritten, printed, noisy, clean).
*   **Activities**:
    1.  **Full Pipeline Runs**: Execute `FreePerformanceOptimizer.adaptive_run_processing` with test image sets.
    2.  **Log Analysis**: Thoroughly review logs from all components (`FreeOCRManager`, `LocalAIServer`, `FreeBatchProcessor`, etc.) for errors, warnings, and performance metrics (timing for each step).
    3.  **Resource Monitoring**: Use `psutil` (as in `SystemMonitor`) and system tools (htop, Task Manager, GPU monitoring tools like `nvidia-smi`) during runs to identify CPU, RAM, VRAM, and I/O bottlenecks.
    4.  **Component Profiling**:
        *   Wrap key function calls (e.g., each OCR engine, each Ollama call, preprocessing) with `time.time()` stamps to measure execution duration.
        *   Python's `cProfile` or `line_profiler` can be used for more detailed analysis if specific Python functions are suspected bottlenecks.
    5.  **Identify Slowest Steps**: Pinpoint which parts of `_process_single_image` consume the most time (e.g., a specific OCR engine, LLaVA analysis, Ollama LLM calls).
    6.  **Error Rate Tracking**: Note how often individual components fail or produce poor results.
*   **Deliverables**:
    *   Bottleneck report detailing the slowest components and resource contention points.
    *   List of most common errors and failure modes.
    *   Initial (qualitative) assessment of output quality.

#### **Ngày 32-33: Model & Prompt Optimization**
*   **Goal**: Improve performance and accuracy based on findings from bottleneck analysis.
*   **Activities**:
    1.  **Ollama Model Selection/Quantization**:
        *   Experiment with different quantizations (e.g., Q4, Q5, Q8) for Ollama models. Smaller quantizations are faster and use less RAM/VRAM but might reduce accuracy. Find a balance.
        *   Test alternative models available via Ollama for specific tasks (e.g., smaller LLaVA variants, different fine-tunes of Llama/Qwen).
    2.  **Prompt Engineering**:
        *   Refine prompts for `LocalAIServer` (enhancement, validation) and `FreeMultiAI` (consensus, analysis) based on observed LLM outputs.
        *   Make prompts more specific, add few-shot examples if needed, or simplify instructions if models struggle with complex prompts.
        *   Adjust prompts for `format="json"` in Ollama to improve reliability of JSON output.
    3.  **OCR Engine Configuration**:
        *   Tune `Tesseract` PSM modes or character whitelists.
        *   If a specific OCR engine is consistently poor or very slow for little gain, consider making it optional or removing it.
        *   Check if Hugging Face `transformers` models (TrOCR, Pix2Tex) can be run with `float16` for speed on GPUs.
    4.  **Preprocessing Strategy Refinement**:
        *   If `smart_preprocessing` using LLaVA is too slow, develop more heuristic-based rules or use a much smaller vision model (like Moondream through Ollama) for preprocessing hints.
        *   Test different OpenCV preprocessing steps (thresholding types, denoising parameters).
*   **Deliverables**:
    *   Updated Ollama model tags and configurations.
    *   Revised prompt templates.
    *   Optimized image preprocessing pipeline.
    *   Report on performance/accuracy changes from optimizations.

#### **Ngày 34-35: Accuracy Evaluation & Benchmarking**
*   **Goal**: Quantitatively measure the accuracy of the OCR system against a ground truth dataset.
*   **Activities**:
    1.  **Prepare Ground Truth Dataset**:
        *   Select 20-30 representative images.
        *   Manually transcribe the correct LaTeX for each image. This is time-consuming but essential.
    2.  **Define Accuracy Metrics**:
        *   **Exact Match (EM)**: Percentage of OCR outputs that exactly match the ground truth LaTeX. (Very strict).
        *   **Token Edit Distance (e.g., Levenshtein distance)**: Measures similarity at the token level (LaTeX commands, symbols, numbers). Normalize by length.
        *   **Semantic Similarity (Advanced)**: Use an LLM to judge if the OCR output is mathematically equivalent to the ground truth. Prompt: "Are these two LaTeX expressions mathematically equivalent? Respond YES or NO. Expression 1: {ocr_output} Expression 2: {ground_truth}".
        *   **Compilation Check**: Percentage of outputs that compile correctly with a LaTeX compiler (e.g., `pdflatex`).
    3.  **Run Benchmark**: Process the ground truth image set with the optimized pipeline.
    4.  **Calculate Metrics**: Compare generated LaTeX with ground truth using the defined metrics.
    5.  **Analyze Discrepancies**: Manually review cases where OCR output differs significantly from ground truth to understand common error types.
*   **Deliverables**:
    *   Ground truth dataset (images + LaTeX).
    *   Accuracy benchmark report with scores for each metric.
    *   Analysis of common error patterns.
    *   Comparison against the 85%+ target accuracy.

### **TUẦN 6: UI & Deployment Preparation**

#### **Ngày 36-38: Simple UI Development (Streamlit or Gradio)**
*   **Goal**: Create a basic user interface for easy interaction with the Math OCR system.
*   **Technology Choice**: Streamlit (more Pythonic) or Gradio (often quicker for ML demos). Let's assume Streamlit.
*   **Core UI Features**:
    ```python
    # streamlit_app.py
    import streamlit as st
    import os
    import asyncio
    from datetime import datetime

    # Need to make FreePerformanceOptimizer and its dependencies importable
    # from free_performance_optimizer import FreePerformanceOptimizer
    # from free_batch_processor import FreeBatchProcessor
    # Placeholder for actual backend classes
    class PlaceholderFreeBatchProcessor:
        async def process_batch_free(self, image_paths):
            st.info(f"Simulating processing for {len(image_paths)} images...")
            results = []
            for i, p in enumerate(image_paths):
                await asyncio.sleep(0.1) # Simulate work
                results.append({
                    'original_image_path': p,
                    'processed_image_path': p.replace('.', '_proc.'),
                    'raw_ocr': {'easyocr': 'sim_easy', 'tesseract': 'sim_tess'},
                    'consensus_latex': f"\\text{{Consensus for }} {os.path.basename(p)}",
                    'enhanced_latex': f"\\text{{Enhanced for }} {os.path.basename(p)}: x^2+y^2=z^2",
                    'validation': {"valid": True, "confidence": 80+i, "issues": []},
                    'timestamp': datetime.now().isoformat()
                })
            st.success("Simulation complete!")
            return results
        def export_results_free(self, results, batch_name):
            st.info(f"Simulating export for {batch_name} with {len(results)} results.")
            output_dir = os.path.join("streamlit_ocr_output", batch_name)
            os.makedirs(output_dir, exist_ok=True)
            json_path = os.path.join(output_dir, "results.json")
            # with open(json_path, "w") as f: json.dump(results, f) # Actual dump
            return {'json_export': json_path, 'latex_export': 'sim.tex', 'summary_report': 'sim.txt', 'output_directory': output_dir}

    class PlaceholderFreePerformanceOptimizer:
        def __init__(self, batch_processor):
            self.batch_processor = batch_processor
        async def adaptive_run_processing(self, image_paths, batch_name):
            # In real app, this calls batch_processor.process_batch_free and export_results_free
            results = await self.batch_processor.process_batch_free(image_paths)
            if results:
                export_info = self.batch_processor.export_results_free(results, batch_name)
                return results, export_info
            return [], None

    # --- Streamlit App ---
    st.set_page_config(layout="wide", page_title="Free Math OCR")
    st.title("🆓 Free Math OCR System")

    # Initialize backend (ideally once, using st.singleton or st.cache_resource for real app)
    # For simplicity in this plan, we'll instantiate directly.
    # Ensure Ollama server is running separately.
    if 'backend_initialized' not in st.session_state:
        # batch_proc = FreeBatchProcessor(output_base_dir="./streamlit_ocr_output")
        # perf_optimizer = FreePerformanceOptimizer(batch_processor_instance=batch_proc)
        batch_proc = PlaceholderFreeBatchProcessor() # Using placeholder for plan
        perf_optimizer = PlaceholderFreePerformanceOptimizer(batch_proc) # Using placeholder
        st.session_state.perf_optimizer = perf_optimizer
        st.session_state.backend_initialized = True
        st.success("Backend components initialized (simulated). Ensure Ollama is running!")

    perf_optimizer = st.session_state.perf_optimizer

    # File uploader
    uploaded_files = st.file_uploader("Upload math question images (PNG, JPG)", 
                                      type=["png", "jpg", "jpeg"], 
                                      accept_multiple_files=True)

    temp_upload_dir = "temp_streamlit_uploads"
    if not os.path.exists(temp_upload_dir):
        os.makedirs(temp_upload_dir)

    image_paths_to_process = []
    if uploaded_files:
        st.write(f"{len(uploaded_files)} images selected.")
        for uploaded_file in uploaded_files:
            # Save to a temporary location to get a stable file path
            file_path = os.path.join(temp_upload_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            image_paths_to_process.append(file_path)
            st.image(file_path, caption=uploaded_file.name, width=200)

    if st.button("🚀 Process Batch", disabled=not image_paths_to_process):
        if image_paths_to_process:
            batch_name = f"streamlit_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with st.spinner(f"Processing {len(image_paths_to_process)} images... This may take a while."):
                # Run the async processing
                # Streamlit doesn't natively support top-level async functions in event handlers easily
                # A common pattern is to use asyncio.run() or nest_asyncio if already in an event loop.
                # For simplicity in this plan, we'll assume direct call works or use placeholder.
                
                # --- This is how you'd typically run async code in Streamlit ---
                # loop = asyncio.new_event_loop()
                # asyncio.set_event_loop(loop)
                # results, export_info = loop.run_until_complete(
                #     perf_optimizer.adaptive_run_processing(image_paths_to_process, batch_name)
                # )
                # loop.close()
                # --- Placeholder direct call for this plan structure ---
                results, export_info = asyncio.run(perf_optimizer.adaptive_run_processing(image_paths_to_process, batch_name))


            st.session_state.results = results
            st.session_state.export_info = export_info
            st.success(f"Batch '{batch_name}' processed!")
            if export_info:
                 st.info(f"Results exported to: {export_info['output_directory']}")
                 # Provide download for summary
                 summary_path = export_info.get('summary_report')
                 if summary_path and os.path.exists(summary_path):
                     with open(summary_path, "r", encoding='utf-8') as f_sum:
                         st.download_button("Download Summary Report", f_sum.read(), file_name="summary_report.txt")


    # Display results
    if 'results' in st.session_state and st.session_state.results:
        st.subheader("📊 Processing Results")
        results_to_display = st.session_state.results
        
        for i, res in enumerate(results_to_display):
            st.markdown(f"--- \n ### 🖼️ Image: {os.path.basename(res['original_image_path'])}")
            col1, col2 = st.columns(2)
            with col1:
                st.image(res['original_image_path'], use_column_width=True)
            with col2:
                if 'error' in res:
                    st.error(f"Processing Error: {res['error']}")
                else:
                    st.markdown("**Enhanced LaTeX:**")
                    st.code(res['enhanced_latex'], language='latex')
                    try:
                        st.latex(res['enhanced_latex']) # Render LaTeX if possible
                    except st.StreamlitAPIException as e:
                        st.warning(f"Could not render LaTeX directly: {e}. Displaying as code.")
                    
                    val = res['validation']
                    st.markdown(f"**Validation:** Valid: `{val['valid']}`, Confidence: `{val['confidence']}%`")
                    if val['issues']:
                        st.markdown("**Issues:**")
                        for issue in val['issues']:
                            st.caption(f"- {issue}")
            
            with st.expander("Show Full OCR Details"):
                st.json(res) # Display full JSON for debugging

    # Clean up temporary uploaded files (optional, or do it more robustly)
    # This is tricky with Streamlit's execution model. A cron job or manual cleanup might be better.
    ```
*   **Deliverables**:
    *   Working Streamlit/Gradio application script (`streamlit_app.py`).
    *   Brief user guide on how to run and use the UI.

#### **Ngày 39-40: Documentation & Packaging**
*   **Goal**: Prepare the system for easy setup and use by others (or future self).
*   **Activities**:
    1.  **README.md**:
        *   Project overview, goals, features.
        *   Detailed setup instructions:
            *   OS prerequisites (Python, Ollama).
            *   Cloning the repository.
            *   Installing Ollama and pulling required models (`ollama pull ...`).
            *   Setting up Python virtual environment.
            *   Installing Python dependencies (`pip install -r requirements.txt`).
        *   How to run the OCR pipeline (CLI command, or running the Streamlit app).
        *   Troubleshooting common issues.
    2.  **`requirements.txt`**: Generate `pip freeze > requirements.txt`.
    3.  **`Dockerfile` (Optional, for containerization)**:
        *   Base image (e.g., Python official, or one with CUDA for GPU if PyTorch/PaddlePaddle GPU is used).
        *   Copy project files.
        *   Install system dependencies (like Tesseract OCR language data if not handled by pip).
        *   Install Python dependencies from `requirements.txt`.
        *   Set up Ollama (or assume it runs on host and container connects to it). This can be tricky. A simpler Dockerfile might assume Ollama is run separately.
        *   Define entrypoint/CMD.
    4.  **Code Comments and Docstrings**: Ensure all major functions and classes have clear docstrings explaining their purpose, arguments, and return values.
*   **Deliverables**:
    *   Comprehensive `README.md`.
    *   `requirements.txt` file.
    *   (Optional) `Dockerfile` and instructions for building/running the Docker image.
    *   Well-commented codebase.

#### **Ngày 41-42: Final Review, Demo & Release Plan**
*   **Goal**: Conduct a final review of the entire system, prepare a demonstration, and outline steps for a "v1.0 Free" release.
*   **Activities**:
    1.  **Code Review**: Final pass over the codebase for clarity, consistency, and any obvious bugs.
    2.  **Documentation Review**: Check `README.md` and other docs for accuracy and completeness.
    3.  **Test Suite (Basic)**: If time permits, write a few basic automated tests (e.g., using `pytest`) for critical utility functions or a simple end-to-end test with one image.
    4.  **Prepare Demo**: Plan a demonstration showcasing:
        *   Setup process.
        *   Running the Streamlit UI with a few example images.
        *   Showing the output (LaTeX, validation, summary reports).
        *   Highlighting key features (local processing, multi-engine OCR, AI enhancement).
    5.  **"Release" Checklist**:
        *   Confirm all Ollama model tags used are public and correct.
        *   Verify licensing of all open-source components is compatible with the project's intended use/distribution.
        *   Tag a version in Git (e.g., `v1.0.0-free`).
        *   Consider where to host the code (e.g., GitHub repository).
    6.  **Future Work Brainstorm**: List potential improvements or features for future iterations (e.g., more advanced UI, support for PDF inputs, more sophisticated error analysis, active learning loop for model fine-tuning).
*   **Deliverables**:
    *   Final internal demo.
    *   "Release candidate" codebase tagged in version control.
    *   Short document outlining potential future work.
    *   Project ready for sharing/internal release.

--- END OF FILE free_math_ocr_plan.md ---