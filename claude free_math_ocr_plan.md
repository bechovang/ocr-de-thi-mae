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
- minicpm-v:8b (Efficient vision model)
- moondream2 (Lightweight vision)

# Specialized Models
- mathpix-alternative (Open source math OCR)
- nomeroff-net (Text detection)
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
paddlepaddle         # Paddle OCR
nltk                 # Natural language processing

# Utilities
numpy, scipy         # Numerical computing
pandas               # Data processing
matplotlib           # Plotting
tqdm                 # Progress bars
asyncio              # Async processing
```

---

## 📅 TIMELINE MIỄN PHÍ: 6 TUẦN

| Phase | Thời gian | Mục tiêu chính |
|-------|-----------|----------------|
| **Phase 1** | Tuần 1-2 | Setup Local AI + Core OCR |
| **Phase 2** | Tuần 3-4 | Local AI Integration |
| **Phase 3** | Tuần 5-6 | Optimization + UI |

---

## 🚀 PHASE 1: LOCAL AI SETUP (Tuần 1-2)

### **TUẦN 1: Environment & Local AI Setup**

#### **Ngày 1-2: Local AI Environment**
```bash
# 1. Install Ollama (Local AI Runtime)
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Pull free math-specialized models
ollama pull llama3.1:8b
ollama pull qwen2.5-math:7b
ollama pull llava:7b
ollama pull codellama:7b

# 3. Install Python dependencies
pip install torch transformers
pip install opencv-python pillow
pip install easyocr paddlepaddle
pip install pytesseract
```

**Local AI Server Setup:**
```python
# local_ai_server.py
import ollama
from typing import Dict, Any
import asyncio

class LocalAIServer:
    def __init__(self):
        self.models = {
            'math': 'qwen2.5-math:7b',      # Specialized for math
            'general': 'llama3.1:8b',       # General purpose  
            'vision': 'llava:7b',           # Vision + text
            'code': 'codellama:7b'          # LaTeX generation
        }
    
    async def enhance_latex(self, raw_latex: str, image_path: str = None):
        prompt = f"""
        Fix and enhance this LaTeX OCR output:
        {raw_latex}
        
        Tasks:
        1. Correct syntax errors
        2. Standardize mathematical notation
        3. Add missing brackets
        4. Verify mathematical logic
        
        Return only the corrected LaTeX, no explanations.
        """
        
        if image_path:
            # Use vision model for image context
            response = ollama.generate(
                model=self.models['vision'],
                prompt=prompt,
                images=[image_path]
            )
        else:
            # Use specialized math model
            response = ollama.generate(
                model=self.models['math'],
                prompt=prompt
            )
        
        return response['response']
    
    async def validate_math(self, latex_content: str):
        prompt = f"""
        Validate this mathematical expression for correctness:
        {latex_content}
        
        Check for:
        1. Syntax correctness
        2. Mathematical validity
        3. Completeness
        
        Return JSON: {{"valid": true/false, "confidence": 0-100, "issues": []}}
        """
        
        response = ollama.generate(
            model=self.models['math'],
            prompt=prompt
        )
        
        return response['response']
```

#### **Ngày 3-7: Multi-Engine OCR Setup**
```python
# free_ocr_engines.py
import easyocr
import cv2
from PIL import Image
import pytesseract
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

class FreeOCRManager:
    def __init__(self):
        # Initialize all free OCR engines
        self.easy_ocr = easyocr.Reader(['en', 'vi'])
        
        # TrOCR for handwritten math
        self.trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
        self.trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
        
        # Configure Tesseract for math
        self.tesseract_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789+-=()[]{}ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\\^_{}[]()+-=<>≤≥∫∑∏√∞αβγδεζηθικλμνξοπρστυφχψω'
    
    async def process_with_all_engines(self, image_path: str):
        """Process image with all available free OCR engines"""
        image = cv2.imread(image_path)
        pil_image = Image.open(image_path)
        
        results = {}
        
        # EasyOCR
        try:
            easy_result = self.easy_ocr.readtext(image)
            results['easyocr'] = ' '.join([item[1] for item in easy_result])
        except Exception as e:
            results['easyocr'] = f"Error: {str(e)}"
        
        # Tesseract  
        try:
            tesseract_result = pytesseract.image_to_string(pil_image, config=self.tesseract_config)
            results['tesseract'] = tesseract_result.strip()
        except Exception as e:
            results['tesseract'] = f"Error: {str(e)}"
        
        # TrOCR
        try:
            pixel_values = self.trocr_processor(pil_image, return_tensors="pt").pixel_values
            generated_ids = self.trocr_model.generate(pixel_values)
            trocr_result = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            results['trocr'] = trocr_result
        except Exception as e:
            results['trocr'] = f"Error: {str(e)}"
        
        return results
    
    async def consensus_processing(self, results: Dict[str, str]):
        """Use local AI to find consensus among OCR results"""
        ai_server = LocalAIServer()
        
        consensus_prompt = f"""
        Compare these OCR results and provide the best consolidated output:
        
        EasyOCR: {results.get('easyocr', 'N/A')}
        Tesseract: {results.get('tesseract', 'N/A')}  
        TrOCR: {results.get('trocr', 'N/A')}
        
        Return the most accurate mathematical expression in LaTeX format.
        """
        
        consensus = await ai_server.enhance_latex(consensus_prompt)
        return consensus
```

### **TUẦN 2: Advanced Free Processing**

#### **Ngày 8-10: Local Vision AI Integration**
```python
# local_vision_ai.py  
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image

class LocalVisionAI:
    def __init__(self):
        # Load free vision-language model
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # Load local LLaVA via Ollama for complex vision tasks
        self.llava_model = "llava:7b"
    
    async def analyze_image_layout(self, image_path: str):
        """Analyze mathematical content layout using free vision AI"""
        # Basic image captioning
        image = Image.open(image_path)
        inputs = self.processor(image, return_tensors="pt")
        
        out = self.model.generate(**inputs, max_length=50)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        
        # Enhanced analysis with local LLaVA
        analysis_prompt = """
        Analyze this mathematical content image:
        1. Type of math content (equation, graph, table, etc.)
        2. Complexity level (simple, moderate, complex)
        3. Key elements to focus on during OCR
        4. Preprocessing recommendations
        
        Provide structured analysis.
        """
        
        import ollama
        response = ollama.generate(
            model=self.llava_model,
            prompt=analysis_prompt,
            images=[image_path]
        )
        
        return {
            'basic_caption': caption,
            'detailed_analysis': response['response']
        }
    
    async def smart_preprocessing(self, image_path: str):
        """AI-guided image preprocessing using free tools"""
        import cv2
        import numpy as np
        
        # Load image
        image = cv2.imread(image_path)
        
        # Get AI recommendations for preprocessing
        analysis = await self.analyze_image_layout(image_path)
        
        # Apply preprocessing based on AI analysis
        processed_image = image.copy()
        
        # Standard preprocessing pipeline
        if 'dark' in analysis['basic_caption'].lower():
            # Brighten dark images
            processed_image = cv2.convertScaleAbs(processed_image, alpha=1.2, beta=30)
        
        if 'blurry' in analysis['basic_caption'].lower():
            # Sharpen blurry images
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            processed_image = cv2.filter2D(processed_image, -1, kernel)
        
        # Noise reduction
        processed_image = cv2.medianBlur(processed_image, 3)
        
        # Save processed image
        processed_path = image_path.replace('.', '_processed.')
        cv2.imwrite(processed_path, processed_image)
        
        return processed_path
```

#### **Ngày 11-14: Batch Processing System**
```python
# free_batch_processor.py
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime
import os

class FreeBatchProcessor:
    def __init__(self):
        self.ocr_manager = FreeOCRManager()
        self.vision_ai = LocalVisionAI()
        self.ai_server = LocalAIServer()
        self.max_workers = 4
    
    async def process_batch_free(self, image_paths: list):
        """Process batch of images using only free tools"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Stage 1: Parallel preprocessing
            preprocessing_tasks = [
                self.vision_ai.smart_preprocessing(img_path) 
                for img_path in image_paths
            ]
            processed_paths = await asyncio.gather(*preprocessing_tasks)
            
            # Stage 2: Parallel OCR processing
            ocr_tasks = [
                self.ocr_manager.process_with_all_engines(processed_path)
                for processed_path in processed_paths
            ]
            ocr_results = await asyncio.gather(*ocr_tasks)
            
            # Stage 3: AI enhancement with local models
            for i, (original_path, ocr_result) in enumerate(zip(image_paths, ocr_results)):
                # Consensus processing
                consensus = await self.ocr_manager.consensus_processing(ocr_result)
                
                # Local AI enhancement
                enhanced = await self.ai_server.enhance_latex(consensus, original_path)
                
                # Validation
                validation = await self.ai_server.validate_math(enhanced)
                
                result = {
                    'image_path': original_path,
                    'processed_path': processed_paths[i],
                    'raw_ocr': ocr_result,
                    'consensus': consensus,
                    'enhanced_latex': enhanced,
                    'validation': validation,
                    'timestamp': datetime.now().isoformat()
                }
                
                results.append(result)
                print(f"✅ Processed {i+1}/{len(image_paths)}: {os.path.basename(original_path)}")
        
        return results
    
    async def export_results_free(self, results: list, output_dir: str):
        """Export results using free tools only"""
        os.makedirs(output_dir, exist_ok=True)
        
        # JSON export
        json_path = os.path.join(output_dir, 'math_ocr_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # LaTeX compilation
        latex_path = os.path.join(output_dir, 'compiled_questions.tex')
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write("\\documentclass{article}\n")
            f.write("\\usepackage{amsmath,amsfonts,amssymb}\n")
            f.write("\\usepackage[utf8]{inputenc}\n")
            f.write("\\begin{document}\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"\\section{{Question {i}}}\n")
                f.write(f"$$\n{result['enhanced_latex']}\n$$\n\n")
            
            f.write("\\end{document}\n")
        
        # Summary report
        summary_path = os.path.join(output_dir, 'processing_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("MATH OCR PROCESSING SUMMARY (FREE VERSION)\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total questions processed: {len(results)}\n")
            f.write(f"Processing date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Success rate: {len([r for r in results if 'Error' not in str(r['enhanced_latex'])])/len(results)*100:.1f}%\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"Question {i}:\n")
                f.write(f"  File: {os.path.basename(result['image_path'])}\n")
                f.write(f"  LaTeX: {result['enhanced_latex'][:100]}...\n")
                f.write(f"  Status: {'✅ Success' if 'Error' not in str(result['enhanced_latex']) else '❌ Error'}\n\n")
        
        return {
            'json_export': json_path,
            'latex_export': latex_path,
            'summary_report': summary_path
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
from typing import List, Dict

class FreeMultiAI:
    def __init__(self):
        self.models = {
            'math_specialist': 'qwen2.5-math:7b',
            'general_llm': 'llama3.1:8b', 
            'code_specialist': 'codellama:7b',
            'vision_llm': 'llava:7b'
        }
    
    async def multi_model_consensus(self, latex_input: str, image_path: str = None):
        """Get consensus from multiple local AI models"""
        
        tasks = []
        
        # Task 1: Math specialist
        math_prompt = f"""
        As a mathematics expert, analyze and correct this LaTeX:
        {latex_input}
        
        Focus on mathematical accuracy and proper notation.
        Return only the corrected LaTeX.
        """
        tasks.append(self._query_model('math_specialist', math_prompt))
        
        # Task 2: General language model  
        general_prompt = f"""
        Review this mathematical LaTeX for syntax and clarity:
        {latex_input}
        
        Ensure proper LaTeX syntax and readability.
        Return only the improved LaTeX.
        """
        tasks.append(self._query_model('general_llm', general_prompt))
        
        # Task 3: Code specialist for LaTeX syntax
        code_prompt = f"""
        Check this LaTeX code for syntax errors:
        {latex_input}
        
        Fix any LaTeX compilation issues.
        Return only the corrected code.
        """
        tasks.append(self._query_model('code_specialist', code_prompt))
        
        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        valid_results = [r for r in results if not isinstance(r, Exception)]
        
        if len(valid_results) >= 2:
            # Use another AI to choose best result
            return await self._consensus_vote(valid_results)
        elif len(valid_results) == 1:
            return valid_results[0]
        else:
            return latex_input  # Return original if all failed
    
    async def _query_model(self, model_key: str, prompt: str, image_path: str = None):
        """Query a specific local model"""
        try:
            if image_path and model_key == 'vision_llm':
                response = ollama.generate(
                    model=self.models[model_key],
                    prompt=prompt,
                    images=[image_path]
                )
            else:
                response = ollama.generate(
                    model=self.models[model_key],
                    prompt=prompt
                )
            return response['response'].strip()
        except Exception as e:
            raise Exception(f"Model {model_key} failed: {str(e)}")
    
    async def _consensus_vote(self, results: List[str]):
        """Use AI to vote on best result among candidates"""
        vote_prompt = f"""
        Compare these LaTeX expressions and choose the best one:
        
        Option A: {results[0]}
        Option B: {results[1]}
        {f"Option C: {results[2]}" if len(results) > 2 else ""}
        
        Consider: mathematical accuracy, LaTeX syntax, clarity
        Return only the letter of the best option (A, B, or C).
        """
        
        vote_response = await self._query_model('math_specialist', vote_prompt)
        
        # Parse vote and return corresponding result
        if 'A' in vote_response.upper():
            return results[0]
        elif 'B' in vote_response.upper():
            return results[1] 
        elif len(results) > 2 and 'C' in vote_response.upper():
            return results[2]
        else:
            return results[0]  # Default to first option
```

#### **Ngày 18-21: Free Question Intelligence**
```python
# free_question_intelligence.py
class FreeQuestionIntelligence:
    def __init__(self):
        self.ai_server = LocalAIServer()
        self.multi_ai = FreeMultiAI()
    
    async def analyze_question_free(self, latex_content: str, image_path: str = None):
        """Comprehensive question analysis using free local AI"""
        
        analysis_prompt = f"""
        Analyze this mathematical question comprehensively:
        {latex_content}
        
        Provide analysis in this exact JSON format:
        {{
            "question_type": "multiple-choice/calculation/proof/etc",
            "subject_area": "algebra/geometry/calculus/statistics/etc", 
            "difficulty_level": 1-10,
            "key_concepts": ["concept1", "concept2"],
            "solution_approach": "brief description of solution method",
            "time_estimate_minutes": 1-60,
            "prerequisites": ["prerequisite1", "prerequisite2"],
            "complexity_score": 1-100,
            "notation_quality": 1-100
        }}
        
        Return only valid JSON, no additional text.
        """
        
        if image_path:
            # Use vision model for image context
            analysis = await self.multi_ai._query_model('vision_llm', analysis_prompt, image_path)
        else:
            # Use math specialist
            analysis = await self.multi_ai._query_model('math_specialist', analysis_prompt)
        
        try:
            import json
            return json.loads(analysis)
        except:
            # Fallback if JSON parsing fails
            return self._parse_analysis_fallback(analysis)
    
    def _parse_analysis_fallback(self, analysis_text: str):
        """Fallback parser if JSON parsing fails"""
        return {
            "question_type": "general",
            "subject_area": "mathematics", 
            "difficulty_level": 5,
            "key_concepts": ["mathematical expression"],
            "solution_approach": "standard mathematical methods",
            "time_estimate_minutes": 10,
            "prerequisites": ["basic mathematics"],
            "complexity_score": 50,
            "notation_quality": 70
        }
    
    async def generate_metadata_batch(self, processed_results: List[Dict]):
        """Generate metadata for batch of questions"""
        metadata = []
        
        for i, result in enumerate(processed_results):
            try:
                analysis = await self.analyze_question_free(
                    result['enhanced_latex'], 
                    result['image_path']
                )
                
                metadata.append({
                    'question_id': i + 1,
                    'filename': os.path.basename(result['image_path']),
                    'analysis': analysis,
                    'processing_quality': self._assess_processing_quality(result),
                    'confidence_score': self._calculate_confidence(result)
                })
                
            except Exception as e:
                metadata.append({
                    'question_id': i + 1,
                    'filename': os.path.basename(result['image_path']),
                    'analysis': None,
                    'error': str(e),
                    'processing_quality': 'unknown',
                    'confidence_score': 0
                })
        
        return metadata
    
    def _assess_processing_quality(self, result: Dict) -> str:
        """Assess quality of OCR processing"""
        latex = result.get('enhanced_latex', '')
        
        if 'Error' in latex:
            return 'poor'
        elif len(latex) < 10:
            return 'low'
        elif any(char in latex for char in ['\\', '{', '}', '^', '_']):
            return 'good'
        else:
            return 'fair'
    
    def _calculate_confidence(self, result: Dict) -> int:
        """Calculate confidence score 0-100"""
        latex = result.get('enhanced_latex', '')
        
        if 'Error' in latex:
            return 0
        
        score = 50  # Base score
        
        # Bonus for LaTeX syntax
        if '\\' in latex: score += 20
        if any(char in latex for char in ['{', '}']): score += 10
        if any(char in latex for char in ['^', '_']): score += 10
        
        # Penalty for very short results
        if len(latex) < 5: score -= 30
        
        return max(0, min(100, score))
```

### **TUẦN 4: Optimization & Advanced Features**

#### **Ngày 22-24: Performance Optimization**
```python
# free_performance_optimizer.py
import psutil
import time
from typing import Dict, List
import asyncio

class FreePerformanceOptimizer:
    def __init__(self):
        self.system_monitor = SystemMonitor()
        
    def optimize_batch_size(self, total_images: int) -> int:
        """Determine optimal batch size based on system resources"""
        available_ram = psutil.virtual_memory().available / (1024**3)  # GB
        cpu_count = psutil.cpu_count()
        
        if available_ram > 8:
            return min(10, total_images)  # Large batches for high RAM
        elif available_ram > 4:
            return min(5, total_images)   # Medium batches
        else:
            return min(2, total_images)   # Small batches for low RAM
    
    async def adaptive_processing(self, image_paths: List[str]):
        """Adaptive processing based on system resources"""
        total_images = len(image_paths)
        optimal_batch_size = self.optimize_batch_size(total_images)
        
        print(f"🔧 Processing {total_images} images in batches of {optimal_batch_size}")
        
        all_results = []
        batch_processor = FreeBatchProcessor()
        
        # Process in optimized batches
        for i in range(0, total_images, optimal_batch_size):
            batch = image_paths[i:i + optimal_batch_size]
            batch_num = i // optimal_batch_size + 1
            total_batches = (total_images + optimal_batch_size - 1) // optimal_batch_size
            
            print(f"📦 Processing batch {batch_num}/{total_batches}")
            
            # Monitor system before processing
            self.system_monitor.log_system_status()
            
            # Process batch
            start_time = time.time()
            batch_results = await batch_processor.process_batch_free(batch)
            end_time = time.time()
            
            # Log performance
            batch_time = end_time - start_time
            avg_time_per_image = batch_time / len(batch)
            
            print(f"⏱️  Batch {batch_num} completed in {batch_time:.1f}s ({avg_time_per_image:.1f}s per image)")
            
            all_results.extend(batch_results)
            
            # Cool down between batches if system is stressed
            if self.system_monitor.is_system_stressed():
                print("🌡️  System cooling down...")
                await asyncio.sleep(5)
        
        return all_results

class SystemMonitor:
    def log_system_status(self):
        """Log current system resource usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        print(f"💻 System Status - CPU: {cpu_percent}%, RAM: {memory.percent}% used")
    
    def is_system_stressed(self) -> bool:
        """Check if system is under high load"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        return cpu_percent > 80 or memory_percent > 85
```

#### **Ngày 25-28: Error Recovery & Caching**
```python
# free_error_recovery.py
import hashlib
import pickle
import os
from typing import Optional, Dict, Any

class FreeErrorRecovery:
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def get_image_hash(self, image_path: str) -> str:
        """Generate hash for image to use as cache key"""
        with open(image_path, 'rb') as f:
            content = f.