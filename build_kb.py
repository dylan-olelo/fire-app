# build_kb.py: Script to build/update the curated knowledge base from PDFs

import os
import json
import fitz  # PyMuPDF for PDF parsing
from openai import OpenAI
from typing import List, Dict
import re  # For potential simple pattern matching as fallback

from dotenv import load_dotenv
load_dotenv()

# Add after imports
from PIL import Image
import pytesseract
import io  # For image bytes

# Optionally set Tesseract path (uncomment if needed)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows example

# Configuration - adjust as needed
# Update OPENAI_MODEL for vision
OPENAI_MODEL = "gpt-5-nano"  # Supports vision; use this for image descriptions
PDF_DIR = "documents_testing"  # Directory containing PDF subfolders
OUTPUT_PATH = "kb/curated_kb.json"  # Where to save the KB

def build_knowledge_base(pdf_dir: str = PDF_DIR, output_path: str = OUTPUT_PATH, api_key: str = None) -> None:
    """
    Builds or updates a structured knowledge base from PDF documents.
    
    - Parses each PDF to extract text and metadata.
    - Uses OpenAI to generate Q&A entries (intents, patterns, answers, page refs).
    - Saves/updates a JSON file with the structure: {model: {intents: {intent_name: {patterns: [], answer: str, pages: [], pdf: str}}}}
    
    If the output file exists, it merges new entries without overwriting existing ones.
    """
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    client = OpenAI(api_key=api_key)
    
    # Load existing KB if it exists
    kb = {}
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            kb = json.load(f)
        print(f"Loaded existing KB with {len(kb)} models.")
    
    # Walk through PDF directory (now assumes flat structure in pdf_dir)
    for root, _, files in os.walk(pdf_dir):
        for file in files:
            if not file.lower().endswith(".pdf"):
                continue
            pdf_path = os.path.join(root, file)
            pdf_filename = os.path.basename(file)
            
            print(f"Processing PDF: {pdf_path} (filename: {pdf_filename})")
            
            # Extract text from PDF
            doc = fitz.open(pdf_path)
            pdf_text = {}
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pdf_text[page_num + 1] = page.get_text("text")  # 1-based page nums
            
            doc.close()

            # NEW: Extract model name from PDF text
            model_name = extract_model_name(client, pdf_text.get(1,""), pdf_filename)
            print(f"Extracted model: {model_name}")
            
            # NEW: Extract images with bounding boxes and OCR text
            doc = fitz.open(pdf_path)  # Reopen for images
            num_pages = len(doc)
            
            # NEW: Detect doc type
            is_rescue_sheet = "rescue-sheet" in pdf_filename.lower() or num_pages < 10
            doc_type = "rescue sheet" if is_rescue_sheet else "emergency response guide"
            print(f"Detected doc type: {doc_type} ({num_pages} pages)")
            
            pdf_images = {}
            for page_num in range(num_pages):
                page = doc.load_page(page_num)
                images = page.get_images(full=True)
                page_images = []
                for img_index, img in enumerate(images):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    img_width, img_height = base_image["width"], base_image["height"]
                    
                    # NEW: Filter unimportant images
                    if img_width * img_height < 200 * 200:  # Skip small icons/logos
                        print(f"Skipped small image ({img_width}x{img_height}) on page {page_num+1}, img {img_index}")
                        continue
                    
                    # Get bounding box (rect)
                    rect = page.get_image_rects(xref)[0] if page.get_image_rects(xref) else None
                    if rect:
                        bbox = {"x": rect.x0, "y": rect.y0, "w": rect.width, "h": rect.height}
                    else:
                        bbox = None
                    
                    # # OCR only for non-sheets
                    # ocr_text = ""
                    # if not is_rescue_sheet:
                    #     try:
                    #         pil_image = Image.open(io.BytesIO(image_bytes))
                    #         ocr_text = pytesseract.image_to_string(pil_image, lang='eng', config='--psm 6').strip()  # eng lang, PSM 6 for multi-line
                    #         if not ocr_text:
                    #             print(f"No text detected in image on page {page_num+1}, image {img_index}")
                    #         else:
                    #             print(f"Extracted {len(ocr_text)} chars from image on page {page_num+1}, image {img_index}: '{ocr_text[:50]}...'")  # Truncated preview
                    #     except Exception as e:
                    #         print(f"OCR failed for image on page {page_num+1}: {e}")
                    #         ocr_text = ""

                    # if not ocr_text:
                    #     print(f"No OCR text for {img_path} - AI will infer description from context.")

                    # Generate description using GPT-4V
                    desc = ""
                    try:
                        # Prepare vision prompt
                        vision_prompt = (
                            f"Briefly describe this {model_name} EV guide image for first responders."
                            f"{'This is a breif rescue sheet, heavy on visuals and diagrams, so focus on what is important.' if is_rescue_sheet else 'Incorporate text if present.'} Suggest a highlight box if relevant."
                            f"\nPage Context: {pdf_text.get(page_num + 1, '')[:200]}..." 
                            "\n Output only the description. IF image is extraneaous or irrelevant (ex. a placard or symbol), output 'SKIP'" # Truncated context
                        )
                        # Base64 encode image for API
                        import base64
                        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                        
                        response = client.chat.completions.create(
                            model=OPENAI_MODEL,
                            messages=[
                                {"role": "user", "content": [
                                    {"type": "text", "text": vision_prompt},
                                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                                ]}
                            ],
                            max_completion_tokens=100,
                        )
                        desc = response.choices[0].message.content.strip()
                        print(f"Generated desc for page {page_num+1}, img {img_index}: {desc[:50]}...")
                    except Exception as e:
                        print(f"Vision desc failed for page {page_num+1}, img {img_index}: {e}")
                        # desc = f"Inferred diagram from context: {ocr_text[:50]}..." if ocr_text else "Visual diagram (no text detected)"

                    if desc != "SKIP":
                        # In build_knowledge_base, after pdf_images extraction, ensure dir for saved images
                        # Create dir for extracted images
                        extract_dir = f"images/extracted/{pdf_filename}"
                        os.makedirs(extract_dir, exist_ok=True)

                        # Save image for review
                        img_path = f"{extract_dir}/page_{page_num+1}_img_{img_index}.png"
                        with open(img_path, "wb") as f:
                            f.write(image_bytes)
                        print(f"Saved image for review: {img_path}")
                    else:
                        desc = ""

                    page_images.append({
                        "index": img_index,
                        "bbox": bbox,
                        "desc": desc  # NEW: AI-generated description
                    })
                pdf_images[page_num + 1] = page_images
            
            doc.close()
            
            
            # Only generate structured entries if model is not already in the KB
            if model_name not in kb or not kb[model_name]["intents"]:
                entries = generate_kb_entries(client, model_name, pdf_text, pdf_filename, pdf_images, doc_type)
            else:
                print(f"Skipping {model_name}: already present in KB with {len(kb[model_name]['intents'])} intents.")
                entries = {}
            
            # Merge into KB using extracted model_name
            if model_name not in kb:
                kb[model_name] = {"intents": {}}
            for intent_name, entry in entries.items():
                if intent_name not in kb[model_name]["intents"]:
                    kb[model_name]["intents"][intent_name] = entry
                else:
                    # Simple merge: append new patterns, keep existing answer/pages
                    existing = kb[model_name]["intents"][intent_name]
                    existing["patterns"].extend([p for p in entry["patterns"] if p not in existing["patterns"]])
                    print(f"Updated existing intent '{intent_name}' for {model_name}")
    
    # Save updated KB
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=4, ensure_ascii=False)
    print(f"Knowledge base saved/updated at {output_path} with {len(kb)} models.")

def generate_kb_entries(client: OpenAI, model_name: str, pdf_text: Dict[int, str], pdf_filename: str, pdf_images: Dict[int, List[Dict]], doc_type: str) -> Dict[str, Dict]:
    """Uses OpenAI to generate Q&A entries from PDF text."""
    entries = {}
    
    # Prompt to generate intents from the entire PDF
    system_prompt = "You are an expert at extracting structured knowledge from EV emergency guides for first responders."
    user_prompt = (
        f"Analyze the following PDF text and image data from the {model_name} emergency guide (file: {pdf_filename})."
        "Generate 15-30 key intents (e.g., 'disable_high_voltage', 'extricate_trapped_person'). Ensure to cover all relevant information. Create one intent for most important safety information."
        "For each intent, provide:\n"
        "- patterns: 10-15 natural language query variations (e.g., 'how to cut power in Model 3'). Include variations that are logical questions a first responder might ask.\n"
        "- answer: A brief, step-by-step response informed ONLY from the text and images. Ensure all information is relevant, actionable, and encompassess all the necessary information. Ensure safety considerations are included too. Also include a brief explanation for why. If images are important to the answer, reference them in the answer.\n"
        "- pages: List of relevant page numbers (integers) ordered by most important first.\n"
        "- pdf: The PDF filename (e.g., '2024-Model-3-Emergency-Response-Guide_en.pdf')\n"
        "- images: list of {'page': int, 'desc': str (brief description of relevant image), 'highlight': {'x': float, 'y': float, 'w': float, 'h': float} (suggested bounding box for highlight, based on provided image bboxes)} ordered by most important first."
        "Use the provided image bboxes and generated descriptions to inform relevant 'images' for each intent."
        "Output the result strictly as a valid JSON object using this structure: {intent_name: {\"patterns\": [string], \"answer\": string, \"pages\": [int], \"pdf\": string, \"images\": [object]}}. Only output the JSON; do not include any explanation, markdown formatting, or extra text.\n"
        f"\n\nPDF Images (page: [{{index: int, bbox: dict or null, desc: str}}]):\n{json.dumps(pdf_images, ensure_ascii=False)}"
        f"\n\nPDF Text (page: text):\n{json.dumps(pdf_text, ensure_ascii=False)}"
        f"\n\nNote: This is a {doc_type}. Rescue sheets are far more visual than emergency response guides, so emphasize visual elements in descriptions in rescue sheet entries. "
    )

    # "Output as JSON: {intent_name: {patterns: [str], answer: str, pages: [int], pdf: str}}"
    
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        # temperature=0.2,
        reasoning_effort='low',
        summary='null',
        max_completion_tokens=6000,
    )
    
    try:
        content = response.choices[0].message.content.strip()
        # NEW: Clean markdown if present
        if content.startswith("```json"):
            content = content.removeprefix("```json").strip()
        if content.endswith("```"):
            content = content.removesuffix("```").strip()
        
        print(f"Cleaned OpenAI response: {content[:200]}...")  # Truncated preview for debugging
        
        generated = json.loads(content)
        entries.update(generated)
    except json.JSONDecodeError as e:
        print(f"Error parsing OpenAI response: {e}")
        print("Raw response content:", response.choices[0].message.content)  # For debugging
    
    return entries

# NEW function to extract model name
def extract_model_name(client: OpenAI, pdf_text: Dict[int, str], pdf_filename: str) -> str:
    """Uses OpenAI to extract the vehicle model from PDF text. Falls back to filename patterns if needed."""
    # # First, try simple heuristic: look for common patterns in first few pages
    # intro_text = pdf_text
    # patterns = [
    #     r"Tesla Model [3YSX]",  # Common Tesla models
    #     r"Model [3YSX] Emergency Response Guide",
    #     # Add more for other EVs if needed, e.g., r"Ford Mustang Mach-E"
    # ]
    # for pat in patterns:
    #     match = re.search(pat, intro_text, re.IGNORECASE)
    #     if match:
    #         return match.group(0).strip()
    
    # If heuristic fails, use OpenAI
    system_prompt = "You are an expert at identifying model of vehicle or subject of EV emergency guide text."
    user_prompt = (
        f"Extract the exact vehicle make and model name (e.g., 'Tesla Model 3') from this PDF text. "
        "Look for titles or headers. If unclear, use the filename as a hint.\n\n Only return the vehicle make and model name in ALL CAPS, NO OTHER TEXT.\n\n"
        f"Filename: {pdf_filename}\n"
        f"PDF Text (first page): {json.dumps(pdf_text, ensure_ascii=False)}"
    )
    
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        # temperature=0,
        max_completion_tokens=50,
    )
    
    extracted = response.choices[0].message.content.strip()
    if not extracted:
        # Ultimate fallback: use cleaned filename
        extracted = re.sub(r"[_-]", " ", os.path.splitext(pdf_filename)[0]).title()
    
    return extracted

if __name__ == "__main__":
    build_knowledge_base()
