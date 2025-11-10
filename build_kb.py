# build_kb.py: Script to build/update the curated knowledge base from PDFs

import os
import json
import fitz  # PyMuPDF for PDF parsing
from openai import OpenAI
from typing import List, Dict
import re  # For pattern matching in detection
import base64  # For image encoding

from dotenv import load_dotenv
load_dotenv()

# Configuration - adjust as needed
# Update OPENAI_MODEL for vision
OPENAI_MODEL = "gpt-4o-mini"  # Supports vision; use this for image descriptions
PDF_DIR = "documents"  # Directory containing model subfolders
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
    
    # Walk through PDF directory (now assumes subfolders for models)
    for root, _, files in os.walk(pdf_dir):
        if root == pdf_dir:
            continue  # Skip top-level directory; only process model subfolders

        model_name = os.path.basename(root)
        print(f"Processing model folder: {model_name}")

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
            
            num_pages = len(pdf_text)
            
            # NEW: Detect doc type with improved matching
            is_rescue_sheet = bool(re.search(r'rescue\s*sheet', re.sub(r'[-_]', ' ', pdf_filename.lower())))
            doc_type = "rescue sheet" if is_rescue_sheet else "emergency response guide"
            print(f"Detected doc type: {doc_type} ({num_pages} pages)")
            
            pdf_images = {}  # No per-image extraction; handle via whole pages for rescue sheets if applicable
            
            # Only generate structured entries if model is not already in the KB
            if model_name not in kb or not kb[model_name]["intents"]:
                entries = generate_kb_entries(client, model_name, pdf_text, pdf_filename, pdf_images, doc_type, pdf_path)
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

def generate_kb_entries(client: OpenAI, model_name: str, pdf_text: Dict[int, str], pdf_filename: str, pdf_images: Dict[int, List[Dict]], doc_type: str, pdf_path: str) -> Dict[str, Dict]:
    """Uses OpenAI to generate Q&A entries from PDF text."""
    entries = {}
    
    # Prompt to generate intents from the entire PDF
    system_prompt = "You are an expert at extracting structured knowledge from EV emergency guides for first responders."
    
    if doc_type == "rescue sheet":
        # Generate base64 images for whole pages
        page_images_content = []
        page_sizes = []
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            width = page.rect.width
            height = page.rect.height
            page_sizes.append(f"Page {page_num+1}: {width:.0f}x{height:.0f} pt (0,0 at top-left)")
            
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("png")
            b64 = base64.b64encode(img_bytes).decode('utf-8')
            page_images_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })
        doc.close() 
        
        user_prompt = (
            f"Analyze the provided page images from the {model_name} rescue sheet (file: {pdf_filename}). "
            "The images are the full pages of the document. Extract text and visual information from them. "
            f"Page dimensions (in points, 72 pt per inch) for reference when suggesting bounding boxes: {'; '.join(page_sizes)}. "
            "Generate 5-15 key intents (e.g., 'disable_high_voltage', 'extricate_trapped_person'). Ensure to cover all relevant safety information for first responders. "
            "*Every intent should be a specific answer based on the topic from a SINGLE section. Do not generalize the answer for multiple sections. IFF the one topic spans multiple sections, THEN include all the sections in the highlights.*"
            "For each intent, provide:\n"
            "- patterns: 5-10 natural language query variations (e.g., 'how to cut power in Model 3'). Include variations that are logical questions a first responder might ask, commands/instructions, and questions a civilian might ask.\n"
            "- answer: A brief, step-by-step response informed ONLY from the page images. Ensure all information is relevant, actionable, and encompasses all necessary details. Include safety considerations and brief explanations. Reference visuals if important.\n"
            "- pages: List of relevant page numbers (integers) ordered by most important first.\n"
            "- pdf: The PDF filename (e.g., 'Cybertruck-Rescue-Sheet.pdf')\n"
            "- images: list of {'page': int, 'desc': str (brief description of relevant area), 'highlight': {'x': float, 'y': float, 'w': float, 'h': float} (suggested bounding box for highlight on the page **ALWAYS MAKE THE BOXES LARGER AND MORE ENCOMPASSING THAN NEEDED TO ENSURE ALL INFORMATIONS IS COVERED**, in page coordinates with 0,0 at top-left and using the provided page dimensions)} ordered by most important first.\n"
            "When suggesting bounding boxes, ensure they are precise: use floats for coordinates, validate that x + w <= page width and y + h <= page height, and encompass the entire relevant diagram/text section without including unrelated areas. Example: For a diagram in the bottom-right of a 612x792 page, you might suggest {'x': 400.0, 'y': 500.0, 'w': 200.0, 'h': 250.0} to cover it fully with some padding.\n"
            "Output the result strictly as a valid JSON object using this structure: {intent_name: {\"patterns\": [string], \"answer\": string, \"pages\": [int], \"pdf\": string, \"images\": [object]}}. Only output the JSON; do not include any explanation, markdown formatting, or extra text.\n"
            f"\n\nNote: This is a {doc_type}. Rescue sheets are highly visual, so emphasize diagrams and suggest highlights for key areas."
        )
        
        user_message = {"role": "user", "content": [{"type": "text", "text": user_prompt}] + page_images_content}
    else:
        user_prompt = (
            f"Analyze the following PDF text from the {model_name} emergency guide (file: {pdf_filename})."
            "Generate 15-30 key intents (e.g., 'disable_high_voltage', 'extricate_trapped_person'). Ensure to cover all relevant information. Create AT LEAST one intent for most important safety information."
            "*Every intent should be a specific answer based on the topic from a SINGLE page. Do not generalize the answer for multiple pages. IFF the one topic spans multiple pages, THEN include all the pages in the answer.*"
            "For each intent, provide:\n"
            "- patterns: 10-15 natural language query variations (e.g., 'how to cut power in Model 3'). Include variations that are logical questions a first responder might ask, commands/instructions, and questions a civilian might ask.\n"
            "- answer: A brief, step-by-step response informed ONLY from the text. Ensure all information is relevant, actionable, and encompasses all the necessary information. Ensure relevant safety considerations are included too. Also include a brief explanation for why.\n"
            "- pages: List of relevant page numbers (integers) ordered by most important first.\n"
            "- pdf: The PDF filename (e.g., '2024-Model-3-Emergency-Response-Guide_en.pdf')\n"
            "Output the result strictly as a valid JSON object using this structure: {intent_name: {\"patterns\": [string], \"answer\": string, \"pages\": [int], \"pdf\": string}}. Only output the JSON; do not include any explanation, markdown formatting, or extra text.\n"
            f"\n\nPDF Text (page: text):\n{json.dumps(pdf_text, ensure_ascii=False)}"
            f"\n\nNote: This is a {doc_type}."
        )
        
        user_message = {"role": "user", "content": user_prompt}
    
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            user_message
        ],
        temperature=0.2,
        max_tokens=6000,
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

if __name__ == "__main__":
    build_knowledge_base()
