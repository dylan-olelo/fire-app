# test_highlights.py: Visualize KB highlights on PDF pages

import fitz  # PyMuPDF
import json

def visualize_highlights(kb_path: str, pdf_path: str, model_name: str, intent_name: str):
    # Load KB
    with open(kb_path, "r") as f:
        kb = json.load(f)
    
    if model_name not in kb or intent_name not in kb[model_name]["intents"]:
        print(f"Intent '{intent_name}' not found for '{model_name}'")
        return
    
    entry = kb[model_name]["intents"][intent_name]
    pdf_filename = entry.get("pdf", "")
    images = entry.get("images", [])  # List of {'page': int, 'desc': str, 'highlight': dict}
    
    if not images:
        print("No images/highlights in this entry.")
        return
    
    # Open PDF
    doc = fitz.open(pdf_path)
    
    for img in images:
        page_num = img.get("page", 1) - 1  # 0-based
        if page_num < 0 or page_num >= len(doc):
            print(f"Invalid page {page_num+1}")
            continue
        
        page = doc[page_num]
        highlight = img.get("highlight", {})
        if not highlight:
            continue
        
        # Draw red rectangle
        rect = fitz.Rect(highlight.get("x", 0), highlight.get("y", 0),
                         highlight.get("x", 0) + highlight.get("w", 0),
                         highlight.get("y", 0) + highlight.get("h", 0))
        page.draw_rect(rect, color=(1, 0, 0), width=2)  # Red border
        
        print(f"Drew highlight on page {page_num+1} for '{img.get('desc', '')}'")
    
    # Save annotated PDF
    output_pdf = f"preview_{pdf_filename}"
    doc.save(output_pdf)
    doc.close()
    print(f"Saved annotated PDF: {output_pdf}. Open to check highlights.")

# Example usage (adjust)
if __name__ == "__main__":
    visualize_highlights(
        kb_path="kb/curated_kb.json",
        pdf_path="documents/TESLA_CYBERTRUCK/Cybertruck-Rescue-Sheet.pdf",
        model_name="TESLA_CYBERTRUCK",
        intent_name="disable_high_voltage"
    )