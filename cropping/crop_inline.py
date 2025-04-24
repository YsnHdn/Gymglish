import json
import os
import fitz  # PyMuPDF

def extract_inline_equations(middle_json_path):
    """
    Extract all inline equation bounding boxes from the middle JSON file.
    
    Args:
        middle_json_path (str): Path to the middle JSON file with layout information
        
    Returns:
        list: List of dictionaries with page_idx, bbox, and content for each inline equation
    """
    # Load the middle JSON file
    with open(middle_json_path, 'r', encoding='utf-8') as f:
        middle_json = json.load(f)
    
    equations = []
    
    # Process each page
    for page in middle_json.get("pdf_info", []):
        page_idx = page["page_idx"]
        page_size = page.get("page_size", [0, 0])
        
        # Process all blocks (including preproc_blocks and para_blocks)
        for block_type in ["preproc_blocks", "para_blocks"]:
            for block in page.get(block_type, []):
                # Process each line in the block
                for line in block.get("lines", []):
                    # Process each span in the line
                    for span in line.get("spans", []):
                        # Check if the span is an inline equation
                        if span.get("type") == "inline_equation":
                            equations.append({
                                "page_idx": page_idx,
                                "bbox": span.get("bbox", []),
                                "content": span.get("content", ""),
                                "page_size": page_size
                            })
    
    return equations

def crop_inline_equations(pdf_path, middle_json_path, output_dir, dpi=300):
    """
    Crops inline equations from PDF pages based on bounding box information from the middle JSON file.
    
    Args:
        pdf_path (str): Path to the PDF file
        middle_json_path (str): Path to the middle JSON file with layout information
        output_dir (str): Directory to save the cropped equations
        dpi (int, optional): Desired DPI for rendering. Defaults to 300.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract inline equations with their bounding boxes
    equations = extract_inline_equations(middle_json_path)
    print(f"Found {len(equations)} inline equations across the document")
    
    # Open the PDF
    pdf_document = fitz.open(pdf_path)
    
    # Process each equation
    for i, eq in enumerate(equations):
        page_idx = eq.get('page_idx')
        bbox = eq.get('bbox')
        content = eq.get('content', '')
        
        if page_idx >= len(pdf_document) or not bbox or len(bbox) != 4:
            print(f"Warning: Invalid page index or bbox for equation {i}")
            continue
        
        page = pdf_document[page_idx]
        page_width, page_height = page.rect.width, page.rect.height
        
        # Extract bbox coordinates
        x0, y0, x1, y1 = bbox
        
        # Ensure coordinates are within page bounds
        x0 = max(0, min(x0, page_width))
        y0 = max(0, min(y0, page_height))
        x1 = max(0, min(x1, page_width))
        y1 = max(0, min(y1, page_height))
        
        # Create rect object for cropping
        clip_rect = fitz.Rect(x0, y0, x1, y1)
        
        # Calculate zoom factor based on desired DPI
        # PDF default is 72 DPI, so zoom is desired DPI / 72
        zoom = dpi / 72.0
        
        # Create transformation matrix
        mat = fitz.Matrix(zoom, zoom)
        
        # Render pixmap with specified DPI
        pix = page.get_pixmap(
            matrix=mat,      # DPI-controlled matrix
            clip=clip_rect,  # Original bbox
            alpha=False      # No transparency
        )
        
        # Create a filename that includes some of the equation content for easier reference
        # Clean the content to make a valid filename
        clean_content = ''.join(c if c.isalnum() else '_' for c in content[:20])
        output_path = os.path.join(output_dir, f"page{page_idx}_eq{i}_{clean_content}.png")
        
        # Save the cropped image
        pix.save(output_path)
        
        # Store the path and details for debugging
        eq['cropped_image_path'] = output_path
        eq['dpi'] = dpi
    
    # Save JSON with cropped equation paths
    equations_json_path = os.path.join(output_dir, 'inline_equations.json')
    with open(equations_json_path, 'w', encoding='utf-8') as f:
        json.dump(equations, f, indent=2, ensure_ascii=False)
    
    # Close the PDF
    pdf_document.close()
    
    print(f"✅ Completed cropping. Processed {len(equations)} equations at {dpi} DPI.")
    print(f"✅ Equations information saved to: {equations_json_path}")
    return equations

if __name__ == "__main__":
    # Example usage
    middle_json_path = "output/hatier_test_page_6/auto/hatier_test_middle.json"
    pdf_path = "output/hatier_test_page_6/auto/hatier_test_layout.pdf"
    output_dir = "output/document/hatier_test_page_6/equations"
    
    # Optional: Adjust DPI as needed
    crop_inline_equations(pdf_path, middle_json_path, output_dir, dpi=300)