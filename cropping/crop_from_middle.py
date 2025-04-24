import json
import os
import fitz  # PyMuPDF


def compute_scale_factors(reference_pdf: str, target_pdf: str):
    """
    Compute width and height scale factors by comparing page‑0 dimensions
    of a reference PDF against the target PDF.
    """
    if not (os.path.exists(reference_pdf) and os.path.exists(target_pdf)):
        return 1.0, 1.0

    doc_ref = fitz.open(reference_pdf)
    doc_tgt = fitz.open(target_pdf)
    w_ref, h_ref = doc_ref[0].rect.width, doc_ref[0].rect.height
    w_tgt, h_tgt = doc_tgt[0].rect.width, doc_tgt[0].rect.height
    doc_ref.close()
    doc_tgt.close()
    return w_tgt / w_ref, h_tgt / h_ref


def extract_text_blocks_from_middle_json(middle_json_path):
    """
    Extract text blocks from the middle JSON file.
    
    Args:
        middle_json_path (str): Path to the middle JSON file
        
    Returns:
        list: List of dictionaries with page_idx, bbox, and concatenated content
    """
    with open(middle_json_path, 'r', encoding='utf-8') as f:
        middle = json.load(f)
    
    text_blocks = []
    processed_blocks = set()  # To track already processed blocks and avoid duplicates
    
    # Process each page
    for page in middle.get("pdf_info", []):
        page_idx = page["page_idx"]
        
        # Extract text blocks from para_blocks
        for block in page.get("para_blocks", []):
            # Skip non-text blocks (but include images now)
            if block.get("type") not in ["text", "title", "list", "image"]:
                continue
            
            # Get the overall block bbox
            bbox = block.get("bbox", [])
            if not bbox or len(bbox) != 4:
                continue
            
            # Create a unique identifier for this block based on page and position
            block_key = f"{page_idx}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
            
            # Skip if we've already processed this block
            if block_key in processed_blocks:
                continue
            
            # Mark this block as processed
            processed_blocks.add(block_key)
            
            # Handle image blocks specially
            if block.get("type") == "image":
                text_blocks.append({
                    "page_idx": page_idx,
                    "bbox": bbox,
                    "content": "Image",  # Placeholder content for images
                    "type": "image",
                    "image_path": block.get("image_path", "")  # Add image path if available
                })
                continue
            
            # Concatenate content from all lines in the block
            concatenated_content = ""
            for line in block.get("lines", []):
                line_content = ""
                for span in line.get("spans", []):
                    if span.get("content"):
                        line_content += span.get("content", "")
                if line_content:
                    if concatenated_content:
                        concatenated_content += " "
                    concatenated_content += line_content
            
            # Add block info to the list
            text_blocks.append({
                "page_idx": page_idx,
                "bbox": bbox,
                "content": concatenated_content,
                "type": block.get("type", "text")
            })
        
        # Also extract from preproc_blocks if available
        for block in page.get("preproc_blocks", []):
            if block.get("type") not in ["text", "title", "list", "image"]:
                continue
            
            bbox = block.get("bbox", [])
            if not bbox or len(bbox) != 4:
                continue
            
            # Create a unique identifier for this block
            block_key = f"{page_idx}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
            
            # Skip if we've already processed this block
            if block_key in processed_blocks:
                continue
            
            # Mark as processed
            processed_blocks.add(block_key)
            
            # Handle image blocks specially
            if block.get("type") == "image":
                text_blocks.append({
                    "page_idx": page_idx,
                    "bbox": bbox,
                    "content": "Image",  # Placeholder content for images
                    "type": "image",
                    "image_path": block.get("image_path", "")  # Add image path if available
                })
                continue
            
            concatenated_content = ""
            for line in block.get("lines", []):
                line_content = ""
                for span in line.get("spans", []):
                    if span.get("content"):
                        line_content += span.get("content", "")
                if line_content:
                    if concatenated_content:
                        concatenated_content += " "
                    concatenated_content += line_content
            
            text_blocks.append({
                "page_idx": page_idx,
                "bbox": bbox,
                "content": concatenated_content,
                "type": block.get("type", "text")
            })
        
        # Extract images directly
        for image in page.get("images", []):
            bbox = image.get("bbox", [])
            if not bbox or len(bbox) != 4:
                continue
                
            # Create a unique identifier for this image
            block_key = f"{page_idx}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
            
            # Skip if we've already processed this image
            if block_key in processed_blocks:
                continue
                
            # Mark as processed
            processed_blocks.add(block_key)
            
            # Add image info to the list
            text_blocks.append({
                "page_idx": page_idx,
                "bbox": bbox,
                "content": "Image",
                "type": "image",
                "image_path": image.get("image_path", "")
            })
    
    print(f"Found {len(text_blocks)} unique text blocks & images across the document")
    return text_blocks


def extract_interline_equations(middle_json_path):
    """
    Extract interline equations from the middle JSON file.
    
    Args:
        middle_json_path (str): Path to the middle JSON file
        
    Returns:
        list: List of dictionaries with page_idx, bbox, and content for interline equations
    """
    with open(middle_json_path, 'r', encoding='utf-8') as f:
        middle = json.load(f)
    
    interline_equations = []
    processed_equations = set()  # To track already processed equations
    
    # Process each page
    for page in middle.get("pdf_info", []):
        page_idx = page["page_idx"]
        
        # Extract interline equations
        for eq in page.get("interline_equations", []):
            bbox = eq.get("bbox", [])
            if not (bbox and len(bbox) == 4):
                continue
                
            # Create a unique identifier for this equation
            eq_key = f"{page_idx}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
            
            # Skip if we've already processed this equation
            if eq_key in processed_equations:
                continue
                
            # Mark as processed
            processed_equations.add(eq_key)
            
            content = ""
            for line in eq.get("lines", []):
                for span in line.get("spans", []):
                    if span.get("content") and span.get("type") == "interline_equation":
                        content = span.get("content", "")
            
            interline_equations.append({
                "page_idx": page_idx,
                "bbox": bbox,
                "content": content,
                "type": "interline_equation"
            })
    
    print(f"Found {len(interline_equations)} unique interline equations across the document")
    return interline_equations


def extract_equation_items(middle_json_path):
    """
    Extract inline equation items from the middle JSON file.
    
    Args:
        middle_json_path (str): Path to the middle JSON file with layout information
        
    Returns:
        list: List of dictionaries with page_idx, bbox, and content for inline equations
    """
    with open(middle_json_path, 'r', encoding='utf-8') as f:
        middle = json.load(f)
    
    equations = []
    processed_equations = set()  # To track already processed equations and avoid duplicates
    
    # Process each page
    for page in middle.get("pdf_info", []):
        page_idx = page["page_idx"]
        
        # Extract inline equations from blocks
        for block_type in ["preproc_blocks", "para_blocks"]:
            for block in page.get(block_type, []):
                # Check if the block has equation attribute
                if block.get("attributes", {}).get("equation"):
                    bbox = block.get("bbox", [])
                    if not (bbox and len(bbox) == 4):
                        continue
                        
                    # Create a unique identifier for this equation
                    eq_key = f"{page_idx}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
                    
                    # Skip if we've already processed this equation
                    if eq_key in processed_equations:
                        continue
                        
                    # Mark as processed
                    processed_equations.add(eq_key)
                    
                    equations.append({
                        "page_idx": page_idx,
                        "bbox": bbox,
                        "content": block.get("text", ""),
                        "type": "equation"
                    })
                
                # Process each line in the block
                for line in block.get("lines", []):
                    # Process each span in the line
                    for span in line.get("spans", []):
                        # Check if the span is an inline equation
                        if span.get("type") == "inline_equation":
                            bbox = span.get("bbox", [])
                            if not (bbox and len(bbox) == 4):
                                continue
                                
                            # Create a unique identifier for this equation
                            eq_key = f"{page_idx}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
                            
                            # Skip if we've already processed this equation
                            if eq_key in processed_equations:
                                continue
                                
                            # Mark as processed
                            processed_equations.add(eq_key)
                            
                            equations.append({
                                "page_idx": page_idx,
                                "bbox": bbox,
                                "content": span.get("content", ""),
                                "type": "inline_equation"
                            })
    
    print(f"Found {len(equations)} unique inline equations across the document")
    return equations


def get_rect_overlap_percentage(r1: fitz.Rect, r2: fitz.Rect) -> float:
    """
    Calculate the percentage of overlap between two rectangles.
    """
    if not r1.intersects(r2):
        return 0.0
    inter = r1 & r2
    area_int = inter.width * inter.height
    smaller = min(r1.width * r1.height, r2.width * r2.height)
    return (area_int / smaller) if smaller > 0 else 0.0


def crop_pdf_content(
    pdf_path: str,
    middle_json_path: str,
    output_dir: str,
    reference_pdf: str = None,
    desired_dpi: int = 300
):
    """
    Crops PDF content based on text blocks and equations from the middle JSON file.
    
    Args:
        pdf_path (str): Path to the PDF file
        middle_json_path (str): Path to the middle JSON file with layout information
        output_dir (str): Directory to save the cropped content
        reference_pdf (str, optional): Reference PDF for scaling. Defaults to None.
        desired_dpi (int, optional): Desired DPI for rendering. Defaults to 300.
    """
    os.makedirs(output_dir, exist_ok=True)
    text_blocks_dir = os.path.join(output_dir, "text_blocks")
    os.makedirs(text_blocks_dir, exist_ok=True)
    eq_dir = os.path.join(output_dir, "equations")
    os.makedirs(eq_dir, exist_ok=True)

    # 1) scale factors
    scale_w, scale_h = (1.0, 1.0)
    if reference_pdf:
        scale_w, scale_h = compute_scale_factors(reference_pdf, pdf_path)

    # 2) high‑res matrix
    zoom = desired_dpi / 72.0
    mat = fitz.Matrix(zoom * scale_w, zoom * scale_h)

    # 3) load content
    text_blocks = extract_text_blocks_from_middle_json(middle_json_path)
    inline_equations = extract_equation_items(middle_json_path)
    interline_equations = extract_interline_equations(middle_json_path)
    doc = fitz.open(pdf_path)

    # Create a combined list for items that will go in text_blocks_dir
    # We'll maintain the order: first text blocks, then images, then interline equations
    combined_items = []
    
    # Add text blocks (excluding images which we'll handle separately)
    for block in text_blocks:
        if block.get('type') != 'image':
            combined_items.append({
                'item': block,
                'type': 'text',
                'page_idx': block.get('page_idx', -1),
                'bbox': block.get('bbox', [])
            })
    
    # Add image blocks
    for block in text_blocks:
        if block.get('type') == 'image':
            combined_items.append({
                'item': block,
                'type': 'image',
                'page_idx': block.get('page_idx', -1),
                'bbox': block.get('bbox', [])
            })

    # Add interline equations
    for eq in interline_equations:
        combined_items.append({
            'item': eq,
            'type': 'interline_equation',
            'page_idx': eq.get('page_idx', -1),
            'bbox': eq.get('bbox', [])
        })
    
    # Sort by page and then by vertical position (top to bottom)
    combined_items.sort(key=lambda x: (x['page_idx'], x['bbox'][1] if len(x['bbox']) >= 2 else 0))
    
    # --- Process combined items (text, images, interline equations) ---
    for i, combined_item in enumerate(combined_items):
        item = combined_item['item']
        item_type = combined_item['type']
        page_idx = item.get('page_idx', -1)
        bbox = item.get('bbox', [])
        
        if not (0 <= page_idx < doc.page_count) or len(bbox) != 4:
            continue
        
        x0, y0, x1, y1 = bbox
        clip = fitz.Rect(x0 * scale_w, y0 * scale_h, x1 * scale_w, y1 * scale_h)
        
        # For images, try to extract directly first
        if item_type == 'image':
            found = False
            page = doc[page_idx]
            
            # Skip direct extraction to avoid errors - just render the region
            try:
                pix = page.get_pixmap(clip=clip, matrix=mat)
                out = os.path.join(text_blocks_dir, f"page{page_idx}_item{i}.png")
                pix.save(out)
                
                # Store the path
                item['cropped_image_path'] = out
                item['dpi'] = desired_dpi
            except Exception as e:
                print(f"Error processing image at page {page_idx}, item {i}: {str(e)}")
        
        # For interline equations and text blocks, render the region
        else:
            try:
                pix = doc[page_idx].get_pixmap(matrix=mat, clip=clip, alpha=False)
                out = os.path.join(text_blocks_dir, f"page{page_idx}_item{i}.png")
                pix.save(out)
                
                # Store the path
                item['cropped_image_path'] = out
                item['dpi'] = desired_dpi
            except Exception as e:
                print(f"Error processing item at page {page_idx}, item {i}: {str(e)}")

    # --- Process inline equations separately ---
    for i, eq in enumerate(inline_equations):
        page_idx = eq.get('page_idx', -1)
        bbox = eq.get('bbox', [])
        if not (0 <= page_idx < doc.page_count) or len(bbox) != 4:
            continue
        
        x0, y0, x1, y1 = bbox
        clip = fitz.Rect(x0 * scale_w, y0 * scale_h, x1 * scale_w, y1 * scale_h)
        
        # Render equation with high DPI
        try:
            pix = doc[page_idx].get_pixmap(matrix=mat, clip=clip, alpha=False)
            out = os.path.join(eq_dir, f"page{page_idx}_eq{i}.png")
            pix.save(out)
            
            # Save the path in the equation info
            eq['cropped_image_path'] = out
            eq['dpi'] = desired_dpi
        except Exception as e:
            print(f"Error processing inline equation at page {page_idx}, eq {i}: {str(e)}")

    # Prepare data for JSON output
    # Extract text blocks, images and interline equations from combined_items
    filtered_text_blocks = []
    images = []
    filtered_interline_eqs = []
    
    for combined_item in combined_items:
        item = combined_item['item']
        if combined_item['type'] == 'text':
            filtered_text_blocks.append(item)
        elif combined_item['type'] == 'image':
            images.append(item)
        elif combined_item['type'] == 'interline_equation':
            filtered_interline_eqs.append(item)

    # Write JSON outputs
    text_blocks_json = os.path.join(text_blocks_dir, 'text_blocks.json')
    with open(text_blocks_json, 'w', encoding='utf-8') as f:
        # Include text items and interline equations
        combined_text_content = filtered_text_blocks + filtered_interline_eqs
        json.dump(combined_text_content, f, indent=2, ensure_ascii=False)

    inline_eq_json = os.path.join(eq_dir, 'inline_equations.json')
    with open(inline_eq_json, 'w', encoding='utf-8') as f:
        json.dump(inline_equations, f, indent=2, ensure_ascii=False)
    
    images_json = os.path.join(text_blocks_dir, 'images.json')
    with open(images_json, 'w', encoding='utf-8') as f:
        json.dump(images, f, indent=2, ensure_ascii=False)
    
    # Create a combined output with ALL content - in the same order as combined_items
    all_content = []
    for combined_item in combined_items:
        all_content.append(combined_item['item'])
    
    
    all_content_json = os.path.join(output_dir, 'all_content.json')
    with open(all_content_json, 'w', encoding='utf-8') as f:
        json.dump(all_content, f, indent=2, ensure_ascii=False)

    print(f"✅ Cropping complete:")
    print(f"   - {len(filtered_text_blocks)} text blocks in text_blocks_dir")
    print(f"   - {len(images)} images in text_blocks_dir")
    print(f"   - {len(filtered_interline_eqs)} interline equations in text_blocks_dir")
    print(f"   - {len(inline_equations)} inline equations in equations_dir")
    print(f"   - Total {len(all_content)} items in all_content.json")


if __name__ == "__main__":
    # Example usage
    middle_json_path = "output/hatier_test_page_6/auto/hatier_test_middle.json"
    pdf_path = "output/hatier_test_page_6/auto/hatier_test_origin.pdf"
    output_dir = "output/document/hatier_test_page_6/content"
    reference_pdf = None  # Set this if scaling is needed
    
    # Process all content
    crop_pdf_content(
        pdf_path, 
        middle_json_path, 
        output_dir,
        reference_pdf=reference_pdf,
        desired_dpi=300
    )