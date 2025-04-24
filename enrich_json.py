import json
import os
import re
from difflib import SequenceMatcher
import math

def match_content_with_bboxes(content_list_path, middle_json_path, output_path):
    """
    Match content list items with their bounding boxes from the middle JSON
    and save the enriched content list.
    """
    print("Starting content-to-layout matching process...")
    
    # Load files
    with open(content_list_path, "r", encoding="utf-8") as f:
        content_list = json.load(f)
    
    with open(middle_json_path, "r", encoding="utf-8") as f:
        middle_json = json.load(f)
    
    print(f"Loaded content list: {len(content_list)} items")
    
    # Preprocess the middle JSON to extract layout information by page
    layout_by_page = preprocess_middle_json(middle_json)
    
    # Extract all content types present in the content list
    content_types = set(item.get("type", "") for item in content_list)
    print(f"Content types in list: {content_types}")
    
    # Store page sizes for the whole document
    page_sizes = {}
    for page_idx, page_data in layout_by_page.items():
        page_sizes[page_idx] = page_data.get("page_size", [0, 0])
    
    # Create a copy of the content list to preserve original order
    enriched_list = [dict(item) for item in content_list]
    unmatched_indices = []
    
    # First pass matching
    for idx, item in enumerate(content_list):
        page_idx = item.get("page_idx")
        if page_idx is None:
            unmatched_indices.append(idx)
            continue
            
        page_layout = layout_by_page.get(page_idx, {
            "text_blocks": [], 
            "images": [], 
            "all_lines": [], 
            "equations": [],
            "inline_equations": [],
            "block_equations": []
        })
        
        # Add page size to each item
        if page_idx in page_sizes:
            enriched_list[idx]["page_size"] = page_sizes[page_idx]
        
        item_type = item.get("type", "").lower()
        
        # Match based on content type
        if item_type == "text":
            match_text_item(item, page_layout, enriched_list[idx])
        elif item_type == "image":
            match_image_item(item, page_layout, enriched_list[idx])
        elif item_type in ["equation", "math"]:
            match_equation_item(item, page_layout, enriched_list[idx])
        elif item_type in ["table", "chart", "figure"]:
            match_table_figure_item(item, page_layout, enriched_list[idx])
        else:
            # Handle other content types or unknown types
            match_generic_item(item, page_layout, enriched_list[idx])
        
        # Check if the item was successfully matched
        if "bbox" not in enriched_list[idx]:
            unmatched_indices.append(idx)
    
    # Try progressive matching for unmatched items
    if unmatched_indices:
        print(f"First pass matching: {len(unmatched_indices)} unmatched items, trying advanced matching...")
        still_unmatched = []
        
        for idx in unmatched_indices:
            item = content_list[idx]
            page_idx = item.get("page_idx")
            if page_idx is None:
                still_unmatched.append(idx)
                continue
                
            page_layout = layout_by_page.get(page_idx, {
                "text_blocks": [], 
                "images": [], 
                "equations": [],
                "inline_equations": [],
                "block_equations": []
            })
            
            item_type = item.get("type", "").lower()
            
            # Try advanced matching based on content type
            if item_type == "text":
                if not try_advanced_text_matching(item, page_layout, enriched_list[idx]):
                    still_unmatched.append(idx)
            elif item_type in ["equation", "math"]:
                if not try_advanced_equation_matching(item, page_layout, enriched_list[idx]):
                    still_unmatched.append(idx)
            elif item_type == "image":
                if not try_advanced_image_matching(item, page_layout, enriched_list[idx]):
                    still_unmatched.append(idx)
            else:
                if not try_advanced_generic_matching(item, page_layout, enriched_list[idx]):
                    still_unmatched.append(idx)
        
        # Last resort: cross-type matching for remaining unmatched items
        if still_unmatched:
            print(f"Advanced matching: {len(still_unmatched)} items still unmatched, trying cross-type matching...")
            final_unmatched = []
            
            for idx in still_unmatched:
                item = content_list[idx]
                page_idx = item.get("page_idx")
                if page_idx is None:
                    final_unmatched.append(idx)
                    continue
                
                page_layout = layout_by_page.get(page_idx, {
                    "text_blocks": [], 
                    "images": [], 
                    "equations": [],
                    "inline_equations": [],
                    "block_equations": []
                })
                
                # Try cross-type matching as second-to-last resort
                if not try_cross_type_matching(item, page_layout, enriched_list[idx]):
                    # Position-based matching as last resort
                    if not position_based_matching(item, page_layout, enriched_list[idx]):
                        final_unmatched.append(idx)
            
            unmatched_indices = final_unmatched
        else:
            unmatched_indices = still_unmatched
            
        print(f"Final unmatched count: {len(unmatched_indices)}")
    
    # Prepare final unmatched items list for reporting
    unmatched_items = [content_list[idx] for idx in unmatched_indices]
    
    # Verify and adjust bounding boxes for consistency
    verify_and_adjust_bboxes(enriched_list, page_sizes)
    
    # Save enriched content list
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched_list, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved enriched file: {output_path}")
    
    # Handle unmatched items
    if unmatched_items:
        unmatched_output_path = os.path.splitext(output_path)[0] + "_unmatched.json"
        with open(unmatched_output_path, "w", encoding="utf-8") as f:
            json.dump(unmatched_items, f, indent=2, ensure_ascii=False)
            
        print(f"🚨 {len(unmatched_items)} unmatched items saved to: {unmatched_output_path}")
    else:
        print("✅ All items matched successfully")
    
    return enriched_list, unmatched_items

def preprocess_middle_json(middle_json):
    """
    Extract and organize layout information from middle JSON for efficient matching.
    Enhanced to handle more content types and structures.
    """
    layout_by_page = {}
    
    for page in middle_json.get("pdf_info", []):
        page_idx = page["page_idx"]
        layout_by_page[page_idx] = {
            "text_blocks": [],
            "images": [],
            "equations": [],       # Generic equations
            "inline_equations": [], # Specifically inline equations
            "block_equations": [],  # Specifically block equations
            "tables": [],          # Tables
            "all_lines": [],       # Store all lines for more granular matching
            "all_spans": [],       # Store all spans for the most granular matching
            "page_size": page.get("page_size", [0, 0])
        }
        
        # Process all blocks (including titles, text, etc.)
        for block_type in ["preproc_blocks", "para_blocks"]:
            for block in page.get(block_type, []):
                block_type_name = block.get("type", "")
                full_text = ""
                all_spans = []
                equation_spans = []
                inline_equation_spans = []
                
                # Collect spans from all lines in the block
                for line in block.get("lines", []):
                    line_text = ""
                    line_spans = []
                    
                    for span in line.get("spans", []):
                        span_content = span.get("content", "").strip()
                        span_type = span.get("type", "")
                        
                        if span_content:
                            line_text += " " + span_content
                            line_spans.append(span)
                            all_spans.append(span)
                            
                            # Store span in appropriate category
                            layout_by_page[page_idx]["all_spans"].append({
                                "text": span_content,
                                "text_lower": span_content.lower(),
                                "bbox": span.get("bbox", []),
                                "type": span_type,
                                "block_type": block_type_name,
                                "block_bbox": block.get("bbox", [])
                            })
                            
                            # Collect equation spans by type
                            if span_type in ["equation", "block_equation"]:
                                equation_spans.append(span)
                                layout_by_page[page_idx]["equations"].append({
                                    "text": span_content,
                                    "bbox": span.get("bbox", []),
                                    "type": span_type,
                                    "block_bbox": block.get("bbox", [])
                                })
                                layout_by_page[page_idx]["block_equations"].append({
                                    "text": span_content,
                                    "bbox": span.get("bbox", []),
                                    "type": span_type,
                                    "block_bbox": block.get("bbox", [])
                                })
                            elif span_type == "inline_equation":
                                inline_equation_spans.append(span)
                                layout_by_page[page_idx]["equations"].append({
                                    "text": span_content,
                                    "bbox": span.get("bbox", []),
                                    "type": span_type,
                                    "block_bbox": block.get("bbox", [])
                                })
                                layout_by_page[page_idx]["inline_equations"].append({
                                    "text": span_content,
                                    "bbox": span.get("bbox", []),
                                    "type": span_type,
                                    "block_bbox": block.get("bbox", [])
                                })
                    
                    if line_text.strip():
                        layout_by_page[page_idx]["all_lines"].append({
                            "text": line_text.strip(),
                            "text_lower": line_text.strip().lower(),
                            "bbox": line.get("bbox", []),
                            "spans": line_spans,
                            "block_type": block_type_name
                        })
                    
                    full_text += " " + line_text
                
                # Add block to text_blocks
                if full_text.strip():
                    layout_by_page[page_idx]["text_blocks"].append({
                        "text": full_text.strip(),
                        "text_lower": full_text.strip().lower(),
                        "bbox": block.get("bbox", []),
                        "type": block_type_name,
                        "spans": all_spans
                    })
        
        # Process interline equations separately (they have their own structure)
        for equation in page.get("interline_equations", []):
            eq_text = ""
            for line in equation.get("lines", []):
                for span in line.get("spans", []):
                    if span.get("type") == "interline_equation":
                        eq_text = span.get("content", "").strip()
            
            if eq_text:
                layout_by_page[page_idx]["equations"].append({
                    "text": eq_text,
                    "bbox": equation.get("bbox", []),
                    "type": "interline_equation",
                    "block_bbox": equation.get("bbox", [])
                })
                layout_by_page[page_idx]["block_equations"].append({
                    "text": eq_text,
                    "bbox": equation.get("bbox", []),
                    "type": "interline_equation",
                    "block_bbox": equation.get("bbox", [])
                })
        
        # Process whole block equations (some PDFs have them as separate blocks)
        for block_type in ["preproc_blocks", "para_blocks"]:
            for block in page.get(block_type, []):
                if block.get("type") in ["equation", "block_equation", "interline_equation"]:
                    equation_text = ""
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            equation_text += " " + span.get("content", "").strip()
                    
                    if equation_text.strip():
                        layout_by_page[page_idx]["equations"].append({
                            "text": equation_text.strip(),
                            "bbox": block.get("bbox", []),
                            "type": "block_equation",
                            "block_bbox": block.get("bbox", [])
                        })
                        layout_by_page[page_idx]["block_equations"].append({
                            "text": equation_text.strip(),
                            "bbox": block.get("bbox", []),
                            "type": "block_equation",
                            "block_bbox": block.get("bbox", [])
                        })
        
        # Process images
        for img in page.get("images", []):
            image_path = ""
            for block in img.get("blocks", []):
                if block.get("type") == "image_body":
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            if span.get("type") == "image":
                                image_path = span.get("image_path", "")
            
            if image_path:
                layout_by_page[page_idx]["images"].append({
                    "image_path": image_path,
                    "bbox": img.get("bbox", []),
                    "type": "image"
                })
        
        # Process tables (if available)
        for table in page.get("tables", []):
            layout_by_page[page_idx]["tables"].append({
                "bbox": table.get("bbox", []),
                "type": "table"
            })
    
    return layout_by_page

def match_text_item(item, page_layout, new_item):
    """
    Match a text item with its corresponding layout element.
    Enhanced to handle different text scenarios and structures.
    """
    # Get text content and clean for comparison
    item_text = item.get("text", "").strip()
    item_text_clean = clean_for_comparison(item_text.lower())
    
    # Check if item has a text level (indicates heading/title)
    is_heading = item.get("text_level") is not None and item.get("text_level") > 0
    
    if is_heading:
        # Try to match with title blocks first
        best_title_match = None
        best_title_score = 0
        
        for block in page_layout["text_blocks"]:
            if block["type"] in ["title", "heading"]:
                block_text_clean = clean_for_comparison(block["text_lower"])
                score = similarity(block_text_clean, item_text_clean)
                
                if score > best_title_score and score >= 0.7:  # Lower threshold for headings
                    best_title_score = score
                    best_title_match = block
        
        if best_title_match:
            new_item["bbox"] = best_title_match["bbox"]
            new_item["match_score"] = best_title_score
            new_item["match_type"] = "block"
            return True
    
    # Try normal text block matching
    best_match = None
    best_score = 0
    
    # Try to match with text blocks
    for block in page_layout["text_blocks"]:
        block_text_clean = clean_for_comparison(block["text_lower"])
        score = similarity(block_text_clean, item_text_clean)
        
        if score > best_score and score >= 0.85:
            best_score = score
            best_match = block
    
    if best_match:
        new_item["bbox"] = best_match["bbox"]
        new_item["match_score"] = best_score
        new_item["match_type"] = "block"
        return True
    
    # If no direct match, check if the text contains LaTeX equations
    if "$" in item_text:
        # This might be a text with embedded equations
        # Try to match based on text fragments and equation parts
        if match_text_with_equations(item_text, page_layout, new_item):
            return True
    
    # Try window-based matching for longer texts that might span multiple blocks
    if len(item_text) > 100:
        if find_and_merge_text_fragments(item_text_clean, page_layout, new_item):
            return True
    
    # Try substring matching for partial matches
    if try_substring_matching(item_text, page_layout, new_item):
        return True
    
    return False

def match_text_with_equations(item_text, page_layout, new_item):
    """
    Handle text that contains embedded LaTeX equations.
    """
    # Extract equations from the text
    equations = re.findall(r'\$(.*?)\$', item_text)
    text_parts = re.split(r'\$.*?\$', item_text)
    
    if not equations:
        return False
    
    # Look for blocks that contain both text parts and equations
    potential_blocks = []
    
    for block in page_layout["text_blocks"]:
        block_text = block["text"]
        match_count = 0
        
        # Check if block contains the text parts
        for text_part in text_parts:
            if text_part.strip() and text_part.strip().lower() in block_text.lower():
                match_count += 1
        
        # Check if block contains the equations
        for span in block.get("spans", []):
            if span.get("type") in ["inline_equation", "equation"]:
                span_content = span.get("content", "").strip()
                for eq in equations:
                    eq_clean = clean_equation_for_comparison(eq)
                    span_clean = clean_equation_for_comparison(span_content)
                    if similarity(eq_clean, span_clean) > 0.7:
                        match_count += 1
        
        # If we found most of the parts, consider it a match
        required_matches = max(1, len(text_parts) + len(equations) - 1) // 2
        if match_count >= required_matches:
            potential_blocks.append((block, match_count))
    
    if potential_blocks:
        # Use the block with the most matches
        best_block = max(potential_blocks, key=lambda x: x[1])[0]
        new_item["bbox"] = best_block["bbox"]
        new_item["match_type"] = "text_with_equations"
        return True
    
    # Look for spans that might match equation parts
    equation_spans = []
    for eq in equations:
        eq_clean = clean_equation_for_comparison(eq)
        
        for eq_obj in page_layout["inline_equations"] + page_layout["block_equations"]:
            eq_text = eq_obj["text"]
            eq_text_clean = clean_equation_for_comparison(eq_text)
            
            if similarity(eq_clean, eq_text_clean) > 0.7:
                equation_spans.append(eq_obj)
    
    if equation_spans and text_parts:
        # Look for text blocks that match the text parts
        text_blocks = []
        for text_part in text_parts:
            if not text_part.strip():
                continue
                
            text_part_clean = clean_for_comparison(text_part.lower())
            
            for block in page_layout["text_blocks"]:
                block_text_clean = clean_for_comparison(block["text_lower"])
                if text_part_clean in block_text_clean:
                    text_blocks.append(block)
        
        if text_blocks and len(text_blocks) + len(equation_spans) >= (len(text_parts) + len(equations)) // 2:
            # Merge bounding boxes of text blocks and equation spans
            all_bboxes = [block["bbox"] for block in text_blocks] + [span["bbox"] for span in equation_spans]
            new_item["bbox"] = merge_bbox(all_bboxes)
            new_item["match_type"] = "merged_text_equations"
            return True
    
    return False

def match_image_item(item, page_layout, new_item):
    """
    Match an image item with its corresponding layout element.
    """
    img_path = item.get("img_path", "")
    if not img_path:
        return False
        
    img_filename = os.path.basename(img_path)
    
    # First try exact filename match
    for img in page_layout["images"]:
        image_path = img.get("image_path", "")
        if image_path and os.path.basename(image_path) == img_filename:
            new_item["bbox"] = img["bbox"]
            new_item["match_type"] = "exact_image"
            return True
    
    # If no exact match, try fuzzy filename matching
    for img in page_layout["images"]:
        image_path = img.get("image_path", "")
        if image_path:
            base_name = os.path.basename(image_path)
            # Check if filenames are similar regardless of extension
            base_name_no_ext = os.path.splitext(base_name)[0]
            img_name_no_ext = os.path.splitext(img_filename)[0]
            
            if similarity(base_name_no_ext, img_name_no_ext) > 0.7:
                new_item["bbox"] = img["bbox"]
                new_item["match_type"] = "fuzzy_image"
                return True
    
    return False

def match_equation_item(item, page_layout, new_item):
    """
    Match an equation item with its corresponding layout element.
    Enhanced to handle different equation formats and structures.
    """
    equation_text = item.get("text", "").strip()
    if not equation_text:
        return False
    
    # Determine equation type based on content or format attribute
    equation_format = item.get("text_format", "").lower()
    is_block_equation = equation_format == "latex" or equation_text.startswith("$$") or equation_text.startswith("\\begin{")
    
    # Clean the equation text for comparison
    equation_text_clean = clean_equation_for_comparison(equation_text)
    
    # Lists of equation objects to check, prioritizing matching type
    primary_eq_list = page_layout["block_equations"] if is_block_equation else page_layout["inline_equations"]
    secondary_eq_list = page_layout["inline_equations"] if is_block_equation else page_layout["block_equations"]
    
    # First try exact matches with the appropriate equation type
    for eq_list, match_threshold, match_type_prefix in [
        (primary_eq_list, 0.75, "primary_"),
        (secondary_eq_list, 0.75, "secondary_"),
        (page_layout["equations"], 0.7, "any_")
    ]:
        best_match = None
        best_score = 0
        
        for equation in eq_list:
            eq_text_clean = clean_equation_for_comparison(equation["text"])
            score = similarity(eq_text_clean, equation_text_clean)
            
            if score > best_score and score >= match_threshold:
                best_score = score
                best_match = equation
        
        if best_match:
            new_item["bbox"] = best_match["bbox"]
            new_item["match_score"] = best_score
            new_item["match_type"] = f"{match_type_prefix}equation"
            return True
    
    # If no direct match, try structural equation matching
    math_structure = extract_equation_structure(equation_text)
    
    if math_structure:
        for equation in page_layout["equations"]:
            eq_structure = extract_equation_structure(equation["text"])
            structure_similarity = compare_equation_structures(math_structure, eq_structure)
            
            if structure_similarity > 0.7:
                new_item["bbox"] = equation["bbox"]
                new_item["match_type"] = "structure_equation"
                return True
    
    # Try to find equation embedded in text blocks
    if find_equation_in_blocks(equation_text_clean, page_layout, new_item):
        return True
    
    return False

def match_table_figure_item(item, page_layout, new_item):
    """
    Match a table or figure item with its corresponding layout element.
    """
    # First check if there's a direct match in tables collection
    if "tables" in page_layout and page_layout["tables"]:
        # For tables, we don't have much to match on except position
        # So we'll just use the first table on the page if there's only one
        if len(page_layout["tables"]) == 1:
            new_item["bbox"] = page_layout["tables"][0]["bbox"]
            new_item["match_type"] = "only_table"
            return True
        
        # If multiple tables, we could try to match based on content
        # But this is challenging without table content
        # For now, just return false to fall back to advanced matching
    
    return False

def match_generic_item(item, page_layout, new_item):
    """
    Match a generic or unknown item type with a best-effort approach.
    """
    # Try to infer type from content
    item_text = item.get("text", "").strip()
    
    # If it has text, treat similar to text item but with lower threshold
    if item_text:
        item_text_clean = clean_for_comparison(item_text.lower())
        
        best_match = None
        best_score = 0
        
        # Try to match with any text block
        for block in page_layout["text_blocks"]:
            block_text_clean = clean_for_comparison(block["text_lower"])
            score = similarity(block_text_clean, item_text_clean)
            
            if score > best_score and score >= 0.75:  # Lower threshold for generic items
                best_score = score
                best_match = block
        
        if best_match:
            new_item["bbox"] = best_match["bbox"]
            new_item["match_score"] = best_score
            new_item["match_type"] = "generic_block"
            return True
        
        # If no direct match, try substring matching
        if try_substring_matching(item_text, page_layout, new_item):
            return True
    
    return False

def try_advanced_text_matching(item, page_layout, new_item):
    """
    Advanced text matching techniques for difficult cases.
    Enhanced to handle various text scenarios.
    """
    item_text = item.get("text", "").strip()
    item_text_clean = clean_for_comparison(item_text.lower())
    
    # Very short text - try character-level matching
    if len(item_text) < 30:
        if try_character_level_matching(item_text, page_layout, new_item):
            return True
    
    # Try matching with individual spans instead of whole blocks
    for span in page_layout["all_spans"]:
        span_text_clean = clean_for_comparison(span["text_lower"])
        score = similarity(span_text_clean, item_text_clean)
        
        if score >= 0.8:
            new_item["bbox"] = span["bbox"] if span["bbox"] else span["block_bbox"]
            new_item["match_score"] = score
            new_item["match_type"] = "span_match"
            return True
    
    # Try matching with individual lines
    for line in page_layout["all_lines"]:
        line_text_clean = clean_for_comparison(line["text_lower"])
        score = similarity(line_text_clean, item_text_clean)
        
        if score >= 0.8:
            new_item["bbox"] = line["bbox"]
            new_item["match_score"] = score
            new_item["match_type"] = "line_match"
            return True
    
    # Try n-gram matching for pieces of text
    ngram_size = 4  # Use 4-word ngrams
    if try_ngram_matching(item_text, page_layout, new_item, ngram_size):
        return True
    
    # Try layout-based pattern matching as last resort
    if try_layout_pattern_matching(item, page_layout, new_item):
        return True
    
    return False

def try_advanced_equation_matching(item, page_layout, new_item):
    """
    Advanced matching techniques for equations.
    Enhanced to handle mathematical structures and partial matches.
    """
    equation_text = item.get("text", "").strip()
    if not equation_text:
        return False
    
    # Extract key mathematical symbols and operators
    eq_clean = clean_equation_for_comparison(equation_text)
    math_elements = extract_math_elements(eq_clean)
    
    if math_elements:
        # Check all equations for similar elements
        for equation in page_layout["equations"]:
            eq_text = clean_equation_for_comparison(equation["text"])
            matches = sum(1 for elem in math_elements if elem in eq_text)
            
            if matches >= len(math_elements) * 0.6:  # 60% of elements match
                new_item["bbox"] = equation["bbox"]
                new_item["match_type"] = "equation_elements"
                return True
        
        # Look for equation parts in spans
        for span in page_layout["all_spans"]:
            if span["type"] in ["equation", "inline_equation", "block_equation", "interline_equation"]:
                span_text = clean_equation_for_comparison(span["text"])
                matches = sum(1 for elem in math_elements if elem in span_text)
                
                if matches >= len(math_elements) * 0.6:
                    new_item["bbox"] = span["bbox"]
                    new_item["match_type"] = "equation_span_elements"
                    return True
        
        # Check text blocks for potential equation content
        for block in page_layout["text_blocks"]:
            block_text = block.get("text", "")
            matches = sum(1 for elem in math_elements if elem in block_text)
            
            if matches >= len(math_elements) * 0.6:
                new_item["bbox"] = block["bbox"]
                new_item["match_type"] = "equation_in_text"
                return True
    
    # Check for specific LaTeX commands or structures
    latex_patterns = get_latex_patterns(equation_text)
    if latex_patterns:
        for equation in page_layout["equations"]:
            eq_patterns = get_latex_patterns(equation["text"])
            common_patterns = set(latex_patterns) & set(eq_patterns)
            
            if common_patterns and len(common_patterns) >= len(latex_patterns) * 0.5:
                new_item["bbox"] = equation["bbox"]
                new_item["match_type"] = "latex_patterns"
                return True
    
    return False

def try_advanced_image_matching(item, page_layout, new_item):
    """
    Advanced image matching when filename matching fails.
    """
    # If there's only one image on the page, use it
    if len(page_layout["images"]) == 1:
        new_item["bbox"] = page_layout["images"][0]["bbox"]
        new_item["match_type"] = "only_image_on_page"
        return True
    
    # Try captions if available
    captions = item.get("img_caption", [])
    if captions:
        caption_text = " ".join(captions).lower()
        
        # Look for blocks that might be captions based on proximity to images
        for img in page_layout["images"]:
            img_bbox = img["bbox"]
            
            for block in page_layout["text_blocks"]:
                block_text = block["text_lower"]
                
                # Check if this block might be a caption based on position and content
                if is_likely_caption(block["bbox"], img_bbox) and similarity(block_text, caption_text) > 0.6:
                    new_item["bbox"] = img["bbox"]
                    new_item["match_type"] = "caption_match"
                    return True
    
    return False

def try_advanced_generic_matching(item, page_layout, new_item):
    """
    Advanced matching for generic or unknown content types.
    """
    # Try text-based matching first if item has text
    if "text" in item and item["text"]:
        return try_advanced_text_matching(item, page_layout, new_item)
    
    # If no text but we know the page, try position-based assignment
    page_size = page_layout.get("page_size", [0, 0])
    if page_size[0] > 0 and page_size[1] > 0:
        # Place in the middle of the page as a fallback
        new_item["bbox"] = [
            page_size[0] * 0.2,  # 20% from left
            page_size[1] * 0.4,  # 40% from top
            page_size[0] * 0.8,  # 80% from left
            page_size[1] * 0.6   # 60% from top
        ]
        new_item["match_type"] = "generic_position"
        return True
    
    return False

def try_cross_type_matching(item, page_layout, new_item):
    """
    Try matching across different content types - sometimes content is categorized 
    differently between the two JSON files.
    """
    item_type = item.get("type", "").lower()
    item_text = item.get("text", "").strip()
    
    # If no text to match on, can't do cross-type matching
    if not item_text:
        return False
    
    # Clean the text for comparison
    item_text_clean = clean_for_comparison(item_text.lower())
    
    # For equation type, try matching with text blocks
    if item_type in ["equation", "math"]:
        for block in page_layout["text_blocks"]:
            block_text = block["text"]
            
            # Look for the equation content in the text block
            # This happens when an equation in content list is treated as regular text in layout
            if item_text in block_text or similarity(item_text_clean, clean_for_comparison(block_text.lower())) > 0.7:
                new_item["bbox"] = block["bbox"]
                new_item["match_type"] = "equation_as_text"
                return True
    
    # For text type, try matching with equations
    if item_type == "text" and ("$" in item_text or "\\(" in item_text or "\\begin" in item_text):
        equation_text = re.sub(r'[^$\\()\[\]{}a-zA-Z0-9+\-*/=]', '', item_text)
        
        for equation in page_layout["equations"]:
            eq_text = re.sub(r'[^$\\()\[\]{}a-zA-Z0-9+\-*/=]', '', equation["text"])
            
            if equation_text in eq_text or eq_text in equation_text:
                new_item["bbox"] = equation["bbox"]
                new_item["match_type"] = "text_as_equation"
                return True
    
    return False

def position_based_matching(item, page_layout, new_item):
    """
    Last resort matching based on position in the document and item characteristics.
    Enhanced to be more intelligent about positioning.
    """
    page_size = page_layout.get("page_size", [0, 0])
    if not page_size[0] or not page_size[1]:
        return False
    
    item_type = item.get("type", "").lower()
    
    # Create appropriate fallback bbox based on content type and characteristics
    if item_type == "text":
        text_length = len(item.get("text", ""))
        is_heading = item.get("text_level", 0) > 0
        
        if is_heading:
            # Headings are typically at the top of the page or section
            y_position = page_size[1] * 0.1  # 10% down from top
            height = page_size[1] * 0.05     # 5% of page height
            
            new_item["bbox"] = [
                page_size[0] * 0.1,  # 10% from left
                y_position,
                page_size[0] * 0.9,  # 90% from left
                y_position + height
            ]
        else:
            # Regular text - estimate position based on length
            if text_length < 100:  # Short text
                y_position = page_size[1] * 0.3
                height = page_size[1] * 0.1
            elif text_length < 500:  # Medium text
                y_position = page_size[1] * 0.4
                height = page_size[1] * 0.2
            else:  # Long text
                y_position = page_size[1] * 0.3
                height = page_size[1] * 0.4
            
            new_item["bbox"] = [
                page_size[0] * 0.1,
                y_position,
                page_size[0] * 0.9,
                y_position + height
            ]
    
    elif item_type in ["equation", "math"]:
        # Equations are often centered
        y_position = page_size[1] * 0.5  # Middle of page
        height = page_size[1] * 0.1      # 10% of page height
        
        new_item["bbox"] = [
            page_size[0] * 0.25,  # 25% from left
            y_position - height/2,
            page_size[0] * 0.75,  # 75% from left
            y_position + height/2
        ]
    
    elif item_type == "image":
        # Images are often larger and can be anywhere
        y_position = page_size[1] * 0.5  # Middle of page
        height = page_size[1] * 0.3      # 30% of page height
        
        new_item["bbox"] = [
            page_size[0] * 0.2,  # 20% from left
            y_position - height/2,
            page_size[0] * 0.8,  # 80% from left
            y_position + height/2
        ]
    
    else:
        # Generic fallback
        new_item["bbox"] = [
            page_size[0] * 0.1,
            page_size[1] * 0.4,
            page_size[0] * 0.9,
            page_size[1] * 0.6
        ]
    
    new_item["match_type"] = "position_fallback"
    return True

def verify_and_adjust_bboxes(enriched_list, page_sizes):
    """
    Validate and adjust bounding boxes for consistency and correctness.
    """
    for item in enriched_list:
        if "bbox" not in item:
            continue
            
        bbox = item["bbox"]
        page_idx = item.get("page_idx")
        
        if page_idx is not None and page_idx in page_sizes:
            page_width, page_height = page_sizes[page_idx]
            
            # Ensure bbox has 4 values
            if len(bbox) != 4:
                # Default to a central position
                item["bbox"] = [
                    page_width * 0.1,
                    page_height * 0.4,
                    page_width * 0.9,
                    page_height * 0.6
                ]
                continue
            
            # Ensure bbox values are within page boundaries
            x0, y0, x1, y1 = bbox
            
            # Ensure correct ordering (x0 < x1, y0 < y1)
            if x0 > x1:
                x0, x1 = x1, x0
            if y0 > y1:
                y0, y1 = y1, y0
            
            # Ensure within page boundaries
            x0 = max(0, min(x0, page_width))
            y0 = max(0, min(y0, page_height))
            x1 = max(0, min(x1, page_width))
            y1 = max(0, min(y1, page_height))
            
            # Ensure minimum size
            if x1 - x0 < 5:
                x1 = min(x0 + 5, page_width)
            if y1 - y0 < 5:
                y1 = min(y0 + 5, page_height)
            
            item["bbox"] = [x0, y0, x1, y1]

def clean_for_comparison(text):
    """
    Clean text for better matching by normalizing whitespace and handling special content.
    Enhanced to handle more text variations.
    """
    if not text:
        return ""
        
    # Normalize whitespace and lowercase
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Handle LaTeX equations consistently
    text = re.sub(r'\$\$.*?\$\$', 'BLOCK_EQUATION', text)
    text = re.sub(r'\$.*?\$', 'INLINE_EQUATION', text)
    text = re.sub(r'\\begin\{.*?\}.*?\\end\{.*?\}', 'BLOCK_EQUATION', text, flags=re.DOTALL)
    
    # Normalize unicode characters
    text = text.replace('−', '-')  # Replace unicode minus with hyphen
    text = text.replace('…', '...') # Replace ellipsis
    
    return text

def clean_equation_for_comparison(equation_text):
    """
    Clean equation text for comparison by removing LaTeX formatting commands.
    Enhanced to handle more LaTeX variations.
    """
    if not equation_text:
        return ""
    
    # Remove the outer $$ or $ delimiters
    equation_text = re.sub(r'^\$\$(.*)\$\$', r'\1', equation_text)
    equation_text = re.sub(r'^\$(.*)\$', r'\1', equation_text)
    
    # Normalize whitespace
    equation_text = re.sub(r'\s+', ' ', equation_text).strip()
    
    # Remove common LaTeX formatting commands
    replacements = [
        (r'\\begin\{array\}(\{[^}]*\})', ' '),
        (r'\\end\{array\}', ' '),
        (r'\\left', ' '),
        (r'\\right', ' '),
        (r'\\big', ' '),
        (r'\\Big', ' '),
        (r'\\bigg', ' '),
        (r'\\Bigg', ' '),
        (r'\\mathrm', ' '),
        (r'\\mathbf', ' '),
        (r'\\mathit', ' '),
        (r'\\mathsf', ' '),
        (r'\\boldsymbol', ' '),
        (r'\\color\{[^}]*\}', ' '),
        (r'\\textcolor\{[^}]*\}', ' '),
        (r'\{', ''),
        (r'\}', ''),
        (r'\&', ' '),
        (r'\\\\', ' '),
        (r'\\quad', ' '),
        (r'\\qquad', ' '),
        (r'\\hspace\{[^}]*\}', ' '),
        (r'\\vspace\{[^}]*\}', ' '),
        (r'\\operatorname\*?\{[^}]*\}', ''),
        (r'\\ensuremath', ''),
        (r'\\mathbb\{[^}]*\}', ''),
        (r'\\overrightarrow', ''),
        (r'\\overleftarrow', ''),
        (r'\\overline', ''),
        (r'\\underline', ''),
        (r'\\frac', ''),
        (r'\\dfrac', ''),
        (r'\\partial', 'd'),
        (r'\\infty', 'infinity'),
        (r'\\to', '->'),
        (r'\\rightarrow', '->'),
        (r'\\leftarrow', '<-'),
        (r'\\Rightarrow', '=>'),
        (r'\\Leftarrow', '<='),
        (r'\\iff', '<=>'),
        (r'\\cdot', '*')
    ]
    
    for pattern, replacement in replacements:
        equation_text = re.sub(pattern, replacement, equation_text)
    
    # Normalize unicode characters
    equation_text = equation_text.replace('−', '-')  # Replace unicode minus with hyphen
    
    # Normalize whitespace again
    equation_text = re.sub(r'\s+', ' ', equation_text).strip()
    
    return equation_text

def ultra_clean_for_comparison(text):
    """
    Very aggressive cleaning for difficult matches - keep only alphanumeric chars.
    """
    if not text:
        return ""
        
    # Keep only alphanumeric characters and convert to lowercase
    return re.sub(r'[^a-zA-Z0-9]', '', text.lower())

def extract_math_elements(eq_text):
    """
    Extract key mathematical elements (operators, variables, functions) from equation text.
    """
    if not eq_text:
        return []
    
    # Extract common math operators and functions
    operators = re.findall(r'[+\-*/=<>^_]+', eq_text)
    functions = re.findall(r'\\[a-zA-Z]+', eq_text)  # LaTeX commands like \sin, \log
    variables = re.findall(r'[a-zA-Z][a-zA-Z0-9]*', eq_text)  # Variables like x, y, a1
    
    # Combine all elements
    elements = operators + functions + [v for v in variables if len(v) == 1]  # Single-letter variables are more likely to be mathematical
    
    return elements

def extract_equation_structure(eq_text):
    """
    Extract the structural elements of an equation for comparison.
    """
    if not eq_text:
        return {}
    
    structure = {
        'delimiters': [],
        'operators': [],
        'functions': [],
        'variables': []
    }
    
    # Extract delimiters (parentheses, brackets, etc.)
    structure['delimiters'] = re.findall(r'[\(\)\[\]\{\}]', eq_text)
    
    # Extract operators
    structure['operators'] = re.findall(r'[+\-*/=<>^]', eq_text)
    
    # Extract LaTeX functions
    structure['functions'] = re.findall(r'\\[a-zA-Z]+', eq_text)
    
    # Extract likely variables
    structure['variables'] = re.findall(r'[a-zA-Z](?![a-zA-Z])', eq_text)  # Single letters
    
    return structure

def compare_equation_structures(struct1, struct2):
    """
    Compare the structures of two equations and return a similarity score.
    """
    if not struct1 or not struct2:
        return 0
    
    # Compare each component type
    similarity_scores = []
    
    for component in ['delimiters', 'operators', 'functions', 'variables']:
        set1 = set(struct1.get(component, []))
        set2 = set(struct2.get(component, []))
        
        # Calculate Jaccard similarity if either set is non-empty
        if set1 or set2:
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            
            if union > 0:
                similarity_scores.append(intersection / union)
            else:
                similarity_scores.append(0)
    
    # Return average similarity if we have scores
    return sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0

def get_latex_patterns(text):
    """
    Extract key LaTeX command patterns for matching.
    """
    if not text:
        return []
    
    # Match LaTeX commands with their arguments
    patterns = re.findall(r'\\[a-zA-Z]+(\{[^}]*\})*', text)
    
    # Match key LaTeX structures
    structures = re.findall(r'\\begin\{[^}]*\}|\\end\{[^}]*\}', text)
    
    return patterns + structures

def try_character_level_matching(item_text, page_layout, new_item):
    """
    Character-level matching for short texts that might have formatting differences.
    """
    # Ultra-clean both item text and block texts - keep only alphanumeric
    item_chars = ultra_clean_for_comparison(item_text)
    if not item_chars:
        return False
    
    for block in page_layout["text_blocks"]:
        block_chars = ultra_clean_for_comparison(block["text"])
        
        # Check if one is contained within the other
        if (item_chars in block_chars or 
            block_chars in item_chars or 
            similarity(item_chars, block_chars) > 0.75):
            
            new_item["bbox"] = block["bbox"]
            new_item["match_type"] = "character_level"
            return True
    
    # Try individual spans if block matching failed
    for span in page_layout["all_spans"]:
        span_chars = ultra_clean_for_comparison(span["text"])
        
        if (item_chars in span_chars or 
            span_chars in item_chars or 
            similarity(item_chars, span_chars) > 0.75):
            
            new_item["bbox"] = span["bbox"] if span["bbox"] else span["block_bbox"]
            new_item["match_type"] = "character_span_level"
            return True
    
    return False

def try_ngram_matching(item_text, page_layout, new_item, ngram_size=4):
    """
    Match text based on n-grams (sequences of n words).
    """
    item_text = item_text.lower()
    words = re.findall(r'\b\w+\b', item_text)
    
    if len(words) < ngram_size:
        return False
    
    # Create n-grams from the item text
    ngrams = []
    for i in range(len(words) - ngram_size + 1):
        ngram = ' '.join(words[i:i+ngram_size])
        ngrams.append(ngram)
    
    # Look for blocks containing any of these n-grams
    matching_blocks = []
    
    for block in page_layout["text_blocks"]:
        block_text = block["text_lower"]
        matches = 0
        
        for ngram in ngrams:
            if ngram in block_text:
                matches += 1
        
        if matches > 0:
            matching_blocks.append((block, matches))
    
    # Use the block with the most matches
    if matching_blocks:
        matching_blocks.sort(key=lambda x: x[1], reverse=True)
        best_block = matching_blocks[0][0]
        
        new_item["bbox"] = best_block["bbox"]
        new_item["match_type"] = "ngram"
        return True
    
    return False

def find_and_merge_text_fragments(item_text, page_layout, new_item):
    """
    Try to find a sequence of line fragments that together match the item text.
    Enhanced to handle more complex text fragments.
    """
    # Sort all lines by vertical position for logical reading order
    all_lines = sorted(page_layout["all_lines"], key=lambda l: l.get("bbox", [0, 0, 0, 0])[1])
    
    # Try sliding windows of different sizes
    max_window = min(10, len(all_lines))  # Increase max window size to 10 lines
    
    for window_size in range(1, max_window + 1):
        for i in range(len(all_lines) - window_size + 1):
            window_lines = all_lines[i:i+window_size]
            
            # Skip if the lines are too far apart vertically
            if window_size > 1:
                line_bboxes = [line.get("bbox", []) for line in window_lines if line.get("bbox")]
                if line_bboxes and not are_lines_connected(line_bboxes):
                    continue
            
            # Combine text from the window
            window_text = " ".join(line["text_lower"] for line in window_lines)
            window_text_clean = clean_for_comparison(window_text)
            
            # Try different similarity metrics
            exact_match = item_text.lower() in window_text_clean
            similarity_score = similarity(window_text_clean, item_text)
            
            if exact_match or similarity_score >= 0.8:
                # Found a match - merge bounding boxes
                bboxes = [line.get("bbox", []) for line in window_lines if line.get("bbox")]
                if bboxes:
                    new_item["bbox"] = merge_bbox(bboxes)
                    new_item["match_score"] = similarity_score if not exact_match else 1.0
                    new_item["match_type"] = "window"
                    return True
    
    return False

def find_equation_in_blocks(equation_text_clean, page_layout, new_item):
    """
    Try to find an equation within text blocks when it's not found in dedicated equation blocks.
    Enhanced to handle more equation variations.
    """
    # Look for equation spans in text blocks
    for block in page_layout["text_blocks"]:
        for span in block.get("spans", []):
            if span.get("type") in ["equation", "inline_equation", "block_equation", "interline_equation"]:
                span_text = clean_equation_for_comparison(span.get("content", ""))
                score = similarity(span_text, equation_text_clean)
                
                if score >= 0.7:  # Lower threshold for equations
                    # If span has bbox, use it; otherwise use block bbox
                    if span.get("bbox"):
                        new_item["bbox"] = span["bbox"]
                    else:
                        new_item["bbox"] = block.get("bbox", [])
                    new_item["match_score"] = score
                    new_item["match_type"] = "equation_in_block"
                    return True
    
    # Look for equation-like patterns in text
    for block in page_layout["text_blocks"]:
        block_text = block.get("text", "")
        # Extract potential equations using regex
        equation_candidates = re.findall(r'\$(.*?)\$', block_text)
        equation_candidates += re.findall(r'\\begin\{equation\}(.*?)\\end\{equation\}', block_text, re.DOTALL)
        
        for candidate in equation_candidates:
            candidate_clean = clean_equation_for_comparison(candidate)
            score = similarity(candidate_clean, equation_text_clean)
            
            if score >= 0.7:
                new_item["bbox"] = block.get("bbox", [])
                new_item["match_score"] = score
                new_item["match_type"] = "equation_pattern"
                return True
    
    # Try structural matching
    math_elements = extract_math_elements(equation_text_clean)
    
    if math_elements:
        for block in page_layout["text_blocks"]:
            block_text = block.get("text", "")
            block_elements = extract_math_elements(block_text)
            
            # Check for overlap in mathematical elements
            common_elements = set(math_elements) & set(block_elements)
            
            if common_elements and len(common_elements) >= len(math_elements) * 0.6:
                new_item["bbox"] = block.get("bbox", [])
                new_item["match_type"] = "equation_elements_in_text"
                return True
    
    return False

def try_substring_matching(item_text, page_layout, new_item):
    """
    Match based on significant overlapping substrings.
    Enhanced to handle partial text matches.
    """
    item_clean = clean_for_comparison(item_text.lower())
    
    # Extract words and significant phrases (words with 4+ chars)
    words = re.findall(r'\b\w{4,}\b', item_clean)
    if not words:
        words = re.findall(r'\b\w{3,}\b', item_clean)  # Try with 3+ char words
        if not words:
            return False
    
    # Find blocks with the most matching significant words
    best_block = None
    best_matches = 0
    best_match_ratio = 0
    
    for block in page_layout["text_blocks"]:
        block_clean = clean_for_comparison(block["text_lower"])
        matches = sum(1 for word in words if word in block_clean)
        
        if matches > 0:
            match_ratio = matches / len(words)
            
            # Use both absolute count and ratio for scoring
            if (matches > best_matches and match_ratio > 0.3) or (matches >= best_matches and match_ratio > best_match_ratio):
                best_matches = matches
                best_match_ratio = match_ratio
                best_block = block
    
    if best_block and best_match_ratio >= 0.2:  # At least 40% of words match
        new_item["bbox"] = best_block["bbox"]
        new_item["match_score"] = best_match_ratio
        new_item["match_type"] = "substring"
        return True
    
    return False

def try_layout_pattern_matching(item, page_layout, new_item):
    """
    Match based on contextual patterns and layout structure.
    Enhanced to use more document structure cues.
    """
    # Special case for headings and titles
    if item.get("text_level", 0) > 0:  # It's a heading
        for block in page_layout["text_blocks"]:
            if block["type"] in ["title", "heading"]:
                # Check for minimal similarity
                item_text = item.get("text", "").lower()
                block_text = block.get("text_lower", "")
                
                # For titles, use a lower threshold and check both ways
                if (item_text in block_text or 
                    block_text in item_text or
                    similarity(clean_for_comparison(block_text), 
                              clean_for_comparison(item_text)) > 0.4):
                    new_item["bbox"] = block["bbox"]
                    new_item["match_type"] = "heading_pattern"
                    return True
    
    # Special case for footnote-like text
    item_text = item.get("text", "").strip().lower()
    if item_text.startswith("note:") or item_text.startswith("remark:") or item_text.startswith("n.b."):
        # Look for blocks at the bottom of the page
        page_height = page_layout.get("page_size", [0, 0])[1]
        if page_height > 0:
            bottom_blocks = []
            
            for block in page_layout["text_blocks"]:
                bbox = block.get("bbox", [])
                if bbox and len(bbox) >= 4:
                    # Check if block is in the bottom 20% of page
                    if bbox[3] > page_height * 0.6:
                        bottom_blocks.append(block)
            
            # Find most similar block in bottom section
            if bottom_blocks:
                best_match = None
                best_score = 0
                
                for block in bottom_blocks:
                    score = similarity(clean_for_comparison(block["text_lower"]), 
                                      clean_for_comparison(item_text))
                    if score > best_score and score > 0.5:
                        best_score = score
                        best_match = block
                
                if best_match:
                    new_item["bbox"] = best_match["bbox"]
                    new_item["match_type"] = "footnote_pattern"
                    return True
    
    return False

def is_likely_caption(block_bbox, img_bbox):
    """
    Check if a text block is likely to be a caption for an image
    based on their relative positions.
    """
    if not block_bbox or not img_bbox or len(block_bbox) < 4 or len(img_bbox) < 4:
        return False
    
    # Check if text block is directly below or above the image
    # and has similar horizontal alignment
    
    # Text below image
    below_image = (block_bbox[1] >= img_bbox[3] and  # Top of text >= bottom of image
                  block_bbox[1] - img_bbox[3] < 50 and  # Within 50 units
                  overlap_horizontal(block_bbox, img_bbox) > 0.5)  # 50% horizontal overlap
    
    # Text above image
    above_image = (block_bbox[3] <= img_bbox[1] and  # Bottom of text <= top of image
                  img_bbox[1] - block_bbox[3] < 50 and  # Within 50 units
                  overlap_horizontal(block_bbox, img_bbox) > 0.5)  # 50% horizontal overlap
    
    return below_image or above_image

def overlap_horizontal(bbox1, bbox2):
    """
    Calculate the percentage of horizontal overlap between two bounding boxes.
    """
    # Get horizontal ranges
    left1, right1 = bbox1[0], bbox1[2]
    left2, right2 = bbox2[0], bbox2[2]
    
    # Calculate overlap
    overlap_left = max(left1, left2)
    overlap_right = min(right1, right2)
    
    if overlap_right <= overlap_left:
        return 0  # No overlap
    
    overlap_width = overlap_right - overlap_left
    width1 = right1 - left1
    width2 = right2 - left2
    
    # Return overlap as a fraction of the smaller width
    return overlap_width / min(width1, width2)

def are_lines_connected(bboxes):
    """
    Check if a series of line bounding boxes likely form a paragraph.
    Enhanced to handle different text layouts.
    """
    if len(bboxes) <= 1:
        return True
        
    # Sort by vertical position
    sorted_bboxes = sorted(bboxes, key=lambda b: b[1])
    
    # Check vertical spacing and horizontal alignment
    for i in range(len(sorted_bboxes) - 1):
        current_bbox = sorted_bboxes[i]
        next_bbox = sorted_bboxes[i+1]
        
        # Current line height
        current_height = current_bbox[3] - current_bbox[1]
        
        # Gap between current and next line
        gap = next_bbox[1] - current_bbox[3]
        
        # Horizontal overlap ratio
        horiz_overlap = overlap_horizontal(current_bbox, next_bbox)
        
        # If gap is too large or horizontal overlap too small, lines are not connected
        if gap > current_height * 2 or horiz_overlap < 0.3:
            return False
    
    return True

def merge_bbox(bboxes):
    """
    Merge multiple bounding boxes into one encompassing box.
    """
    if not bboxes:
        return []
    
    valid_bboxes = [bbox for bbox in bboxes if len(bbox) >= 4]
    if not valid_bboxes:
        return []
    
    x0 = min(bbox[0] for bbox in valid_bboxes)
    y0 = min(bbox[1] for bbox in valid_bboxes)
    x1 = max(bbox[2] for bbox in valid_bboxes)
    y1 = max(bbox[3] for bbox in valid_bboxes)
    
    return [x0, y0, x1, y1]

def similarity(a, b):
    """
    Calculate string similarity ratio.
    """
    return SequenceMatcher(None, a, b).ratio()

if __name__ == "__main__":
    # Paths
    content_list_path = "output/hatier_test_page_6/auto/hatier_test_content_list.json"
    middle_json_path = "output/hatier_test_page_6/auto/hatier_test_middle.json"
    output_path = "output/hatier_test_page_6/auto/hatier_test_content_list_enriched.json"
    
    # Run the matching process
    match_content_with_bboxes(content_list_path, middle_json_path, output_path)