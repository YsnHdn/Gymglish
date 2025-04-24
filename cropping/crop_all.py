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


def extract_equation_items(enriched_json_path, middle_json_path=None):
    with open(enriched_json_path, 'r', encoding='utf-8') as f:
        content_list = json.load(f)

    equations = []
    for item in content_list:
        if (item.get('attributes', {}).get('equation')
            or item.get('type') in ('equation', 'inline_equation')):
            if 'page_idx' in item and 'bbox' in item:
                equations.append(item)

    if middle_json_path and os.path.exists(middle_json_path):
        with open(middle_json_path, 'r', encoding='utf-8') as f:
            middle = json.load(f)
        for page in middle.get("pdf_info", []):
            pidx = page["page_idx"]
            for blk_type in ("preproc_blocks", "para_blocks"):
                for blk in page.get(blk_type, []):
                    if blk.get("attributes", {}).get("equation"):
                        equations.append({
                            "page_idx": pidx,
                            "bbox": blk.get("bbox", []),
                            "content": blk.get("text", ""),
                            "type": "equation"
                        })
                    for line in blk.get("lines", []):
                        for span in line.get("spans", []):
                            if span.get("type") == "inline_equation":
                                equations.append({
                                    "page_idx": pidx,
                                    "bbox": span.get("bbox", []),
                                    "content": span.get("content", ""),
                                    "type": "inline_equation"
                                })

    print(f"Found {len(equations)} equations across the document")
    return equations


def get_rect_overlap_percentage(r1: fitz.Rect, r2: fitz.Rect) -> float:
    if not r1.intersects(r2):
        return 0.0
    inter = r1 & r2
    area_int = inter.width * inter.height
    smaller = min(r1.width * r1.height, r2.width * r2.height)
    return (area_int / smaller) if smaller > 0 else 0.0


def crop_pdf_content(
    pdf_path: str,
    enriched_json_path: str,
    output_dir: str,
    reference_pdf: str = None,
    middle_json_path: str = None,
    desired_dpi: int = 300
):
    os.makedirs(output_dir, exist_ok=True)
    eq_dir = os.path.join(output_dir, "equations")
    os.makedirs(eq_dir, exist_ok=True)

    # 1) scale factors
    scale_w, scale_h = (1.0, 1.0)
    if reference_pdf:
        scale_w, scale_h = compute_scale_factors(reference_pdf, pdf_path)

    # 2) high‑res matrix
    zoom = desired_dpi / 72.0
    mat  = fitz.Matrix(zoom * scale_w, zoom * scale_h)

    # 3) load content
    with open(enriched_json_path, 'r', encoding='utf-8') as f:
        content_list = json.load(f)

    content_by_page = {}
    for itm in content_list:
        p = itm.get('page_idx')
        if p is not None and 'bbox' in itm:
            content_by_page.setdefault(p, []).append(itm)

    equations = extract_equation_items(enriched_json_path, middle_json_path)
    doc = fitz.open(pdf_path)

    # --- crop normal items ---
    for pidx, items in content_by_page.items():
        if not (0 <= pidx < doc.page_count):
            continue
        page = doc[pidx]
        for idx, itm in enumerate(items):
            if itm.get('attributes', {}).get('equation') or itm.get('type') in ('equation', 'inline_equation'):
                continue

            x0, y0, x1, y1 = itm['bbox']
            clip = fitz.Rect(x0 * scale_w, y0 * scale_h, x1 * scale_w, y1 * scale_h)

            if itm.get('type') != 'image':
                pix = page.get_pixmap(matrix=mat, clip=clip)
                out = os.path.join(output_dir, f"page{pidx}_item{idx}.png")
                pix.save(out)
                itm['cropped_image_path'] = out
            else:
                # try direct extraction via image rects
                found = False
                for img in page.get_images(full=True):
                    xref = img[0]
                    rects = page.get_image_rects(xref)
                    if not rects:
                        continue
                    img_rect = rects[0]
                    if get_rect_overlap_percentage(clip, img_rect) > 0.5:
                        data = doc.extract_image(xref)["image"]
                        out = os.path.join(output_dir, f"page{pidx}_item{idx}_img.png")
                        with open(out, 'wb') as f:
                            f.write(data)
                        itm['cropped_image_path'] = out
                        found = True
                        break

                if not found:
                    pix = page.get_pixmap(matrix=mat, clip=clip)
                    out = os.path.join(output_dir, f"page{pidx}_item{idx}_region.png")
                    pix.save(out)
                    itm['cropped_image_path'] = out

    # --- crop equations ---
    for i, eq in enumerate(equations):
        pidx = eq.get('page_idx', -1)
        bbox = eq.get('bbox', [])
        if not (0 <= pidx < doc.page_count) or len(bbox) != 4:
            continue

        x0, y0, x1, y1 = bbox
        clip = fitz.Rect(x0 * scale_w, y0 * scale_h, x1 * scale_w, y1 * scale_h)
        pix  = doc[pidx].get_pixmap(matrix=mat, clip=clip, alpha=False)

        snippet = ''.join(c if c.isalnum() else '_' for c in eq.get('content', '')[:20]) or f"eq{i}"
        out = os.path.join(eq_dir, f"page{pidx}_eq{i}_{snippet}.png")
        pix.save(out)

        eq['cropped_image_path'] = out
        eq['dpi'] = desired_dpi

    # write JSON outputs
    updated = os.path.join(output_dir, 'content_with_crops.json')
    with open(updated, 'w', encoding='utf-8') as f:
        json.dump(content_list + equations, f, indent=2, ensure_ascii=False)

    eq_json = os.path.join(eq_dir, 'equations.json')
    with open(eq_json, 'w', encoding='utf-8') as f:
        json.dump(equations, f, indent=2, ensure_ascii=False)

    doc.close()
    print(f"✅ Cropping complete. All outputs in {output_dir}")


if __name__ == "__main__":
    # Example usage
    enriched_json_path = "output/hatier_test_page_6/auto/hatier_test_content_list_enriched.json"
    middle_json_path = "output/hatier_test_page_6/auto/hatier_test_middle.json"
    pdf_path = "output/hatier_test_page_6/auto/hatier_test_origin.pdf"
    output_dir = "output/document/hatier_test_page_6/content"
    reference_pdf = "output/hatier_test_page_6/auto/hatier_test_origin.pdf"  # Optional: if your JSON was generated on this same PDF
    # Process all content including equations with higher DPI
    crop_pdf_content(
        pdf_path, 
        enriched_json_path, 
        output_dir,
        middle_json_path=middle_json_path,  # Optional: add middle JSON for more equation extraction
        desired_dpi=300                    # Optional: set DPI for equation rendering
    )