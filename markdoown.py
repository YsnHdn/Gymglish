import json
import os

def generate_markdown_with_katex(json_path, output_path, image_base_path=None):
    """
    Generate a Markdown file with Katex support from the enriched JSON.
    
    Args:
        json_path (str): Path to the enriched JSON file
        output_path (str): Path to save the generated Markdown file
        image_base_path (str, optional): Base path to prepend to image paths
    """
    # Load the JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        content_list = json.load(f)
    
    # Initialize the Markdown content
    markdown_lines = []
    
    # Process each item in the content list
    for item in content_list:
        item_type = item.get('type')
        
        if item_type == 'text':
            # Handle text elements
            text_content = item.get('text', '')
            text_level = item.get('text_level')
            
            # Process all text regardless of content
            if text_level is not None:
                # This is a heading
                heading_marker = '#' * int(text_level)  # Ensure text_level is treated as int
                markdown_lines.append(f"{heading_marker} {text_content}")
            else:
                # This is regular text
                markdown_lines.append(text_content)
            
            # Always add an empty line after any text element
            markdown_lines.append("")
        
        elif item_type == 'equation':
            # Handle equation elements
            equation_content = item.get('text', '')
            
            # Add the equation as is (assuming it already has $$ delimiters)
            markdown_lines.append(equation_content)
            markdown_lines.append("")  # Add empty line after equation
        
        elif item_type == 'image':
            # Handle image elements
            img_path = item.get('img_path', '')
            
            # Adjust image path if base path is provided
            if image_base_path and img_path and not os.path.isabs(img_path):
                img_path = os.path.join(image_base_path, img_path)
                # Convert backslashes to forward slashes for markdown
                img_path = img_path.replace('\\', '/')
            
            # Handle image captions
            caption_text = ""
            if 'img_caption' in item and item['img_caption']:
                for caption in item['img_caption']:
                    if isinstance(caption, str):
                        caption_text += caption
                    elif isinstance(caption, dict) and 'text' in caption:
                        caption_text += caption['text']
            
            # Add the image with caption as alt text
            if img_path:
                markdown_lines.append(f"![{caption_text}]({img_path})")
                
                # If there's a caption, display it below the image as well
                if caption_text:
                    markdown_lines.append(f"{caption_text}")
                
                markdown_lines.append("")  # Add empty line after image
            
            # Handle image footnotes
            if 'img_footnote' in item and item['img_footnote']:
                for footnote in item['img_footnote']:
                    if isinstance(footnote, str):
                        markdown_lines.append(f"<small>*{footnote}*</small>")
                    elif isinstance(footnote, dict) and 'text' in footnote:
                        markdown_lines.append(f"<small>*{footnote['text']}*</small>")
                if item['img_footnote']:  # Only add empty line if there were footnotes
                    markdown_lines.append("")
        
        elif item_type == 'table':
            # Handle table elements
            table_body = item.get('table_body', '')
            
            # Handle table captions
            if 'table_caption' in item and item['table_caption']:
                caption_text = ""
                for caption in item['table_caption']:
                    if isinstance(caption, str):
                        caption_text += caption
                    elif isinstance(caption, dict) and 'text' in caption:
                        caption_text += caption['text']
                if caption_text:
                    markdown_lines.append(f"**Table: {caption_text}**")
                    markdown_lines.append("")  # Add empty line after caption
            
            # Add the table content
            if table_body:
                markdown_lines.append(table_body)
                markdown_lines.append("")  # Add empty line after table
            
            # Handle table footnotes
            if 'table_footnote' in item and item['table_footnote']:
                for footnote in item['table_footnote']:
                    if isinstance(footnote, str):
                        markdown_lines.append(f"<small>*{footnote}*</small>")
                    elif isinstance(footnote, dict) and 'text' in footnote:
                        markdown_lines.append(f"<small>*{footnote['text']}*</small>")
                if item['table_footnote']:  # Only add empty line if there were footnotes
                    markdown_lines.append("")
    
    # Write the Markdown content to the output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(markdown_lines))
    
    print(f"✅ Markdown file with Katex support generated: {output_path}")


if __name__ == "__main__":
    # Example usage
    json_path = "output/model/qwen/hatier_test_content_list_updated.json"
    output_path = "output/model/qwen/qwen.md"

    
    generate_markdown_with_katex(json_path, output_path, image_base_path)