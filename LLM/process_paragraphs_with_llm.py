import os
import json
import time
import requests
import base64
import mimetypes
import re

def get_model_folder_name(model):
    """
    Extract the vendor/provider name from the model string.
    
    Args:
        model (str): Full model name like "anthropic/claude-3.7-sonnet"
        
    Returns:
        str: Folder name based on vendor (e.g., "anthropic")
    """
    # Split by slash and take the first part
    # For "anthropic/claude-3.7-sonnet" it will return "anthropic"
    if '/' in model:
        return model.split('/')[0]
    # Fallback to a safe default if no slash in the name
    return "default_model"

def encode_image_to_base64(path):
    """
    Reads a local image file and returns a base64-encoded data URL.
    """
    mime_type, _ = mimetypes.guess_type(path)
    if not mime_type:
        raise ValueError(f"Could not determine MIME type for file: {path}")
    
    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded_string}"

def calling_openrouter_with_image(prompt_text, image_path, api_key, model="anthropic/claude-3.7-sonnet", temperature=1, max_tokens=4096):
    """
    Sends a text prompt and a local image to a multimodal-compatible model on OpenRouter.
    """
    image_data_url = encode_image_to_base64(image_path)

    while True:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            data=json.dumps({
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_data_url
                                }
                            }
                        ]
                    }
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            })
        )

        try:
            response_json = response.json()

            if response_json.get("error", {}).get("code") == 502 and response_json.get("error", {}).get("message") == "Overloaded":
                print("Server is overloaded. Retrying in 10 seconds...")
                time.sleep(10)
                continue

            if 'choices' in response_json and len(response_json['choices']) > 0:
                return response_json['choices'][0]['message']['content']
            else:
                print("[ERROR] Unexpected response structure:", response_json)
                return "Error: Unexpected response structure."

        except json.JSONDecodeError:
            print("Failed to decode JSON response:", response.text)
            return "Error: Failed to decode JSON response."

def process_paragraph_images_with_llm(content_with_crops_path, api_key, model="anthropic/claude-3.7-sonnet", output_base_dir="output"):
    """
    Process all types of images with the LLM to get accurately transcribed text with correct LaTeX.
    
    Args:
        content_with_crops_path (str): Path to the JSON file with paragraph crops
        api_key (str): OpenRouter API key
        model (str): Model to use
        output_base_dir (str): Base directory for output
    """
    # Load the content with crops JSON
    with open(content_with_crops_path, 'r', encoding='utf-8') as f:
        content_list = json.load(f)
    
    # Define prompts for different content types
    type_prompts = {
        'title': """
        Transcribe the title in this image exactly as it appears. 
        Pay attention to capitalization, special characters, and formatting.
        This is for educational content, so accurate transcription and LaTeX are critical.Do not add any explanations or comments.
        """,
        
        'text': """
        Your task is to transcribe the text in this image EXACTLY as it appears, with one specific improvement: 
        ensure all mathematical formulas are properly formatted in LaTeX

        Rules:
        1. Return the exact text you see in the image
        2. Format all mathematical expressions with proper LaTeX between $ signs
        3. Keep all non-mathematical text identical to the image
        4. Use inline formulas (single $) for expressions within sentences
        5. Use block formulas (double $$) for displayed equations
        6. Do not add any explanations or comments

        This is for educational content, so accurate transcription and LaTeX are critical.Do not add any explanations or comments.
        """,
        
        'list': """
        Transcribe this list exactly as it appears, maintaining:
        - List structure
        - Bullet points or numbering
        - Original wording
        - Any mathematical expressions (use LaTeX formatting)
        
        Ensure that mathematical expressions are correctly formatted:
        - Inline formulas: use single $ 
        - Block formulas: use double $$

        This is for educational content, so accurate transcription and LaTeX are critical.Do not add any explanations or comments.
        """,
        
        'inline_equation': """
        Transcribe this mathematical expression exactly, ensuring:
        - Precise LaTeX formatting
        - Exact representation of the original image
        - Correct mathematical notation
        
        Use single $ for inline formula formatting.

        This is for educational content, so accurate transcription and LaTeX are critical.Do not add any explanations or comments.
        """,
        
        'interline_equation': """
        Transcribe this mathematical expression exactly, ensuring:
        - Precise LaTeX formatting
        - Exact representation of the original image
        - Correct mathematical notation
        
        Use double $$ for block formula formatting.

        This is for educational content, so accurate transcription and LaTeX are critical.Do not add any explanations or comments.
        """,
        
        'image': """
        Describe the contents of this image precisely:
        - Provide a detailed, objective description
        - Note any text, diagrams, or key visual elements
        - If it's a graph, chart, or mathematical visualization, describe its key components

        This is for educational content, so accurate transcription and LaTeX are critical.
        Do not add any explanations or comments.
        """
    }
    
    # Track which items have been processed
    processed_count = 0
    
    # Create model-specific output directory
    model_folder = get_model_folder_name(model)
    model_output_dir = os.path.join(output_base_dir, model_folder)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Process each item in the content list
    for i, item in enumerate(content_list):
        # Only process items that have a cropped image path
        if 'cropped_image_path' in item:
            image_path = item.get('cropped_image_path')
            item_type = item.get('type', 'unknown')
            
            if not os.path.exists(image_path):
                print(f"[WARNING] Image not found for item {i}: {image_path}")
                continue
            
            # Get the appropriate prompt based on item type
            prompt = type_prompts.get(item_type, """
            Transcribe the contents of this image exactly as it appears.
            Preserve all original formatting and characteristics.
            """)
            
            print(f"Processing {item_type} {processed_count+1}: {os.path.basename(image_path)}")
            
            # Call the LLM to get the text with corrected LaTeX
            llm_response = calling_openrouter_with_image(prompt, image_path, api_key, model)
            
            # Store the original text and the LLM-corrected text
            item['original_content'] = item.get('content', '')
            item['content'] = llm_response.strip()
            
            processed_count += 1
            
            # Add a short delay to avoid rate limiting
            time.sleep(1)
    
    # Save the updated content list
    updated_json_path = os.path.join(model_output_dir, "content_with_corrected_latex.json")
    
    with open(updated_json_path, 'w', encoding='utf-8') as f:
        json.dump(content_list, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Processed {processed_count} items with the LLM")
    print(f"✅ Updated content saved to: {updated_json_path}")
    
    return content_list

def generate_markdown_with_katex(corrected_json_path, output_path):
    """
    Generate a Markdown file with Katex support from the corrected JSON.
    
    Args:
        corrected_json_path (str): Path to the JSON file with corrected content
        output_path (str): Path to save the generated Markdown file
    """
    # Load the JSON file
    with open(corrected_json_path, 'r', encoding='utf-8') as f:
        content_list = json.load(f)
    
    # Initialize the Markdown content
    markdown_lines = []
    
    # Process each item in the content list
    for item in content_list:
        item_type = item.get('type')
        
        if item_type in ['text', 'title', 'list']:
            # Handle text elements
            text_content = item.get('content', '')
            
            # Add content based on type
            if item_type == 'title':
                # This is a heading (use level based on item)
                heading_level = item.get('text_level', 1)  # Default to level 1 if not specified
                heading_marker = '#' * int(heading_level)
                markdown_lines.append(f"{heading_marker} {text_content}")
            else:
                # Regular text or list
                markdown_lines.append(text_content)
            
            # Always add an empty line after any text element
            markdown_lines.append("")
        
        elif item_type in ['inline_equation', 'interline_equation']:
            # Handle equation elements
            equation_content = item.get('content', '')
            
            # Add the equation as is (assuming it already has $ or $$ delimiters)
            markdown_lines.append(equation_content)
            markdown_lines.append("")  # Add empty line after equation
        
        elif item_type == 'image':
            # Handle image elements
            img_path = item.get('cropped_image_path', '')
            
            # Add the image without caption as alt text (just empty alt)
            if img_path:
                markdown_lines.append(f"![]({img_path})")
                markdown_lines.append("")  # Add empty line after image
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write the Markdown content to the output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(markdown_lines))
    
    print(f"✅ Markdown file with Katex support generated: {output_path}")
    
    return output_path

if __name__ == "__main__":
    # Example usage
    content_with_crops_path = "output/document/hatier_test_page_6/content/all_content.json"
    
    # Replace with your actual OpenRouter API key
    api_key = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    
    # Specify the model to use
    model = "anthropic/claude-3.7-sonnet"  # Change as needed
    
    # Step 1: Process paragraph images with LLM to correct LaTeX
    corrected_content = process_paragraph_images_with_llm(
        content_with_crops_path, 
        api_key, 
        model=model
    )
    
    # Get the path to the corrected JSON
    model_folder = get_model_folder_name(model)
    corrected_json_path = f"output/{model_folder}/content_with_corrected_latex.json"
    
    # Step 2: Generate Markdown with Katex from the corrected content
    markdown_output_path = f"output/{model_folder}/document_with_corrected_latex.md"
    generate_markdown_with_katex(corrected_json_path, markdown_output_path)