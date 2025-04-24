import os
import json
import time
import requests
import base64
import mimetypes
import re
import pathlib

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


def calling_openrouter_with_image(prompt_text, image_path, api_key, model="anthropic/claude-3.7-sonnet", temperature=1, max_tokens=1024):
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


def extract_latex_from_response(response):
    """
    Extract just the LaTeX code from the LLM response.
    Handles cases where the LLM might include explanatory text.
    
    Args:
        response (str): Raw response from the LLM
        
    Returns:
        str: Extracted LaTeX code
    """
    # If the response contains $ signs, try to extract just the formula
    if '$' in response:
        # Look for formulas between $ signs
        import re
        matches = re.findall(r'\$(.*?)\$', response, re.DOTALL)
        if matches:
            return '$' + matches[0] + '$'
        
        # Look for formulas between $$ signs
        matches = re.findall(r'\$\$(.*?)\$\$', response, re.DOTALL)
        if matches:
            return '$$' + matches[0] + '$$'
    
    # If no $ signs or extraction failed, return the raw response
    # (hopefully the LLM followed instructions to output just the LaTeX)
    return response.strip()


def process_equations_with_llm(equations_json_path, api_key, model="anthropic/claude-3.7-sonnet", max_tokens=1024, output_path=None):
    """
    Process all equation images with the LLM to get the LaTeX.
    
    Args:
        equations_json_path (str): Path to the JSON file with equation info
        api_key (str): OpenRouter API key
        model (str): Model to use
        max_tokens (int): Maximum tokens for response
        output_path (str, optional): Custom path to save results
        
    Returns:
        list: Updated list of equations with LLM-generated LaTeX
    """
    # Load the equations JSON
    with open(equations_json_path, 'r', encoding='utf-8') as f:
        equations = json.load(f)
    
    # Prompt for the LLM
    prompt = "Your task is to generate the latex code of this formula. This is EDU content so there is really no room for error. Your output should be solely the latex code. Do not add any other text. the latex code should be escaped between dollar signs and compilable with MATHJAX. and again just the formula no additional comments"
    
    # Process each equation
    for i, equation in enumerate(equations):
        image_path = equation.get('cropped_image_path')
        
        if not image_path or not os.path.exists(image_path):
            print(f"[WARNING] Image not found for equation {i}: {image_path}")
            continue
        
        print(f"Processing equation {i+1}/{len(equations)}: {os.path.basename(image_path)}")
        
        # Call the LLM to get the LaTeX
        llm_response = calling_openrouter_with_image(prompt, image_path, api_key, model, max_tokens=max_tokens)
        
        # Extract just the LaTeX code (assuming the LLM properly follows instructions)
        latex_code = extract_latex_from_response(llm_response)
        
        # Store the LLM-generated LaTeX
        equation['llm_latex'] = latex_code
        
        # Add a short delay to avoid rate limiting
        time.sleep(1)
    
    # Save the updated equations JSON
    if output_path is None:
        output_path = os.path.splitext(equations_json_path)[0] + '_with_llm.json'
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(equations, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Processed {len(equations)} equations with the LLM")
    print(f"✅ Updated equations saved to: {output_path}")
    
    return equations


def update_middle_json_with_llm_latex(middle_json_path, equations, output_path=None):
    """
    Update the middle JSON file with LLM-generated LaTeX.
    
    Args:
        middle_json_path (str): Path to the middle JSON file
        equations (list): List of equations with LLM-generated LaTeX
        output_path (str, optional): Path to save the updated JSON file. 
                                     If None, original file will be updated.
    """
    # Load the middle JSON file
    with open(middle_json_path, 'r', encoding='utf-8') as f:
        middle_json = json.load(f)
    
    # Create a lookup dictionary for faster access
    equation_lookup = {}
    for eq in equations:
        page_idx = eq.get('page_idx')
        bbox = tuple(eq.get('bbox', []))  # Convert list to tuple for dict key
        if page_idx is not None and bbox:
            equation_lookup[(page_idx, bbox)] = eq
    
    updates_count = 0
    
    # Process each page
    for page in middle_json.get("pdf_info", []):
        page_idx = page["page_idx"]
        
        # Process all blocks (including preproc_blocks and para_blocks)
        for block_type in ["preproc_blocks", "para_blocks"]:
            for block in page.get(block_type, []):
                # Process each line in the block
                for line in block.get("lines", []):
                    # Process each span in the line
                    for span in line.get("spans", []):
                        # Check if the span is an inline equation
                        if span.get("type") == "inline_equation":
                            bbox = tuple(span.get("bbox", []))
                            
                            # Look up the equation
                            eq = equation_lookup.get((page_idx, bbox))
                            if eq and 'llm_latex' in eq:
                                # Update the content with LLM-generated LaTeX
                                original_content = span.get("content", "")
                                span["content"] = eq["llm_latex"].strip('$')  # Remove $ signs as they're added elsewhere
                                span["original_content"] = original_content  # Store original for reference
                                updates_count += 1
    
    # Determine output path
    if output_path is None:
        output_path = middle_json_path
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the updated middle JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(middle_json, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Updated {updates_count} equations in the middle JSON")
    print(f"✅ Updated middle JSON saved to: {output_path}")
    
    return middle_json


def update_content_json_with_llm_latex(content_json_path, equations_with_llm, output_path=None):
    """
    Update the content JSON file with LLM-generated LaTeX.
    This is more complex as formulas are embedded in text.
    
    Args:
        content_json_path (str): Path to the content JSON file
        equations_with_llm (list): List of equations with LLM-generated LaTeX
        output_path (str, optional): Path to save the updated JSON file.
                                     If None, original file will be updated.
    """
    # Load the content JSON file
    with open(content_json_path, 'r', encoding='utf-8') as f:
        content_json = json.load(f)
    
    # Create a dictionary of substitutions for each page
    substitutions_by_page = {}
    for eq in equations_with_llm:
        if 'llm_latex' in eq and 'content' in eq:
            page_idx = eq.get('page_idx')
            original = eq.get('content')
            replacement = eq.get('llm_latex')
            
            if page_idx not in substitutions_by_page:
                substitutions_by_page[page_idx] = []
            
            # Handle case where original might not have $ signs
            if not original.startswith('$'):
                original = '$' + original + '$'
            
            substitutions_by_page[page_idx].append((original, replacement))
    
    updates_count = 0
    
    # Process each item in the content list
    for item in content_json:
        page_idx = item.get('page_idx')
        
        # Only process text items on pages where we have substitutions
        if item.get('type') == 'text' and page_idx in substitutions_by_page:
            text_content = item.get('text', '')
            
            # Apply all substitutions for this page
            for original, replacement in substitutions_by_page[page_idx]:
                if original in text_content:
                    text_content = text_content.replace(original, replacement)
                    updates_count += 1
            
            # Update the item text
            item['text'] = text_content
    
    # Determine output path
    if output_path is None:
        output_path = content_json_path
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the updated content JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(content_json, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Made {updates_count} equation updates in the content JSON")
    print(f"✅ Updated content JSON saved to: {output_path}")
    
    return content_json


if __name__ == "__main__":
    # Example usage
    equations_json_path = "output/document/hatier_test_page_6/equations/inline_equations.json"
    middle_json_path = "output/hatier_test_page_6/auto/hatier_test_middle.json"
    content_json_path = "output/hatier_test_page_6/auto/hatier_test_content_list.json"
    
    # Replace with your actual OpenRouter API key
    api_key = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    
    # Specify the model to use
    model = "qwen/qwen2.5-coder-7b-instruct"  # Change as needed
    
    # Create model-specific output directory
    model_folder = get_model_folder_name(model)
    model_output_dir = f"output/model/{model_folder}"
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Update output paths to use model folder
    updated_equations_path = os.path.join(model_output_dir, "inline_equations_with_llm.json")
    middle_json_output_path = os.path.join(model_output_dir, "hatier_test_middle_updated.json")
    content_json_output_path = os.path.join(model_output_dir, "hatier_test_content_list_updated.json")
    
    # Step 1: Process equations with LLM
    equations_with_llm = process_equations_with_llm(
        equations_json_path, 
        api_key, 
        model=model, 
        max_tokens=1024,
        output_path=updated_equations_path
    )
    
    # Step 2: Update middle JSON with LLM-generated LaTeX
    update_middle_json_with_llm_latex(
        middle_json_path, 
        equations_with_llm, 
        middle_json_output_path
    )
    
    # Step 3: Update content JSON with LLM-generated LaTeX
    update_content_json_with_llm_latex(
        content_json_path, 
        equations_with_llm, 
        content_json_output_path
    )