import os
import requests
import json
import base64
from io import BytesIO
from PIL import Image

def get_vision_analysis(image_path: str, prompt: str) -> str:
    """Analyze an image using the Ollama vision model."""
    try:
        # Process image
        with Image.open(image_path) as img:
            # Convert to base64 for Ollama
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Get base URL and remove /v1 if present
        ollama_url = (os.getenv('OLLAMA_URL', 'http://10.0.0.29:11434')).replace('/v1', '').rstrip('/')

        # Make the vision model request
        response = requests.post(
            f"{ollama_url}/api/chat",
            json={
                'model': os.getenv('OLLAMA_VISION_MODEL', 'llama3.2-vision:11b-instruct-q8_0'),
                'messages': [{
                    'role': 'user',
                    'content': prompt,
                    'images': [encoded_image]
                }]
            },
            headers={'Content-Type': 'application/json'},
            timeout=30,
            stream=True
        )

        # Process the streaming response
        full_content = ""
        for line in response.iter_lines():
            if line:
                try:
                    json_response = json.loads(line)
                    if 'message' in json_response and 'content' in json_response['message']:
                        content = json_response['message']['content']
                        full_content += content
                        print(content, end='', flush=True)
                except json.JSONDecodeError:
                    continue
        return full_content

    except Exception as e:
        print(f"Error analyzing image: {e}")
        return str(e)

if __name__ == "__main__":
    image_path = 'screenshot_0.png'
    prompt = "What financial information can you see in this image? Please extract any stock prices, market data, news headlines, or technical indicators, p/e ratios, market cap, Beta, EPS, and any other financial data."
    
    print("Analyzing image...")
    result = get_vision_analysis(image_path, prompt)
    print("\nFinal Analysis:")
    print(result)