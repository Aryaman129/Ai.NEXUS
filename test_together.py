import os
import requests
import json

# Set API key
api_key = "4ec34405c082ae11d558aabe290486bd73ae6497fb623ba0bba481df21f5ec39"

# API endpoint
url = "https://api.together.xyz/v1/completions"

# Request payload
payload = {
    "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "prompt": "Hello, how are you?",
    "max_tokens": 20,
    "temperature": 0.7,
}

# Headers with API key
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# Make the request
print("Testing Together AI API...")
print(f"Using API key: {api_key[:8]}...{api_key[-8:]}")

try:
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()  # Raise exception for HTTP errors
    
    # Parse and print the response
    result = response.json()
    print("\nAPI call successful!")
    print(f"Status code: {response.status_code}")
    print(f"Response data: {json.dumps(result, indent=2)}")
    
except requests.exceptions.HTTPError as e:
    print(f"\nHTTP Error: {e}")
    print(f"Status code: {e.response.status_code}")
    print(f"Response text: {e.response.text}")
    
except Exception as e:
    print(f"\nError: {e}")
