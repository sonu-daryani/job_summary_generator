#!/usr/bin/env python3
"""
Test script to verify Hugging Face API key permissions.
"""
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_huggingface_api():
    """Test if the Hugging Face API key works."""
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    
    if not api_key:
        print("❌ No API key found in .env file")
        return False
    
    print(f"🔑 Testing API key: {api_key[:10]}...")
    
    # Test with a simple model
    url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": "Hello, how are you?",
        "parameters": {
            "max_new_tokens": 50,
            "temperature": 0.7
        }
    }
    
    try:
        print("🚀 Testing API connection...")
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ API key is working!")
            result = response.json()
            print(f"📝 Sample response: {result}")
            return True
        elif response.status_code == 401:
            print("❌ Invalid API key - check your token")
            return False
        elif response.status_code == 403:
            print("❌ Insufficient permissions - create a new token with 'Read' access")
            return False
        elif response.status_code == 404:
            print("⚠️  Model not found - but API key is valid")
            return True
        else:
            print(f"❌ API error: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ Request timeout - API might be slow")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Hugging Face API Key...")
    print("=" * 50)
    
    success = test_huggingface_api()
    
    print("=" * 50)
    if success:
        print("🎉 API key is working! Your application should work now.")
    else:
        print("💡 Please check your API key and try again.")
        print("   Get a new token at: https://huggingface.co/settings/tokens")
