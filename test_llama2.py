#!/usr/bin/env python3
"""
Test script for Llama 2 integration.
This script tests the AI service with sample bullet points.
"""
import asyncio
import os
from ai_service import AIService

async def test_llama2_integration():
    """Test the Llama 2 integration with sample data."""
    print("Testing Llama 2 integration...")
    print("=" * 50)
    
    # Sample bullet points for testing
    test_bullet_points = [
        "5 years of experience in Python development",
        "Led a team of 8 developers on multiple projects",
        "Expert in Django, Flask, and FastAPI frameworks",
        "Implemented CI/CD pipelines using Jenkins and Docker",
        "Strong background in machine learning and data analysis"
    ]
    
    print("Sample bullet points:")
    for i, point in enumerate(test_bullet_points, 1):
        print(f"{i}. {point}")
    print()
    
    try:
        # Initialize AI service
        print("Initializing AI service...")
        ai_service = AIService()
        
        print(f"Using {'local Llama 2' if ai_service.use_local_model else 'Hugging Face API'}")
        print()
        
        # Generate summary
        print("Generating summary...")
        summary = await ai_service.generate_summary(test_bullet_points)
        
        print("Generated Summary:")
        print("-" * 30)
        print(summary)
        print("-" * 30)
        print()
        
        print("✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have installed the required dependencies:")
        print("   pip install -r requirements.txt")
        print("2. Check your .env file configuration")
        print("3. For local Llama 2, ensure you have enough memory (8GB+ recommended)")
        print("4. For Hugging Face API, ensure you have a valid API key")

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run the test
    asyncio.run(test_llama2_integration())
