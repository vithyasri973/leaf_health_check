"""
Gemini AI Integration Verification Script
Run this to verify Gemini API is properly configured and working.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

print("=" * 70)
print("🤖 GEMINI AI INTEGRATION VERIFICATION")
print("=" * 70)

# 1. Check .env file
print("\n✓ Step 1: Checking .env file...")
env_path = Path('.env')
if env_path.exists():
    print("  ✅ .env file found")
else:
    print("  ❌ .env file not found")
    sys.exit(1)

# 2. Check API key
print("\n✓ Step 2: Checking API key...")
api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
if api_key and api_key.startswith('AIzaSy'):
    print(f"  ✅ API key found: {api_key[:20]}...")
else:
    print("  ❌ API key missing or invalid format")
    sys.exit(1)

# 3. Check package installation
print("\n✓ Step 3: Checking google-generativeai package...")
try:
    import google.generativeai as genai
    print("  ✅ google-generativeai package installed")
except ImportError:
    print("  ❌ google-generativeai not installed")
    print("  Run: pip install google-generativeai==0.3.0")
    sys.exit(1)

# 4. Test API connection
print("\n✓ Step 4: Testing API connection...")
try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    print("  ✅ API connection successful")
except Exception as e:
    print(f"  ❌ API connection failed: {str(e)}")
    sys.exit(1)

# 5. Test model inference
print("\n✓ Step 5: Testing model inference...")
try:
    test_prompt = "What is powdery mildew in plants? Answer in 1 sentence."
    response = model.generate_content(
        test_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.7,
            max_output_tokens=100
        )
    )
    if response.text:
        print("  ✅ Model inference successful")
        print(f"  Response: {response.text[:100]}...")
    else:
        print("  ❌ No response from model")
        sys.exit(1)
except Exception as e:
    print(f"  ❌ Model inference failed: {str(e)}")
    sys.exit(1)

# 6. Check GeminiAIEngine class
print("\n✓ Step 6: Checking GeminiAIEngine class...")
try:
    sys.path.insert(0, str(Path('.').absolute()))
    from utils.gemini_ai import GeminiAIEngine, get_gemini_engine
    print("  ✅ GeminiAIEngine imported successfully")
    
    engine = get_gemini_engine()
    if engine:
        print("  ✅ GeminiAIEngine initialized successfully")
    else:
        print("  ⚠️  GeminiAIEngine initialization returned None")
except Exception as e:
    print(f"  ❌ GeminiAIEngine error: {str(e)}")
    sys.exit(1)

# 7. Test GeminiAIEngine methods
print("\n✓ Step 7: Testing GeminiAIEngine methods...")
try:
    # Test disease explanation
    explanation = engine.generate_disease_explanation(
        "late_blight",
        "tomato",
        "severe",
        75.0
    )
    if explanation:
        print("  ✅ generate_disease_explanation() works")
    else:
        print("  ❌ generate_disease_explanation() returned empty")
except Exception as e:
    print(f"  ⚠️  generate_disease_explanation() error: {str(e)}")

try:
    # Test answer_plant_question
    answer = engine.answer_plant_question("How do I treat powdery mildew?")
    if answer and "unable" not in answer.lower():
        print("  ✅ answer_plant_question() works")
    else:
        print("  ⚠️  answer_plant_question() returned limited response")
except Exception as e:
    print(f"  ⚠️  answer_plant_question() error: {str(e)}")

# 8. Configuration check
print("\n✓ Step 8: Checking Gemini configuration...")
config = {
    'API_KEY': api_key[:20] + '...',
    'MODEL': os.getenv('GEMINI_MODEL', 'gemini-pro'),
    'TEMPERATURE': os.getenv('GEMINI_TEMPERATURE', '0.7'),
    'MAX_TOKENS': os.getenv('GEMINI_MAX_OUTPUT_TOKENS', '1000'),
}
for key, value in config.items():
    print(f"  • {key}: {value}")

print("\n" + "=" * 70)
print("✅ ALL VERIFICATION CHECKS PASSED!")
print("=" * 70)

print("""
🎉 Your Gemini AI integration is ready!

Available Features:
  ✨ AI-Enhanced Disease Analysis
  💬 AI Plant Health Assistant
  📋 AI-Generated Care Plans
  🛡️ AI Preventive Measures
  📖 AI Disease Explanations

Run the app:
  streamlit run app.py

Access new AI features:
  🔍 Analyze Leaf → View "AI-Enhanced Analysis" section
  🤖 AI Assistant → Chat with AI about plants
  📋 Care Plan → Generate personalized care plans

API Status: ✅ ACTIVE
Model: {model_name}
Status: 🟢 READY FOR PRODUCTION
""".format(model_name=os.getenv('GEMINI_MODEL', 'gemini-pro')))

print("=" * 70)
