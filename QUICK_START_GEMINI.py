"""
QUICK START GUIDE - Leaf Health Check with Gemini AI Integration
"""

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║           🍃 LEAF HEALTH CHECK - GEMINI AI INTEGRATION GUIDE 🤖           ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════════════════
📋 QUICK START (5 MINUTES)
═══════════════════════════════════════════════════════════════════════════════

1️⃣  INSTALL DEPENDENCIES
    pip install -r requirements.txt

2️⃣  VERIFY .env CONFIGURATION
    • Check file: .env (should exist in project root)
    • Verify API Key: AIzaSyCbFF4dKddz8XtAvnLvOCo2cgUawcAkJwU

3️⃣  RUN VERIFICATION SCRIPT
    python verify_gemini.py
    
    Expected output:
    ✅ ALL VERIFICATION CHECKS PASSED!

4️⃣  START THE APPLICATION
    streamlit run app.py
    
    Browser opens at: http://localhost:8501

5️⃣  TEST AI FEATURES
    a) 🔍 Analyze Leaf
       - Upload sample leaf image
       - View AI-Enhanced Analysis section
    
    b) 🤖 AI Assistant
       - Type: "How do I treat powdery mildew?"
       - Get instant AI response
    
    c) 📋 Care Plan
       - Select plant and disease
       - Generate 7-day care plan

═══════════════════════════════════════════════════════════════════════════════
🔑 API KEY INFORMATION
═══════════════════════════════════════════════════════════════════════════════

API Provider:          Google AI
Model:                 Gemini-Pro
API Key:               AIzaSyCbFF4dKddz8XtAvnLvOCo2cgUawcAkJwU
Status:                ✅ ACTIVE
Free Tier Limits:      1,000 requests/minute
Cost:                  Free (within limits)

Configuration File:    .env
  • GOOGLE_GEMINI_API_KEY=AIzaSyCbFF4dKddz8XtAvnLvOCo2cgUawcAkJwU
  • GEMINI_MODEL=gemini-pro
  • GEMINI_TEMPERATURE=0.7
  • GEMINI_MAX_OUTPUT_TOKENS=1000

═══════════════════════════════════════════════════════════════════════════════
✨ NEW FEATURES WITH GEMINI AI
═══════════════════════════════════════════════════════════════════════════════

1. AI-ENHANCED DISEASE ANALYSIS
   Location: 🔍 Analyze Leaf → Results → "AI-Enhanced Analysis"
   
   Includes:
   ✓ Detailed disease overview (what it is, how it spreads)
   ✓ AI-personalized rescue tips (5 priority-sorted recommendations)
   ✓ Climate-specific preventive measures
   ✓ Actionable treatment timeline
   
   Example:
   Disease: Late Blight (Severe)
   AI Response: "Late Blight is caused by Phytophthora infestans fungus...
   Immediate actions: 1) Remove infected leaves, 2) Apply metalaxyl..."

2. AI PLANT HEALTH ASSISTANT
   Location: 🤖 AI Assistant tab
   
   Features:
   ✓ Multi-turn conversation (remembers context)
   ✓ Answers plant health questions
   ✓ Provides treatment guidance
   ✓ Suggests preventive measures
   ✓ Recommends products and dosages
   
   Try asking:
   • "What's the difference between early and late blight?"
   • "Can I use copper fungicide on fruit?"
   • "How often should I spray during wet season?"
   • "Are there organic alternatives for fungicides?"

3. PERSONALIZED CARE PLANS
   Location: 📋 Care Plan tab
   
   Features:
   ✓ 7-day daily care schedule
   ✓ Specific spray timing and dosages
   ✓ Environmental condition recommendations
   ✓ Monitoring checkpoints
   ✓ Professional escalation triggers
   
   Process:
   1. Select plant species (Tomato, Potato, Apple, etc.)
   2. Choose disease to treat
   3. Specify severity level
   4. Select climate zone
   5. Download as text file

═══════════════════════════════════════════════════════════════════════════════
🔍 FILE STRUCTURE - GEMINI INTEGRATION
═══════════════════════════════════════════════════════════════════════════════

Project Root/
├── .env                          # ⭐ API key configuration (CRITICAL)
├── .env.example                  # Template for .env
├── requirements.txt              # Python dependencies (includes google-generativeai)
│
├── app.py                        # ✨ Updated with Gemini features
│   ├── Import: from utils.gemini_ai import get_gemini_engine
│   ├── Features: AI Assistant, Care Plan tabs
│   └── Enhanced Analysis section in results
│
├── utils/
│   ├── gemini_ai.py             # ⭐ NEW - Gemini AI Engine
│   │   ├── GeminiAIEngine class
│   │   ├── generate_disease_explanation()
│   │   ├── generate_personalized_tips()
│   │   ├── answer_plant_question()
│   │   ├── generate_care_plan()
│   │   └── identify_preventive_measures()
│   │
│   ├── preprocess.py            # Image processing (unchanged)
│   ├── severity.py              # Severity calculation (unchanged)
│   ├── recommendations.py       # Default tips (unchanged)
│   └── __init__.py              # ✨ Updated imports
│
├── GEMINI_API_SETUP.md          # ⭐ NEW - Detailed Gemini setup guide
├── verify_gemini.py             # ⭐ NEW - Verification script
└── [other files unchanged]

═══════════════════════════════════════════════════════════════════════════════
⚙️ INSTALLATION STEPS
═══════════════════════════════════════════════════════════════════════════════

STEP 1: Update Requirements
─────────────────────────
pip install -r requirements.txt

This installs:
✓ tensorflow (AI models)
✓ streamlit (web interface)
✓ opencv-python (image processing)
✓ google-generativeai (NEW - Gemini AI)
✓ python-dotenv (environment variables)
✓ [all other dependencies]

STEP 2: Verify API Key
──────────────────────
Check .env file contains:
GOOGLE_GEMINI_API_KEY=AIzaSyCbFF4dKddz8XtAvnLvOCo2cgUawcAkJwU

⚠️ SECURITY: Never commit .env to GitHub
✓ Already in .gitignore

STEP 3: Run Verification
────────────────────────
python verify_gemini.py

Output shows:
✅ .env file found
✅ API key found
✅ google-generativeai package installed
✅ API connection successful
✅ Model inference successful
✅ GeminiAIEngine imported
✅ GeminiAIEngine initialized
✅ All methods working

STEP 4: Initialize Database (first time only)
──────────────────────────────────────────────
python database/init_db.py

Creates tables:
✓ plants (plant species database)
✓ diseases (disease information)
✓ tips (default rescue recommendations)
✓ analysis_history (stores analysis records)

STEP 5: Create Pre-trained Models (first time only)
───────────────────────────────────────────────────
python model/create_models.py

Generates:
✓ leaf_disease_model.h5 (disease detection)
✓ plant_species_model.h5 (plant identification)

STEP 6: Launch Application
──────────────────────────
streamlit run app.py

Opens: http://localhost:8501

═══════════════════════════════════════════════════════════════════════════════
🧪 TESTING AI FEATURES
═══════════════════════════════════════════════════════════════════════════════

TEST 1: Verify Gemini is Running
────────────────────────────────
Run: python verify_gemini.py
Expected: ✅ ALL VERIFICATION CHECKS PASSED!

TEST 2: AI-Enhanced Disease Analysis
─────────────────────────────────────
1. Open app: streamlit run app.py
2. Go to: 🔍 Analyze Leaf
3. Upload any plant leaf image
4. Click: 🚀 Analyze Leaf
5. Scroll down to: "AI-Enhanced Analysis (Powered by Gemini)"
6. Should see:
   ✓ Disease Overview (from Gemini)
   ✓ AI-Personalized Tips
   ✓ Preventive Measures

TEST 3: AI Assistant Chat
─────────────────────────
1. Open: 🤖 AI Assistant tab
2. Type: "What is early blight?"
3. Should see AI response about early blight
4. Type: "How do I treat it?"
5. AI responds with treatment options
6. Conversation continues (multi-turn)

TEST 4: Care Plan Generation
────────────────────────────
1. Go to: 📋 Care Plan
2. Select:
   Plant: Tomato
   Disease: Late Blight
   Severity: Severe
3. Click: 📄 Generate Care Plan
4. Should see: 7-day care schedule
5. Click: 📥 Download Care Plan (TXT)

═══════════════════════════════════════════════════════════════════════════════
🔐 SECURITY & BEST PRACTICES
═══════════════════════════════════════════════════════════════════════════════

✅ DO:
   • Keep .env file locally only
   • Add .env to .gitignore (already done)
   • Use environment variables in production
   • Monitor API usage for abuse
   • Rotate API key if compromised

❌ DON'T:
   • Commit .env to version control
   • Share API key in public issues
   • Embed key directly in code
   • Use same key across multiple apps
   • Expose key in error messages/logs

═══════════════════════════════════════════════════════════════════════════════
🆘 TROUBLESHOOTING
═══════════════════════════════════════════════════════════════════════════════

ISSUE 1: "API key invalid"
──────────────────────────
Solution:
1. Check .env file format:
   GOOGLE_GEMINI_API_KEY=AIzaSyCbFF4dKddz8XtAvnLvOCo2cgUawcAkJwU
   
2. Ensure .env is in project root:
   ls .env  # Should exist
   
3. Restart Streamlit:
   streamlit run app.py

ISSUE 2: "ModuleNotFoundError: No module named 'google.generativeai'"
────────────────────────────────────────────────────────────────────
Solution:
pip install google-generativeai==0.3.0

ISSUE 3: "AI responses are empty or slow"
──────────────────────────────────────────
Solution:
1. Check internet connection
2. Verify API key is active
3. Check API quotas: https://ai.google.dev
4. Reduce GEMINI_MAX_OUTPUT_TOKENS in .env

ISSUE 4: "Rate limit exceeded"
──────────────────────────────
Solution:
• Free tier: ~1,000 requests/minute
• Upgrade plan: https://ai.google.dev/pricing
• Implement caching for repeated queries

═══════════════════════════════════════════════════════════════════════════════
📊 GEMINI API USAGE MONITORING
═══════════════════════════════════════════════════════════════════════════════

Monitor your API usage:
1. Visit: https://ai.google.dev
2. Sign in with your Google account
3. Navigate to: Credentials → API quotas
4. View:
   • Requests per minute
   • Daily quota usage
   • Current plan tier

Typical Usage Per Feature:
• Disease Explanation: 1 API call (~0.5 sec)
• Personalized Tips: 1 API call (~1-2 sec)
• Care Plan: 1 API call (~2-3 sec)
• Chat Message: 1 API call (~1 sec)

═══════════════════════════════════════════════════════════════════════════════
🚀 DEPLOYMENT
═══════════════════════════════════════════════════════════════════════════════

WITH DOCKER:
─────────────
docker-compose up

WITH STREAMLIT CLOUD:
─────────────────────
1. Push to GitHub
2. Go to: https://streamlit.io/cloud
3. Connect repo
4. Set secrets in Streamlit Cloud:
   GOOGLE_GEMINI_API_KEY=AIzaSyCbFF4dKddz8XtAvnLvOCo2cgUawcAkJwU

WITH AWS/HEROKU:
────────────────
Set environment variable:
export GOOGLE_GEMINI_API_KEY=AIzaSyCbFF4dKddz8XtAvnLvOCo2cgUawcAkJwU

═══════════════════════════════════════════════════════════════════════════════
📚 ADDITIONAL RESOURCES
═══════════════════════════════════════════════════════════════════════════════

Documentation:
✓ GEMINI_API_SETUP.md - Complete Gemini setup guide
✓ README.md - Full project documentation
✓ verify_gemini.py - Verification and testing

Official Links:
✓ Google AI: https://ai.google.dev
✓ Gemini Docs: https://ai.google.dev/tutorials
✓ Python API: https://github.com/google/generative-ai-python
✓ Pricing: https://ai.google.dev/pricing

═══════════════════════════════════════════════════════════════════════════════
✅ VERIFICATION CHECKLIST
═══════════════════════════════════════════════════════════════════════════════

Setup:
  ☐ .env file exists in project root
  ☐ API key in .env: AIzaSyCbFF4dKddz8XtAvnLvOCo2cgUawcAkJwU
  ☐ .env added to .gitignore
  ☐ Requirements installed: pip install -r requirements.txt

Verification:
  ☐ Verify script passes: python verify_gemini.py
  ☐ Database initialized: python database/init_db.py
  ☐ Models created: python model/create_models.py
  ☐ App starts: streamlit run app.py

Features:
  ☐ ✨ AI-Enhanced Analysis in disease results
  ☐ 🤖 AI Assistant responds to questions
  ☐ 📋 Care Plan generates successfully
  ☐ 🛡️ Preventive measures are AI-generated
  ☐ 📖 Disease explanations appear automatically

═══════════════════════════════════════════════════════════════════════════════

🎉 READY TO USE!

Your Leaf Health Check application is now fully integrated with Google Gemini AI.

Start the app:
    streamlit run app.py

All new features are ready:
    ✨ AI-Enhanced Disease Analysis
    💬 AI Plant Health Assistant  
    📋 AI-Generated Care Plans
    🛡️ AI Preventive Measures

Questions? Check GEMINI_API_SETUP.md

═══════════════════════════════════════════════════════════════════════════════
""")
