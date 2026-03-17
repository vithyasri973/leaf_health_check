# Google Gemini AI Integration Guide

## 🔑 API Key Setup

Your API key has been provided:
```
AIzaSyCbFF4dKddz8XtAvnLvOCo2cgUawcAkJwU
```

### Environment Configuration

1. **Create `.env` file** (copy from `.env.example`):
   ```bash
   cp .env.example .env
   ```

2. **Add your API key to `.env`:**
   ```env
   GOOGLE_GEMINI_API_KEY=AIzaSyCbFF4dKddz8XtAvnLvOCo2cgUawcAkJwU
   GEMINI_MODEL=gemini-pro
   GEMINI_TEMPERATURE=0.7
   GEMINI_MAX_OUTPUT_TOKENS=1000
   ```

3. **Install required package:**
   ```bash
   pip install google-generativeai==0.3.0
   ```

---

## ✨ Features Enabled

### 1. **AI-Enhanced Disease Explanation**
When you analyze a leaf image, Gemini AI will generate:
- Detailed disease overview
- Causes and transmission methods
- Why it's at the current severity level
- Risk assessment

### 2. **Personalized Rescue Tips**
AI generates 5 prioritized tips based on:
- Specific disease characteristics
- Severity level
- Plant species
- Affected area percentage
- Affordability and practicality

### 3. **AI Plant Health Assistant** (Mode: 🤖)
Chat with AI about:
- Disease symptoms and treatments
- Plant care guidance
- Prevention strategies
- Product recommendations
- When to seek expert help

**Example Questions:**
- "How do I prevent late blight in my tomato plants?"
- "What's the difference between early and late blight?"
- "Can I use sulfur spray on my apple trees?"
- "How often should I water during disease treatment?"

### 4. **Preventive Measures**
Automatically generates climate-specific prevention strategies for:
- Spacing and air circulation
- Watering practices
- Seasonal timing
- Companion planting
- Hygiene protocols

### 5. **Personalized Care Plans** (Mode: 📋)
Generate 7-day care plans including:
- Day-by-day actions
- Environmental conditions
- Product dosages and timing
- Monitoring checkpoints
- Professional escalation triggers

---

## 🚀 Usage Examples

### Example 1: Analyze a Leaf with AI Insights
```
1. Go to "🔍 Analyze Leaf"
2. Upload leaf image
3. Click "Analyze Leaf"
4. View AI-Enhanced Analysis section
   - Disease Overview (from Gemini)
   - AI-Personalized Tips (from Gemini)
   - Preventive Measures (from Gemini)
```

### Example 2: Ask AI Questions
```
1. Go to "🤖 AI Assistant"
2. Type: "My tomato plants have yellow spots. What should I do?"
3. AI responds with:
   - Likely disease identification
   - Treatment steps
   - Prevention advice
   - When to escalate
```

### Example 3: Generate Care Plan
```
1. Go to "📋 Care Plan"
2. Select: Tomato, Late Blight, Severe, Tropical
3. Click "Generate Care Plan"
4. Download as TXT file
```

---

## 📊 API Model Comparison

| Feature | Gemini-Pro |
|---------|-----------|
| **Cost** | Free tier available |
| **Speed** | ~1-3 seconds |
| **Quality** | High accuracy for agricultural content |
| **Context Window** | 30K tokens |
| **Max Output** | 1000 tokens (configurable) |
| **Languages** | 100+ languages |
| **Availability** | Global |

---

## ⚙️ Configuration Options

### Temperature (Creativity)
```env
GEMINI_TEMPERATURE=0.7  # Default: balanced
# 0.0 = Deterministic, precise answers
# 0.5 = Moderate creativity
# 1.0 = Maximum creativity/randomness
```

### Max Output Tokens
```env
GEMINI_MAX_OUTPUT_TOKENS=1000  # Default
# Disease explanation: 200 tokens
# Tips generation: 500 tokens
# Care plan: 1500 tokens
```

---

## 🔒 Security Best Practices

### DO:
✅ Store API key in `.env` file only  
✅ Add `.env` to `.gitignore`  
✅ Use environment variables in production  
✅ Monitor API usage for unusual patterns  

### DON'T:
❌ Commit `.env` to version control  
❌ Share API key in public issues/forums  
❌ Embed key directly in code  
❌ Use same key across multiple apps  

---

## 🐛 Troubleshooting

### Issue: "Gemini AI initialization warning"
```
Error: Gemini AI initialization warning
```
**Solution:** Check API key in `.env`
```bash
# Verify key is correct
echo $GOOGLE_GEMINI_API_KEY

# Reinstall package
pip install --upgrade google-generativeai
```

### Issue: "API key invalid"
**Solution:** 
1. Verify key format: `AIzaSy...` (starts with AIzaSy)
2. Check .env file is in project root
3. Restart Streamlit: `streamlit run app.py`

### Issue: "Rate limit exceeded"
**Solution:** Gemini free tier has quotas
- Free: ~1000 requests/minute
- Upgrade plan for higher limits
- Contact: https://ai.google.dev/

### Issue: "No response generated"
**Solution:** 
1. Check internet connection
2. Verify API key is active
3. Try alternative model if available

---

## 📈 Monitoring Usage

### Check API Usage:
1. Go to: https://ai.google.dev/
2. Navigate to: Credentials → API quotas
3. Monitor requests per minute

### Log API Calls (in app.py):
```python
import logging
logging.basicConfig(level=logging.DEBUG)
# Now Gemini calls will log detailed info
```

---

## 🔄 Alternative Models (Future)

If Gemini becomes unavailable, consider:
- **GPT-4** (via OpenAI API)
- **Claude** (via Anthropic API)
- **Llama 2** (open source, self-hosted)
- **Falcon** (open source)

To switch models, update:
```env
GEMINI_MODEL=gemini-pro-vision  # For vision tasks
# or
GEMINI_MODEL=custom-model-name
```

---

## 📚 Resources

- **Google AI Studio:** https://ai.google.dev
- **Gemini Documentation:** https://ai.google.dev/tutorials
- **API Reference:** https://ai.google.dev/api/python
- **Pricing:** https://ai.google.dev/pricing
- **Issues:** https://github.com/google/generative-ai-python/issues

---

## 📞 Support

**For Leaf Health Check Issues:**
- Check this guide first
- Review logs: `streamlit run app.py --logger.level=debug`
- GitHub Issues: [project repo]

**For Gemini API Issues:**
- Google AI Support: https://ai.google.dev/support
- API Status: https://status.cloud.google.com

---

## ✅ Verification Checklist

- [ ] API key added to `.env`
- [ ] `google-generativeai` package installed
- [ ] `.env` file in project root
- [ ] `.env` added to `.gitignore`
- [ ] Streamlit app runs without errors
- [ ] AI Assistant responds to questions
- [ ] Care plan generates successfully
- [ ] Disease explanations appear in analysis

---

**Last Updated:** January 2026  
**Status:** ✅ Active and Integrated

🤖 Your Leaf Health Check app is now powered by Google Gemini AI!
