"""
Leaf Health Check - Main Streamlit Application
AI-powered plant leaf disease detection and diagnosis system.

Deploy: streamlit run app.py
"""

import streamlit as st
import numpy as np
import cv2
from pathlib import Path
import logging
from datetime import datetime
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.preprocess import ImagePreprocessor
from utils.severity import SeverityGrader
from utils.recommendations import RecommendationEngine
from utils.gemini_ai import get_gemini_engine
from model.train import PlantDiseaseModel
from database.init_db import init_database, get_connection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🍃 Leaf Health Check",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f0f8f5; }
    .stButton>button { width: 100%; background-color: #2ecc71; color: white; }
    .disease-card { background: white; padding: 20px; border-radius: 10px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); color: #333333; }
    .severity-badge { padding: 10px 20px; border-radius: 20px; color: white; font-weight: bold; display: inline-block; }
    .stats-metric { text-align: center; padding: 15px; background: white; border-radius: 10px; margin: 10px; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'model' not in st.session_state:
    st.session_state.model = None
if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = False
if 'gemini_engine' not in st.session_state:
    st.session_state.gemini_engine = get_gemini_engine()
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []


@st.cache_resource
def load_model():
    """Load pre-trained model (cached)."""
    try:
        model = PlantDiseaseModel(architecture='efficientnet')
        model_path = Path('model')
        
        # Try to load existing models
        disease_loaded = model.load_model('disease', str(model_path))
        plant_loaded = model.load_model('plant', str(model_path))
        
        if not disease_loaded or not plant_loaded:
            # Build models if not found
            if not disease_loaded:
                logger.info("Building disease model...")
                model.build_disease_model()
            if not plant_loaded:
                logger.info("Building plant model...")
                model.build_plant_model()
        
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None


@st.cache_resource
def init_db():
    """Initialize database (cached)."""
    try:
        db_path = Path('database/plants.db')
        db_path.parent.mkdir(exist_ok=True)
        
        # Initialize if doesn't exist
        if not db_path.exists():
            init_database()
        
        return str(db_path)
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        return None


def analyze_leaf_image(image_path):
    """
    Analyze leaf image and return diagnosis.
    
    Args:
        image_path: Path to uploaded image
        
    Returns:
        dict: Analysis results
    """
    try:
        # Load and preprocess image
        image = ImagePreprocessor.load_image(image_path)
        original_image = image.copy()
        
        # Detect discoloration
        discoloration_data = ImagePreprocessor.detect_discoloration(image)
        
        # Preprocess for model
        processed_image = ImagePreprocessor.preprocess_for_model(image)
        
        # Get model predictions
        st.session_state.model = load_model()
        disease_result = st.session_state.model.predict_disease(processed_image)
        plant_result = st.session_state.model.predict_plant(processed_image)
        
        # Calculate severity
        severity_result = SeverityGrader.calculate_severity(
            discoloration_data,
            disease_result['disease'],
            disease_result['confidence']
        )
        
        # Get recommendations
        db_path = init_db()
        recommendations = RecommendationEngine.get_recommendations(
            disease_result['disease'],
            severity_result['severity_level'],
            plant_result['plant'],
            db_path
        )
        
        # Compile results
        analysis = {
            'plant': plant_result['plant'],
            'plant_confidence': plant_result['confidence'],
            'disease': disease_result['disease'],
            'disease_confidence': disease_result['confidence'],
            'severity': severity_result['severity_level'],
            'affected_percentage': severity_result['affected_percentage'],
            'diagnosis_confidence': severity_result['diagnosis_confidence'],
            'discoloration_breakdown': severity_result['color_breakdown'],
            'recommendations': recommendations,
            'original_image': original_image,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to history and database
        st.session_state.analysis_history.append(analysis)
        
        if db_path:
            RecommendationEngine.save_analysis_history({
                'plant_name': analysis['plant'],
                'disease_name': analysis['disease'],
                'severity': analysis['severity'],
                'confidence': analysis['diagnosis_confidence'],
                'discoloration_percent': analysis['affected_percentage'],
                'image_filename': Path(image_path).name
            }, db_path)
        
        return analysis
    
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        st.error(f"❌ Error analyzing image: {str(e)}")
        return None


def display_severity_badge(severity):
    """Display severity badge."""
    badge = SeverityGrader.get_severity_badge(severity)
    st.markdown(
        f"""<div class="severity-badge" style="background-color: {badge['color']};">
            {badge['emoji']} {badge['display'].upper()}
        </div>""",
        unsafe_allow_html=True
    )
    st.caption(badge['description'])


def main():
    """Main application."""
    
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("# 🍃 Leaf Health Check")
        st.markdown("### AI-Powered Plant Disease Detection & Diagnosis")
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        mode = st.radio(
            "Select Mode:",
            ["🔍 Analyze Leaf", "🤖 AI Assistant", "📊 Analysis History", "📋 Care Plan", "ℹ️ About"]
        )
        
        st.markdown("---")
        st.markdown("### 📚 Guide")
        st.markdown("""
        1. **Upload** a clear photo of a plant leaf
        2. **Analyze** to detect disease and severity
        3. **Follow** personalized rescue tips
        4. **Monitor** plant recovery progress
        """)
        
        st.markdown("---")
        st.markdown("### 🔬 System Info")
        
        if st.button("🔄 Reload Models"):
            st.session_state.model = None
            st.cache_resource.clear()
            st.success("✓ Models cleared. Will reload on next analysis.")
        
        db_path = init_db()
        if db_path:
            st.success(f"✓ Database: Active")
    
    # Main content
    if mode == "🔍 Analyze Leaf":
        st.markdown("## Upload & Analyze Leaf Image")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📤 Upload Image")
            uploaded_file = st.file_uploader(
                "Choose a leaf image (JPG/PNG)",
                type=['jpg', 'jpeg', 'png']
            )
            
            if uploaded_file:
                # Save uploaded file temporarily
                with open('temp_image.jpg', 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                # Display uploaded image
                st.image('temp_image.jpg', caption="Uploaded Image", use_container_width=True)
                
                # Validate image
                is_valid, message = ImagePreprocessor.validate_image('temp_image.jpg')
                if not is_valid:
                    st.error(f"❌ {message}")
                else:
                    st.success("✓ Image valid")
        
        with col2:
            if uploaded_file:
                st.markdown("### 🔬 Analysis")
                
                if st.button("🚀 Analyze Leaf", use_container_width=True):
                    with st.spinner("🔄 Analyzing image..."):
                        analysis = analyze_leaf_image('temp_image.jpg')
                        
                        if analysis:
                            st.markdown("---")
                            st.markdown("## 📋 Diagnosis Results")
                            
                            # Metrics row
                            m1, m2, m3, m4 = st.columns(4)
                            
                            with m1:
                                st.markdown('<div class="stats-metric">', unsafe_allow_html=True)
                                st.markdown(f"**Plant Species**")
                                st.markdown(f"### {analysis['plant'].title()}")
                                st.caption(f"Confidence: {analysis['plant_confidence']:.1%}")
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            with m2:
                                st.markdown('<div class="stats-metric">', unsafe_allow_html=True)
                                st.markdown(f"**Disease Detected**")
                                st.markdown(f"### {analysis['disease'].replace('_', ' ').title()}")
                                st.caption(f"Confidence: {analysis['disease_confidence']:.1%}")
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            with m3:
                                st.markdown('<div class="stats-metric">', unsafe_allow_html=True)
                                st.markdown(f"**Severity Level**")
                                display_severity_badge(analysis['severity'])
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            with m4:
                                st.markdown('<div class="stats-metric">', unsafe_allow_html=True)
                                st.markdown(f"**Affected Area**")
                                st.markdown(f"### {analysis['affected_percentage']:.1f}%")
                                st.caption(f"Confidence: {analysis['diagnosis_confidence']:.1%}")
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.markdown("---")
                            
                            # Detailed analysis
                            st.markdown("## 🔬 Detailed Analysis")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("### 🎨 Discoloration Breakdown")
                                
                                colors = analysis['discoloration_breakdown']
                                total = sum(colors.values())
                                
                                color_data = {
                                    '⚫ Black': colors.get('black_pixels', 0),
                                    '🟤 Brown': colors.get('brown_pixels', 0),
                                    '🟡 Yellow': colors.get('yellow_pixels', 0),
                                    '⚪ White': colors.get('white_pixels', 0),
                                }
                                
                                for color_name, pixels in color_data.items():
                                    if total > 0:
                                        percentage = (pixels / total) * 100
                                        st.write(f"{color_name}: {percentage:.1f}%")
                            
                            with col2:
                                st.markdown("### 📊 Confidence Metrics")
                                
                                metrics = {
                                    'Disease Detection': analysis['disease_confidence'],
                                    'Plant Species': analysis['plant_confidence'],
                                    'Overall Diagnosis': analysis['diagnosis_confidence'],
                                }
                                
                                for metric, value in metrics.items():
                                    st.progress(value, text=f"{metric}: {value:.1%}")
                            
                            st.markdown("---")
                            
                            # Rescue recommendations
                            st.markdown("## 🆘 Rescue Recommendations")
                            
                            for i, tip in enumerate(analysis['recommendations'][:3], 1):
                                st.markdown(f"""
                                <div class="disease-card">
                                <h4>💡 Tip {i}</h4>
                                <p>{tip}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # AI-Enhanced recommendations
                            if st.session_state.gemini_engine:
                                st.markdown("---")
                                st.markdown("## 🤖 AI-Enhanced Analysis (Powered by Gemini)")
                                
                                with st.spinner("🔄 Generating AI insights..."):
                                    # Disease explanation
                                    explanation = st.session_state.gemini_engine.generate_disease_explanation(
                                        disease_name=analysis['disease'],
                                        plant_name=analysis['plant'],
                                        severity=analysis['severity'],
                                        affected_percentage=analysis['affected_percentage']
                                    )
                                    
                                    st.markdown("### 📖 Disease Overview")
                                    st.info(explanation)
                                    
                                    # Enhanced tips
                                    enhanced_tips_result = st.session_state.gemini_engine.generate_personalized_tips(
                                        disease_name=analysis['disease'],
                                        plant_name=analysis['plant'],
                                        severity=analysis['severity'],
                                        affected_percentage=analysis['affected_percentage'],
                                        default_tips=analysis['recommendations']
                                    )
                                    
                                    if enhanced_tips_result['status'] == 'success':
                                        st.markdown("### 🎯 AI-Personalized Tips")
                                        st.success(enhanced_tips_result['enhanced_tips'])
                                    
                                    # Preventive measures
                                    st.markdown("### 🛡️ Preventive Measures")
                                    preventive = st.session_state.gemini_engine.identify_preventive_measures(
                                        plant_name=analysis['plant'],
                                        disease_name=analysis['disease'],
                                        climate_zone="Tropical"
                                    )
                                    for measure in preventive:
                                        st.write(f"✓ {measure}")
                            
                            # Download results
                            st.markdown("---")
                            st.markdown("## 📥 Export Results")
                            
                            import json
                            results_json = json.dumps({
                                'plant': analysis['plant'],
                                'disease': analysis['disease'],
                                'severity': analysis['severity'],
                                'affected_percentage': analysis['affected_percentage'],
                                'diagnosis_confidence': analysis['diagnosis_confidence'],
                                'recommendations': analysis['recommendations'],
                                'timestamp': analysis['timestamp']
                            }, indent=2)
                            
                            st.download_button(
                                label="📋 Download Results (JSON)",
                                data=results_json,
                                file_name=f"leaf_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
        
            else:
                st.info("👈 Upload a leaf image to get started!")
    
    elif mode == "📊 Analysis History":
        st.markdown("## 📊 Analysis History")
        
        if st.session_state.analysis_history:
            st.markdown(f"Total analyses: **{len(st.session_state.analysis_history)}**")
            
            for i, analysis in enumerate(reversed(st.session_state.analysis_history), 1):
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"""
                        <div class="disease-card">
                        <h4>Analysis #{i}</h4>
                        <b>Plant:</b> {analysis['plant'].title()} | 
                        <b>Disease:</b> {analysis['disease'].replace('_', ' ').title()} |
                        <b>Severity:</b> {analysis['severity'].upper()}<br>
                        <b>Affected Area:</b> {analysis['affected_percentage']:.1f}% | 
                        <b>Time:</b> {analysis['timestamp']}
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("👈 Upload a leaf image to get started!")
    
    elif mode == "🤖 AI Assistant":
        st.markdown("## 🤖 AI Plant Health Assistant")
        st.markdown("Ask any question about plant diseases, care, and treatment.")
        
        if st.session_state.gemini_engine:
            # Chat interface
            st.markdown("### 💬 Conversation")
            
            # Display chat history
            for i, msg in enumerate(st.session_state.chat_messages):
                if msg['role'] == 'user':
                    st.markdown(f"**You:** {msg['content']}")
                else:
                    st.info(f"**AI Assistant:** {msg['content']}")
            
            # Input box
            col1, col2 = st.columns([5, 1])
            with col1:
                user_input = st.text_input(
                    "Ask a question about plants or diseases...",
                    key="user_input",
                    placeholder="e.g., How do I treat powdery mildew on my tomato plant?"
                )
            with col2:
                send_button = st.button("Send 📤")
            
            if send_button and user_input:
                # Add user message
                st.session_state.chat_messages.append({
                    'role': 'user',
                    'content': user_input
                })
                
                # Get AI response
                with st.spinner("🤖 Thinking..."):
                    ai_response = st.session_state.gemini_engine.chat(user_input)
                
                # Add AI response
                st.session_state.chat_messages.append({
                    'role': 'assistant',
                    'content': ai_response
                })
                
                st.rerun()
            
            # Clear chat button
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_messages = []
                st.session_state.gemini_engine.clear_history()
                st.success("Chat cleared!")
        else:
            st.error("❌ AI Assistant unavailable. Please check API configuration.")
    
    elif mode == "📋 Care Plan":
        st.markdown("## 📋 Generate Personalized Care Plan")
        st.markdown("Create a detailed care plan for your plant based on disease diagnosis.")
        
        if st.session_state.gemini_engine:
            col1, col2 = st.columns(2)
            
            with col1:
                plant_select = st.selectbox(
                    "Select Plant Species:",
                    ["Tomato", "Potato", "Apple", "Corn", "Wheat"]
                )
            
            with col2:
                disease_select = st.selectbox(
                    "Select Disease:",
                    ["Early Blight", "Late Blight", "Powdery Mildew", "Septoria Leaf Spot", "Rust"]
                )
            
            col1, col2 = st.columns(2)
            with col1:
                severity_select = st.selectbox(
                    "Select Severity Level:",
                    ["Mild", "Moderate", "Severe"]
                )
            
            with col2:
                climate = st.selectbox(
                    "Climate Zone:",
                    ["Tropical", "Subtropical", "Temperate", "Cold"]
                )
            
            if st.button("📄 Generate Care Plan", use_container_width=True):
                with st.spinner("✍️ Creating personalized care plan..."):
                    care_plan = st.session_state.gemini_engine.generate_care_plan(
                        plant_name=plant_select,
                        disease_name=disease_select,
                        severity=severity_select
                    )
                    
                    st.markdown("### 📋 Your Personalized Care Plan")
                    st.success(care_plan)
                    
                    # Download option
                    st.download_button(
                        label="📥 Download Care Plan (TXT)",
                        data=care_plan,
                        file_name=f"care_plan_{plant_select}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
        else:
            st.error("❌ Care Plan feature unavailable. Please check API configuration.")
    
    else:  # About
        st.markdown("## ℹ️ About Leaf Health Check")
        
        st.markdown("""
        ### 🎯 Purpose
        Leaf Health Check is an AI-powered web application that helps farmers, gardeners, and plant enthusiasts:
        - **Diagnose** plant diseases from leaf photos
        - **Assess** disease severity
        - **Get personalized** rescue tips
        - **Monitor** plant health over time
        
        ### 🔬 Technology Stack
        - **Frontend:** Streamlit
        - **Backend:** Python, TensorFlow/Keras
        - **AI Models:** CNN (EfficientNetB0)
        - **Database:** SQLite
        - **Image Processing:** OpenCV, PIL
        
        ### 🌾 Supported Plants & Diseases
        - **Tomato:** Early Blight, Late Blight, Septoria Leaf Spot
        - **Potato:** Early Blight, Late Blight
        - **Apple:** Powdery Mildew, Scab, Rust
        - **Corn:** Northern Corn Leaf Blight, Gray Leaf Spot
        - **Wheat:** Septoria Leaf Blotch, Rust
        
        ### 📊 How It Works
        1. **Image Upload** → Validate and preprocess leaf image
        2. **Species Detection** → CNN identifies plant type
        3. **Disease Analysis** → Deep learning model classifies disease
        4. **Severity Assessment** → Analyze discoloration and symptoms
        5. **Recommendations** → Generate actionable rescue tips
        
        ### ⚠️ Disclaimer
        This application provides AI-assisted diagnosis for educational purposes. 
        For critical agricultural decisions, consult with plant pathologists or agricultural experts.
        
        ### 🤝 Contributing
        Contributions and feedback are welcome! Areas for improvement:
        - Add more plant species and diseases
        - Integrate real-time monitoring
        - Mobile app development
        - Extended weather-based recommendations
        
        ### 📞 Support
        For questions or issues, please contact: support@leafhealthcheck.com
        """)
        
        st.markdown("---")
        st.markdown("Made with ❤️ for sustainable agriculture | v1.0.0")


if __name__ == "__main__":
    main()
