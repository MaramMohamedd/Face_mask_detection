import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cv2

# Configure page
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced Blue and White Theme CSS
st.markdown("""
<style>
    /* Main app styling */
    .main {
        padding-top: 1rem;
    }
    
    .stApp {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #60a5fa 100%);
        min-height: 100vh;
    }
    
    /* Main container with glass effect */
    .main-container {
        background: rgba(255, 255, 255, 0.98);
        padding: 2.5rem;
        border-radius: 25px;
        box-shadow: 0 25px 50px rgba(30, 64, 175, 0.15);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 1rem;
        position: relative;
        overflow: hidden;
    }
    
    .main-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #1d4ed8, #3b82f6, #60a5fa, #93c5fd);
    }
    
    /* Header styling */
    .header-container {
        text-align: center;
        margin-bottom: 2.5rem;
        padding: 2rem;
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        border-radius: 20px;
        color: white;
        box-shadow: 0 10px 30px rgba(30, 64, 175, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: shimmer 3s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { transform: translateX(-100%) translateY(-100%) rotate(0deg); }
        50% { transform: translateX(0%) translateY(0%) rotate(180deg); }
    }
    
    .logo-container {
        display: flex;
        justify-content: center;
        margin-bottom: 1.5rem;
        z-index: 2;
        position: relative;
    }
    
    /* Prediction containers */
    .prediction-container {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        padding: 2rem;
        border-radius: 20px;
        border-left: 6px solid #3b82f6;
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.1);
        transition: all 0.3s ease;
    }
    
    .prediction-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(59, 130, 246, 0.15);
    }
    
    .prediction-positive {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border-left: 6px solid #10b981;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.1);
    }
    
    .prediction-negative {
        background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
        border-left: 6px solid #ef4444;
        box-shadow: 0 8px 25px rgba(239, 68, 68, 0.1);
    }
    
    /* Upload section */
    .upload-section {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 2.5rem;
        border-radius: 20px;
        border: 3px dashed #3b82f6;
        text-align: center;
        margin: 2rem 0;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .upload-section:hover {
        border-color: #1d4ed8;
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(59, 130, 246, 0.1);
    }
    
    /* Metric containers */
    .metric-container {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 8px 25px rgba(30, 64, 175, 0.2);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-container:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(30, 64, 175, 0.3);
    }
    
    .metric-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .metric-container:hover::before {
        left: 100%;
    }
    
    /* Typography */
    h1 {
        color: white;
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        z-index: 2;
        position: relative;
    }
    
    h2 {
        color: #1e40af;
        font-weight: 700;
        text-shadow: 0 1px 2px rgba(30, 64, 175, 0.1);
    }
    
    h3 {
        color: #1e40af;
        font-weight: 600;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 700;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(30, 64, 175, 0.2);
    }
    
    .stButton > button:hover {
        box-shadow: 0 8px 25px rgba(30, 64, 175, 0.4);
        transform: translateY(-2px);
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%);
    }
    
    /* Progress bar customization */
    .stProgress .st-bo {
        background: linear-gradient(90deg, #3b82f6, #60a5fa);
    }
    
    /* File uploader styling */
    .stFileUploader > div > div {
        background: white;
        border: 2px solid #e5e7eb;
        border-radius: 15px;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div > div:hover {
        border-color: #3b82f6;
        box-shadow: 0 5px 15px rgba(59, 130, 246, 0.1);
    }
    
    /* Info boxes */
    .stInfo {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-left: 4px solid #3b82f6;
    }
    
    /* Error styling */
    .stError {
        background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
        border-left: 4px solid #ef4444;
    }
    
    /* Success styling */
    .stSuccess {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border-left: 4px solid #10b981;
    }
    
    /* Spinner customization */
    .stSpinner > div {
        border-top-color: #3b82f6 !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_trained_model():
    """Load the pre-trained face mask detection model"""
    try:
        model = load_model(r'D:\Documents\NTI training\FACE MASK DETECTOR\face detector 2nd edition.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please ensure 'face_mask_detection.keras' is in the same directory as this script.")
        return None

def preprocess_image(image):
    """Preprocess the uploaded image for prediction"""
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Convert to RGB if necessary
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    elif len(img_array.shape) == 3 and img_array.shape[2] == 1:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    # Resize to model input size
    img_resized = cv2.resize(img_array, (128, 128))
    
    # Normalize pixel values
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

def predict_mask(model, processed_image):
    """Make prediction on the processed image"""
    prediction = model.predict(processed_image, verbose=0)
    confidence = float(prediction[0][0])
    
    # Binary classification: 0 = with mask, 1 = without mask
    if confidence > 0.5:
        label = "No Mask Detected"
        confidence_score = confidence
        is_wearing_mask = False
    else:
        label = "Mask Detected"
        confidence_score = 1 - confidence
        is_wearing_mask = True
    
    return label, confidence_score, is_wearing_mask

def main():
    # Main container
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Header section
    st.markdown('<div class="header-container">', unsafe_allow_html=True)
    
    # Logo section
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    try:
        st.image("nti_logo.png", width=200)
    except:
        st.markdown("**NTI Logo**")
        st.caption("(Place 'nti_logo.jpg' in the same directory)")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<h1>🎭 Face Mask Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 1.2rem; margin: 0; opacity: 0.9;">Advanced AI-powered mask detection using deep learning</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Load model
    model = load_trained_model()
    
    if model is None:
        st.stop()
    
    # Upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📸 Upload an Image")
    st.markdown("Support formats: JPG, JPEG, PNG")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Create two columns for image and results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 🖼️ Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True, caption="Uploaded Image")
            
            # Image info
            st.markdown("**Image Details:**")
            st.write(f"• Size: {image.size[0]} x {image.size[1]} pixels")
            st.write(f"• Format: {image.format}")
            st.write(f"• Mode: {image.mode}")
        
        with col2:
            st.markdown("### 🔍 Analysis Results")
            
            with st.spinner("🤖 Analyzing image..."):
                try:
                    # Preprocess image
                    processed_image = preprocess_image(image)
                    
                    # Make prediction
                    prediction_label, confidence, is_wearing_mask = predict_mask(model, processed_image)
                    
                    # Display results
                    prediction_class = "prediction-positive" if is_wearing_mask else "prediction-negative"
                    
                    st.markdown(f'''
                    <div class="prediction-container {prediction_class}">
                        <h2 style="margin: 0; color: {'#16a34a' if is_wearing_mask else '#dc2626'};">
                            {'✅' if is_wearing_mask else '❌'} {prediction_label}
                        </h2>
                        <p style="font-size: 1.1rem; margin: 0.5rem 0; color: #374151;">
                            Confidence: <strong>{confidence:.1%}</strong>
                        </p>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # Progress bar for confidence
                    st.progress(confidence)
                    
                    # Additional metrics
                    col2a, col2b = st.columns(2)
                    with col2a:
                        st.markdown(f'''
                        <div class="metric-container">
                            <h3 style="margin: 0; color: white;">Accuracy</h3>
                            <p style="font-size: 1.5rem; margin: 0; color: white;">{confidence:.1%}</p>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    with col2b:
                        status = "Safe" if is_wearing_mask else "Warning"
                        st.markdown(f'''
                        <div class="metric-container">
                            <h3 style="margin: 0; color: white;">Status</h3>
                            <p style="font-size: 1.5rem; margin: 0; color: white;">{status}</p>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    st.info("Please ensure the image is valid and the model is properly loaded.")
    
    else:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; color: #6b7280;">
            <h3>👆 Please upload an image to begin analysis</h3>
            <p>The AI model will analyze the image and detect whether the person is wearing a face mask or not.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; padding: 1rem;">
        <p><strong>Face Mask Detection System</strong> | Powered by Deep Learning & TensorFlow</p>
        <p>Built with ❤️ using Streamlit</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()