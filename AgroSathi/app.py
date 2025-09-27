import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import json
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import io
import base64
from datetime import datetime
import random

# Page configuration
st.set_page_config(
    page_title="AgriAssist - Agricultural Assistant",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background-color: blue;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #2E8B57;
    }
    .chat-bubble {
        max-width: 70%;
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
    }
    .user-bubble {
        background-color: #2E8B57;
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 5px;
    }
    .bot-bubble {
        background-color: #f0f0f0;
        color: #333;
        margin-right: auto;
        border-bottom-left-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

class AgriculturalChatbot:
    def __init__(self):
        self.crop_data = self.load_crop_data()
        self.disease_model = None
        self.disease_classes = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy',
            'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy',
            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
            'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
            'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
            'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
        ]
        
    def load_crop_data(self):
        # Sample crop data - in production, this would come from a database
        crop_data = {
            'rice': {'N': 80, 'P': 40, 'K': 40, 'temp': 22, 'humidity': 82, 'ph': 6.0},
            'wheat': {'N': 100, 'P': 50, 'K': 50, 'temp': 18, 'humidity': 70, 'ph': 6.5},
            'maize': {'N': 120, 'P': 60, 'K': 60, 'temp': 25, 'humidity': 75, 'ph': 6.8},
            'cotton': {'N': 90, 'P': 45, 'K': 45, 'temp': 28, 'humidity': 65, 'ph': 7.0},
            'sugarcane': {'N': 150, 'P': 75, 'K': 75, 'temp': 30, 'humidity': 80, 'ph': 7.5},
            'soybean': {'N': 70, 'P': 35, 'K': 35, 'temp': 22, 'humidity': 72, 'ph': 6.2},
            'potato': {'N': 110, 'P': 55, 'K': 55, 'temp': 20, 'humidity': 78, 'ph': 5.5},
            'tomato': {'N': 130, 'P': 65, 'K': 65, 'temp': 24, 'humidity': 75, 'ph': 6.0}
        }
        return crop_data
    
    def recommend_crop(self, N, P, K, temperature, humidity, ph):
        # Simple recommendation algorithm based on similarity
        best_crop = None
        best_score = float('inf')
        
        for crop, values in self.crop_data.items():
            score = (
                abs(N - values['N']) / 200 +
                abs(P - values['P']) / 100 +
                abs(K - values['K']) / 100 +
                abs(temperature - values['temp']) / 50 +
                abs(humidity - values['humidity']) / 100 +
                abs(ph - values['ph']) / 14
            )
            
            if score < best_score:
                best_score = score
                best_crop = crop
        
        return best_crop, self.crop_data[best_crop]
    
    def load_disease_model(self):
        # Load a pre-trained model for disease detection
        # In production, you would load your actual trained model
        try:
            model = models.resnet18(pretrained=True)
            model.fc = torch.nn.Linear(model.fc.in_features, len(self.disease_classes))
            return model
        except:
            return None
    
    def predict_disease(self, image):
        # Simple mock prediction - replace with actual model inference
        if self.disease_model is None:
            self.disease_model = self.load_disease_model()
        
        # Mock prediction for demonstration
        predicted_class = random.choice(self.disease_classes)
        confidence = random.uniform(0.7, 0.95)
        
        return predicted_class, confidence
    
    def chat_response(self, message):
        # Simple rule-based chatbot responses
        message = message.lower()
        
        if any(word in message for word in ['hello', 'hi', 'hey']):
            return "Hello! I'm AgriAssist, your agricultural assistant. How can I help you today?"
        
        elif any(word in message for word in ['crop', 'recommend', 'plant']):
            return "I can help with crop recommendations! Please go to the 'Crop Recommendation' tab and enter your soil and weather parameters."
        
        elif any(word in message for word in ['disease', 'sick', 'infected']):
            return "I can identify plant diseases! Please go to the 'Disease Identification' tab and upload an image of the affected plant."
        
        elif any(word in message for word in ['weather', 'temperature', 'humidity']):
            return "Weather conditions are crucial for farming. Ideal conditions vary by crop, but generally, most crops thrive between 20-30°C with moderate humidity."
        
        elif any(word in message for word in ['soil', 'nutrient', 'fertilizer']):
            return "Soil health is important! The main nutrients are Nitrogen (N), Phosphorus (P), and Potassium (K). I can help you determine the right balance for your crops."
        
        elif any(word in message for word in ['thank', 'thanks']):
            return "You're welcome! Happy to help with your agricultural needs."
        
        elif any(word in message for word in ['bye', 'goodbye']):
            return "Goodbye! Feel free to return if you have more agricultural questions."
        
        else:
            return "I'm here to help with agricultural advice! You can ask me about crop recommendations, disease identification, or general farming tips."

def main():
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = AgriculturalChatbot()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Header
    st.markdown('<h1 class="main-header">🌱 AgriAssist - Agricultural Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose a feature", 
                                   ["Chat with AgriAssist", "Crop Recommendation", "Disease Identification"])
    
    # Main content based on selection
    if app_mode == "Chat with AgriAssist":
        chat_interface()
    elif app_mode == "Crop Recommendation":
        crop_recommendation_interface()
    elif app_mode == "Disease Identification":
        disease_identification_interface()

def chat_interface():
    st.header("💬 Chat with AgriAssist")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat container
        chat_container = st.container()
        
        # Display chat history
        with chat_container:
            for chat in st.session_state.chat_history:
                if chat['role'] == 'user':
                    st.markdown(f'<div class="chat-bubble user-bubble">{chat["message"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-bubble bot-bubble">{chat["message"]}</div>', unsafe_allow_html=True)
        
        # Input area
        st.markdown("---")
        user_input = st.text_input("Type your message here:", key="chat_input")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Send") and user_input:
                # Add user message to history
                st.session_state.chat_history.append({'role': 'user', 'message': user_input, 'time': datetime.now()})
                
                # Get bot response
                bot_response = st.session_state.chatbot.chat_response(user_input)
                st.session_state.chat_history.append({'role': 'bot', 'message': bot_response, 'time': datetime.now()})
                
                # Clear input and rerun
                st.rerun()
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>💡 Quick Tips</h3>
            <p>Ask me about:</p>
            <ul>
                <li>Crop recommendations</li>
                <li>Plant diseases</li>
                <li>Soil nutrients</li>
                <li>Weather conditions</li>
                <li>Farming techniques</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

def crop_recommendation_interface():
    st.header("🌾 Crop Recommendation System")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Soil and Weather Parameters")
        
        with st.form("crop_recommendation_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                N = st.slider("Nitrogen (N) ppm", 0, 200, 50)
                P = st.slider("Phosphorus (P) ppm", 0, 100, 30)
                K = st.slider("Potassium (K) ppm", 0, 100, 30)
            
            with col2:
                temperature = st.slider("Temperature (°C)", 0.0, 40.0, 25.0)
                humidity = st.slider("Humidity (%)", 0.0, 100.0, 70.0)
                ph = st.slider("pH Level", 4.0, 9.0, 6.5)
            
            with col3:
                rainfall = st.slider("Rainfall (mm)", 0.0, 500.0, 150.0)
                st.write("")  # Spacer
                st.write("")  # Spacer
                submit_button = st.form_submit_button("Get Recommendation")
        
        if submit_button:
            with st.spinner("Analyzing parameters and finding best crop..."):
                recommended_crop, crop_info = st.session_state.chatbot.recommend_crop(N, P, K, temperature, humidity, ph)
                
                st.success(f"**Recommended Crop: {recommended_crop.title()}**")
                
                # Display crop information
                st.subheader(f"🌱 {recommended_crop.title()} Growing Conditions")
                
                info_col1, info_col2, info_col3 = st.columns(3)
                
                with info_col1:
                    st.metric("Ideal Temperature", f"{crop_info['temp']}°C")
                    st.metric("Ideal Humidity", f"{crop_info['humidity']}%")
                
                with info_col2:
                    st.metric("Nitrogen (N)", f"{crop_info['N']} ppm")
                    st.metric("Phosphorus (P)", f"{crop_info['P']} ppm")
                
                with info_col3:
                    st.metric("Potassium (K)", f"{crop_info['K']} ppm")
                    st.metric("pH Level", f"{crop_info['ph']}")
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Parameter Guidelines</h3>
            <p><strong>Nitrogen (N):</strong> Leaf growth</p>
            <p><strong>Phosphorus (P):</strong> Root development</p>
            <p><strong>Potassium (K):</strong> Fruit quality</p>
            <p><strong>pH Level:</strong> 6.0-7.0 ideal for most crops</p>
        </div>
        """, unsafe_allow_html=True)

def disease_identification_interface():
    st.header("🔍 Plant Disease Identification")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload Plant Image")
        
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Analyze button
            if st.button("Identify Disease"):
                with st.spinner("Analyzing image for diseases..."):
                    predicted_disease, confidence = st.session_state.chatbot.predict_disease(image)
                    
                    # Display results
                    st.subheader("🔬 Analysis Results")
                    
                    if "healthy" in predicted_disease.lower():
                        st.success(f"**Result: Healthy Plant**")
                        st.write(f"Confidence: {confidence:.2%}")
                        st.info("🎉 Great news! Your plant appears to be healthy.")
                    else:
                        st.error(f"**Detected Disease: {predicted_disease.replace('_', ' ').title()}**")
                        st.write(f"Confidence: {confidence:.2%}")
                        
                        # Display treatment suggestions
                        st.subheader("💊 Recommended Treatment")
                        treatment_suggestions = get_treatment_suggestions(predicted_disease)
                        st.write(treatment_suggestions)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>📸 Image Tips</h3>
            <p>For best results:</p>
            <ul>
                <li>Clear, well-lit photos</li>
                <li>Focus on affected leaves</li>
                <li>Multiple angles if possible</li>
                <li>Avoid blurry images</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>🌿 Common Diseases</h3>
            <p>• Early/Late Blight</p>
            <p>• Powdery Mildew</p>
            <p>• Leaf Spot</p>
            <p>• Rust Diseases</p>
            <p>• Bacterial Spot</p>
        </div>
        """, unsafe_allow_html=True)

def get_treatment_suggestions(disease):
    """Return treatment suggestions based on disease type"""
    treatments = {
        'blight': "Remove affected leaves, apply copper-based fungicide, improve air circulation",
        'mildew': "Apply sulfur or potassium bicarbonate, reduce humidity, increase sunlight",
        'rust': "Remove infected parts, apply fungicide, avoid overhead watering",
        'spot': "Apply copper fungicide, remove diseased leaves, water at base only",
        'mosaic': "Remove infected plants, control aphids, use virus-free seeds"
    }
    
    disease_lower = disease.lower()
    for key, treatment in treatments.items():
        if key in disease_lower:
            return treatment
    
    return "Consult with local agricultural extension service for specific treatment recommendations. Generally, remove affected parts and apply appropriate organic or chemical treatments."

if __name__ == "__main__":
    main()