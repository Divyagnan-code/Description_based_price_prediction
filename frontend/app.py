import streamlit as st
import requests
import base64
from PIL import Image
import io

# Page config
st.set_page_config(
    page_title="Product Price Predictor",
    page_icon="💰",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1.5rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.1rem;
        padding: 0.6rem;
        border-radius: 8px;
        border: none;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# API URL
API_URL = "http://localhost:8000"

# Title
st.markdown('<div class="main-header">Product Price Predictor</div>', unsafe_allow_html=True)
st.markdown("---")

# Create two columns for input form
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Product Information")
    
    item_name = st.text_input(
        "Item Name *",
        placeholder="e.g., Amazon Fresh Brand Organic Bananas"
    )
    
    description = st.text_area(
        "Product Description (optional)",
        placeholder="Enter detailed product description...",
        height=100
    )
    
    value = st.number_input(
        "Value *",
        min_value=0.0,
        step=0.1,
        help="Enter numeric value (e.g., quantity, weight, or volume)"
    )
    
    measurement_type = st.selectbox(
        "Measurement Type *",
        options=["number", "weight", "volume"]
    )
    
    unit = ""
    if measurement_type == "weight":
        unit = st.selectbox(
            "Weight Unit *",
            options=["grams", "g", "kilograms", "kg", "pounds", "lbs", "ounces", "oz"]
        )
    elif measurement_type == "volume":
        unit = st.selectbox(
            "Volume Unit *",
            options=["milliliters", "ml", "liters", "l", "fluid ounces", "fl oz", 
                     "gallons", "gal", "quarts", "qt", "pints", "pt", "cups"]
        )
    else:
        unit = st.text_input(
            "Unit (optional)",
            placeholder="e.g., count, pack"
        )

with col2:
    st.subheader("Product Image")
    
    uploaded_file = st.file_uploader(
        "Upload Product Image *",
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

# Predict button
st.markdown("<br>", unsafe_allow_html=True)
predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])

with predict_col2:
    predict_button = st.button("Predict Price", use_container_width=True)

# Prediction logic
if predict_button:
    if not item_name:
        st.error("Please enter an item name.")
    elif value <= 0:
        st.error("Please enter a valid value greater than 0.")
    elif uploaded_file is None:
        st.error("Please upload a product image.")
    elif measurement_type in ["weight", "volume"] and not unit:
        st.error(f"Please select a {measurement_type} unit.")
    else:
        with st.spinner("Predicting price..."):
            try:
                files = {
                    'image': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                }
                
                data = {
                    'item_name': item_name,
                    'description': description if description else "",
                    'value': value,
                    'measurement_type': measurement_type,
                    'unit': unit if unit else ""
                }
                
                response = requests.post(
                    f"{API_URL}/predict",
                    files=files,
                    data=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result['success']:
                        st.markdown("---")
                        st.markdown("### Prediction Result")
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h3 style="color: #1f77b4;">Predicted Price</h3>
                            <h1 style="color: #2ecc71; font-size: 3rem; margin: 0;">
                                ${result['predicted_price']:.2f}
                            </h1>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("Prediction failed. Please check your input.")
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
            
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the API server. Ensure the backend is running.")
            except requests.exceptions.Timeout:
                st.error("Request timed out. Please try again.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Sidebar with information
with st.sidebar:
    st.markdown("## About")
    st.info("""
    This application predicts product prices using:
    - Product name and description
    - Product image (via embeddings)
    - Measurement details
    """)
    
    st.markdown("---")
    st.markdown("## API Status")
    
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            st.success(f"API is {health_data['status']}")
            st.info(f"Device: {health_data.get('device', 'N/A')}")
        else:
            st.warning("API is not responding properly.")
    except:
        st.error("API is offline.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; padding: 1rem;">
    Built with Streamlit and FastAPI
</div>
""", unsafe_allow_html=True)
