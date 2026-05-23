import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# ============================================================================
# PAGE CONFIGURATION & STYLING
# ============================================================================

st.set_page_config(
    page_title="House Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "### 🏠 House Price Predictor\nEstimate house prices using AI-powered machine learning"
    }
)

# Custom CSS for modern, premium design
st.markdown("""
    <style>
    /* Root styling */
    :root {
        --primary-color: #6366f1;
        --primary-dark: #4f46e5;
        --success-color: #10b981;
        --danger-color: #ef4444;
        --warning-color: #f59e0b;
        --bg-dark: #0f172a;
        --bg-light: #1e293b;
        --card-bg: #1e293b;
        --text-primary: #f1f5f9;
        --text-secondary: #cbd5e1;
        --border-color: #334155;
    }
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1a1f35 100%);
        color: var(--text-primary);
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: white;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .header-subtitle {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.9);
        margin: 0.5rem 0 0 0;
        font-weight: 300;
    }
    
    /* Card styling */
    .card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .card:hover {
        border-color: var(--primary-color);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.1);
        transform: translateY(-2px);
    }
    
    .card-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Input styling */
    .stSlider > div > div {
        background-color: var(--border-color) !important;
    }
    
    .stSlider label {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 10px 25px rgba(99, 102, 241, 0.2);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 35px rgba(99, 102, 241, 0.3);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Metric styling */
    .metric-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: var(--primary-color);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.1);
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: var(--text-secondary);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.875rem;
        font-weight: 800;
        color: var(--primary-color);
    }
    
    .metric-unit {
        font-size: 0.875rem;
        color: var(--text-secondary);
        margin-top: 0.25rem;
    }
    
    /* Prediction result styling */
    .prediction-result {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border: 2px solid #10b981;
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
        box-shadow: 0 20px 60px rgba(16, 185, 129, 0.2);
        animation: slideIn 0.5s ease-out;
    }
    
    .prediction-label {
        font-size: 0.875rem;
        color: rgba(255, 255, 255, 0.9);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .prediction-price {
        font-size: 3rem;
        font-weight: 900;
        color: white;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .prediction-range {
        font-size: 0.875rem;
        color: rgba(255, 255, 255, 0.8);
        margin-top: 0.5rem;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: var(--bg-dark) !important;
    }
    
    /* Grid layout */
    .grid-2 {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    @media (max-width: 768px) {
        .grid-2 {
            grid-template-columns: 1fr;
        }
    }
    
    /* Divider */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--border-color), transparent);
        margin: 2rem 0;
    }
    
    /* Success message */
    .success-message {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        color: #065f46;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #10b981;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        color: #1e40af;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
    
    /* Animation */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .slide-in {
        animation: slideIn 0.5s ease-out;
    }
    
    /* Typography */
    h1, h2, h3 {
        letter-spacing: -0.02em;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-dark);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-color);
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# PAGE CONTENT
# ============================================================================

# Header
st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🏠 House Price Predictor</h1>
        <p class="header-subtitle">Accurate real estate valuation powered by AI & Machine Learning</p>
    </div>
""", unsafe_allow_html=True)

# Load model
try:
    model, feature_names = joblib.load("model.pkl")
except:
    st.error("⚠️ Error loading model. Please ensure model.pkl is available.")
    st.stop()

# ============================================================================
# SIDEBAR - INPUT CONTROLS
# ============================================================================

st.sidebar.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h2 style='margin: 0; font-size: 1.5rem;'>📋 Property Details</h2>
        <p style='color: #cbd5e1; margin-top: 0.5rem; font-size: 0.875rem;'>
            Adjust the sliders to update predictions
        </p>
    </div>
""", unsafe_allow_html=True)

with st.sidebar:
    # Location Section
    st.markdown("### 📍 Location")
    col1, col2 = st.columns(2)
    with col1:
        Latitude = st.slider("Latitude", 32.0, 42.0, 36.0, step=0.1, key="lat")
    with col2:
        Longitude = st.slider("Longitude", -125.0, -114.0, -120.0, step=0.1, key="lon")
    
    st.markdown("---")
    
    # Property Features Section
    st.markdown("### 🏡 Property Features")
    MedInc = st.slider("Median Income (×$100k)", 0.0, 15.0, 5.0, step=0.1, key="income")
    HouseAge = st.slider("House Age (years)", 1, 50, 20, step=1, key="age")
    AveRooms = st.slider("Average Rooms", 1.0, 10.0, 5.0, step=0.5, key="rooms")
    AveBedrms = st.slider("Average Bedrooms", 0.5, 5.0, 1.0, step=0.25, key="beds")
    
    st.markdown("---")
    
    # Demographics Section
    st.markdown("### 👥 Demographics")
    Population = st.slider("Population", 100, 5000, 1000, step=100, key="pop")
    AveOccup = st.slider("Average Occupancy", 1.0, 10.0, 3.0, step=0.5, key="occup")

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

# Feature Engineering
rooms_per_house = AveRooms / HouseAge
bedroom_ratio = AveBedrms / AveRooms if AveRooms > 0 else 0

# Create input dictionary
input_dict = {
    "MedInc": MedInc,
    "HouseAge": HouseAge,
    "AveRooms": AveRooms,
    "AveBedrms": AveBedrms,
    "Population": Population,
    "AveOccup": AveOccup,
    "Latitude": Latitude,
    "Longitude": Longitude,
    "rooms_per_house": rooms_per_house,
    "bedroom_ratio": bedroom_ratio
}

# Prepare feature vector
data = np.array([[input_dict[col] for col in feature_names]])

# ============================================================================
# INPUT SUMMARY SECTION
# ============================================================================

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
        <div class="card">
            <div class="card-title">📊 Input Summary</div>
    """, unsafe_allow_html=True)
    
    # Create summary table
    summary_data = {
        "Feature": list(input_dict.keys()),
        "Value": [f"{v:.2f}" if isinstance(v, float) else str(v) for v in input_dict.values()]
    }
    summary_df = pd.DataFrame(summary_data)
    
    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Feature": st.column_config.TextColumn("Feature", width="large"),
            "Value": st.column_config.TextColumn("Value", width="small")
        }
    )
    
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="card">
            <div class="card-title">ℹ️ About</div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        **Model Type**  
        Random Forest Regressor
        
        **Accuracy**  
        73.2% (Test R² Score)
        
        **Dataset**  
        California Housing
        
        **Features Used**  
        10 Features
    """)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================================
# PREDICTION SECTION
# ============================================================================

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("🔮 Predict House Price", use_container_width=True):
        try:
            # Make prediction
            prediction = model.predict(data)[0]
            predicted_price = prediction * 100000
            
            # Display success message
            st.markdown(f"""
                <div class="prediction-result">
                    <div class="prediction-label">💰 Estimated House Price</div>
                    <div class="prediction-price">${predicted_price:,.0f}</div>
                    <div class="prediction-range">
                        Based on {len(feature_names)} property features
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Store prediction in session state for history
            if 'predictions' not in st.session_state:
                st.session_state.predictions = []
            
            st.session_state.predictions.append({
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'price': predicted_price,
                'income': MedInc,
                'age': HouseAge
            })
            
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")

# ============================================================================
# DETAILED METRICS SECTION
# ============================================================================

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

st.markdown("""
    <div class="card">
        <div class="card-title">📈 Property Metrics</div>
    </div>
""", unsafe_allow_html=True)

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📏 Price per Room</div>
            <div class="metric-value">${(prediction * 100000 / AveRooms) if 'prediction' in locals() else 0:,.0f}</div>
        </div>
    """, unsafe_allow_html=True)

with metric_col2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🛏️ Rooms per Bedroom</div>
            <div class="metric-value">{rooms_per_house:.2f}</div>
        </div>
    """, unsafe_allow_html=True)

with metric_col3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">👨‍👩‍👧‍👦 Density</div>
            <div class="metric-value">{AveOccup:.2f}</div>
            <div class="metric-unit">people per house</div>
        </div>
    """, unsafe_allow_html=True)

with metric_col4:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📅 House Age</div>
            <div class="metric-value">{HouseAge}</div>
            <div class="metric-unit">years</div>
        </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PREDICTION HISTORY SECTION
# ============================================================================

if 'predictions' in st.session_state and len(st.session_state.predictions) > 0:
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class="card">
            <div class="card-title">📋 Recent Predictions</div>
        </div>
    """, unsafe_allow_html=True)
    
    history_df = pd.DataFrame(st.session_state.predictions[-5:])
    history_df['price'] = history_df['price'].apply(lambda x: f"${x:,.0f}")
    
    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "timestamp": st.column_config.TextColumn("Time", width="small"),
            "price": st.column_config.TextColumn("Predicted Price", width="large"),
            "income": st.column_config.NumberColumn("Income", width="small", format="%.1f"),
            "age": st.column_config.NumberColumn("Age", width="small", format="%.0f")
        }
    )

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

st.markdown("""
    <div style='text-align: center; color: #cbd5e1; padding: 2rem 0; font-size: 0.875rem;'>
        <p>🏠 House Price Predictor v1.0 | Powered by Machine Learning</p>
        <p>Real Estate Valuation using California Housing Dataset</p>
        <p style='color: #64748b; margin-top: 1rem;'>
            Accuracy: 73.2% | Model: Random Forest | Features: 10
        </p>
    </div>
""", unsafe_allow_html=True)