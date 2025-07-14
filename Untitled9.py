import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score

# Set page config with diamond theme
st.set_page_config(
    page_title="💎 Diamond Price Predictor",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for diamond theme
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .prediction-result {
        background-color: #e8f4f8;
        border-left: 5px solid #2196F3;
        padding: 20px;
        margin: 20px 0;
        border-radius: 5px;
    }
    .price-display {
        font-size: 32px;
        font-weight: bold;
        color: #2196F3;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# App title and description
st.title("💎 Diamond Price Predictor")
st.markdown("""
Predict the price of your diamond based on its characteristics.  
Adjust the parameters below and click **Predict Price** to see the estimated value.
""")

# Load data and model
@st.cache_data
def load_data_and_model():
    # Load the diamonds dataset using the path from the first code
    data_path = "C:/Users/samgi/OneDrive/Documents/diamonds.csv"
    df = pd.read_csv(data_path)
    
    # Clean data (from second code)
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    df.columns = df.columns.str.strip()
    df = df[(df['x'] > 0) & (df['y'] > 0) & (df['z'] > 0)]
    df = df.drop_duplicates()
    
    # Encode categorical features (from second code)
    cut_map = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
    color_map = {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7}
    clarity_map = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8}

    df['cut'] = df['cut'].map(cut_map)
    df['color'] = df['color'].map(color_map)
    df['clarity'] = df['clarity'].map(clarity_map)
    
    # Load the model (from second code)
    model = joblib.load('diamond_price_model.pkl')
    
    return df, model

# Load the data and model
df, model = load_data_and_model()

# Display model performance in sidebar
st.sidebar.header("Model Information")
st.sidebar.write(f"Dataset samples: {len(df)}")
st.sidebar.write("Features used:")
st.sidebar.write(df.columns.tolist())

# --- Input Fields ---
col1, col2, col3 = st.columns(3)

with col1:
    carat = st.slider("Carat Weight", float(df['carat'].min()), float(df['carat'].max()), 1.0, 0.01)
    cut = st.selectbox("Cut Quality", ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
    color = st.selectbox("Color Grade", ['D', 'E', 'F', 'G', 'H', 'I', 'J'])

with col2:
    clarity = st.selectbox("Clarity", ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
    depth = st.slider("Depth (%)", float(df['depth'].min()), float(df['depth'].max()), 60.0, 0.1)
    table = st.slider("Table (%)", float(df['table'].min()), float(df['table'].max()), 55.0, 0.1)

with col3:
    x = st.slider("Length (x)", float(df['x'].min()), float(df['x'].max()), 5.0, 0.1)
    y = st.slider("Width (y)", float(df['y'].min()), float(df['y'].max()), 5.0, 0.1)
    z = st.slider("Height (z)", float(df['z'].min()), float(df['z'].max()), 3.0, 0.1)

# --- Prediction Logic ---
if st.button("Predict Price"):
    # Prepare input data
    cut_map = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
    color_map = {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7}
    clarity_map = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8}

    input_data = pd.DataFrame({
        'carat': [carat],
        'cut': [cut_map[cut]],
        'color': [color_map[color]],
        'clarity': [clarity_map[clarity]],
        'depth': [depth],
        'table': [table],
        'x': [x],
        'y': [y],
        'z': [z]
    })
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display results with enhanced styling
    st.balloons()
    st.markdown(f"""
    <div class="prediction-result">
        <h3>Predicted Diamond Price</h3>
        <div class="price-display">${prediction[0]:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show the input values
    st.subheader("Input Parameters")
    st.json({
        "Carat": carat,
        "Cut": cut,
        "Color": color,
        "Clarity": clarity,
        "Depth": f"{depth}%",
        "Table": f"{table}%",
        "Dimensions": f"{x} × {y} × {z} mm"
    })

st.markdown("---")
st.markdown("Made with ❤ using Streamlit")
