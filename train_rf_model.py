# train_rf_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load and clean the dataset
data_path = "C:/Users/samgi/OneDrive/Documents/diamonds.csv"
df = pd.read_csv(data_path)

# Data cleaning
print("Initial data shape:", df.shape)

# Remove unnecessary columns
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

# Clean column names
df.columns = df.columns.str.strip()

# Remove rows with invalid measurements
df = df[(df['x'] > 0) & (df['y'] > 0) & (df['z'] > 0)]

# Remove duplicates
df = df.drop_duplicates()

print("Cleaned data shape:", df.shape)

# Encode categorical features
cut_map = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
color_map = {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7}
clarity_map = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8}

df['cut'] = df['cut'].map(cut_map)
df['color'] = df['color'].map(color_map)
df['clarity'] = df['clarity'].map(clarity_map)

# Prepare features and target
X = df.drop('price', axis=1)
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining Random Forest model...")
# Train Random Forest model with optimized parameters
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1  # Use all available cores
)

model.fit(X_train, y_train)

# Evaluate model
print("\nModel Evaluation:")
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5

print(f"R² Score: {r2:.4f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Save the model
model_filename = 'diamond_price_rf_model.pkl'
joblib.dump(model, model_filename)
print(f"\nModel saved as {model_filename}")

# Save the mappings for use in Streamlit app
mappings = {
    'cut_map': cut_map,
    'color_map': color_map,
    'clarity_map': clarity_map
}
joblib.dump(mappings, 'category_mappings.pkl')
print("Category mappings saved as category_mappings.pkl")