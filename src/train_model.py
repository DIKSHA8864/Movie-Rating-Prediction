import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

print("Loading dataset...")
# Load dataset with specified encoding
df = pd.read_csv('data/movies.csv', encoding='ISO-8859-1')  # or 'latin1'

print("Cleaning data...")
# Clean Duration column (remove 'min' and convert to numeric)
df['Duration'] = df['Duration'].str.replace(' min', '', regex=False)
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')

# Remove commas and convert 'Votes' to numeric
df['Votes'] = df['Votes'].str.replace(',', '', regex=False)  # Remove commas
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')    # Convert to numeric

# Drop rows where 'Duration' or 'Votes' or 'Rating' are missing
df = df.dropna(subset=['Duration', 'Votes', 'Rating'])

print(f"Data shape after cleaning: {df.shape}")

# Features (Duration and Votes) and Target (Rating)
X = df[['Duration', 'Votes']]  # Features
y = df['Rating']               # Target variable (Rating)

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Training model...")

# Train/test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Save the trained model and scaler to files
print("Saving model and scaler...")
joblib.dump(model, 'model/model.pkl')  # Save the model
joblib.dump(scaler, 'model/scaler.pkl')  # Save the scaler

print("Model and scaler saved successfully!")
