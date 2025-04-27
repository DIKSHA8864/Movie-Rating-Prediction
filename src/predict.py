import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load('model/model.pkl')
scaler = joblib.load('model/scaler.pkl')

def predict_rating(duration, votes):
    # Prepare the data for prediction
    input_data = np.array([[duration, votes]])
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Predict the rating
    predicted_rating = model.predict(input_data_scaled)
    
    return predicted_rating[0]
