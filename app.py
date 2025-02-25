from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.preprocessing import MinMaxScaler
import datetime
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Hello, Flask is working!"

@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify({"message": "StocksGod API is working!", "status": "success"})

# Load the trained LSTM model
model = tf.keras.models.load_model("lstm_stock_model.h5")

# Load the scaler (make sure this file has been saved separately)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

print(f"[DEBUG] Loaded 'model' is of type: {type(model)}")
print(f"[DEBUG] Loaded 'scaler' is of type: {type(scaler)}")

# Tiingo API details (ensure your API key is valid)
TIINGO_API_KEY = "dcbdde936e7866b9ae45c380c36a2f04c62bcc90"

def get_stock_data(stock_symbol):
    """Fetch historical stock data from Tiingo API"""
    try:
        end_date = datetime.datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.datetime.today() - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
        url = f"https://api.tiingo.com/tiingo/daily/{stock_symbol}/prices"
        headers = {
            "Authorization": f"Token {TIINGO_API_KEY}",
            "Content-Type": "application/json"
        }
        params = {"startDate": start_date, "endDate": end_date}

        print(f"[DEBUG] Fetching stock data from URL: {url}")
        print(f"[DEBUG] Request headers: {headers}")
        print(f"[DEBUG] Request parameters: {params}")

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        data = response.json()
        if not data:
            return None

        df = pd.DataFrame(data)
        df.set_index("date", inplace=True)
        df.index = pd.to_datetime(df.index)
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching stock data: {e}")
        return None

def predict_stock_prices(stock_symbol):
    """Predict the next 30 days of stock prices along with current price and key milestones"""
    df = get_stock_data(stock_symbol)
    if df is None or "close" not in df.columns:
        return {"error": "Stock data not found!"}

    close_prices = df["close"].values.reshape(-1, 1)
    # Use the last actual close price as the current price
    current_price = close_prices[-1][0]
    
    try:
        scaled_data = scaler.transform(close_prices)
    except Exception as e:
        print(f"[DEBUG] Error while transforming data with scaler: {e}")
        return {"error": "Scaler error during data transformation!"}

    X_input = scaled_data[-100:].reshape(1, 100, 1)  # Using the last 100 days for prediction input
    predictions = []

    for _ in range(30):  # Predict next 30 days
        pred = model.predict(X_input)
        predictions.append(pred[0][0])
        X_input = np.append(X_input[:, 1:, :], [[[pred[0][0]]]], axis=1)

    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    # Convert NumPy float32 values to Python float (JSON serializable)
    result = {
        "current_price": float(current_price),
        "price_after_15_days": float(predicted_prices[14]),  # 15th day prediction
        "price_after_30_days": float(predicted_prices[29]),  # 30th day prediction
        "dates": [(df.index[-1] + datetime.timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(30)],
        "predicted_prices": [float(p) for p in predicted_prices.tolist()]
    }
    return result

@app.route('/api/predict', methods=['GET'])
def predict():
    stock_symbol = request.args.get("symbol")
    if not stock_symbol:
        return jsonify({"error": "Stock symbol is required!"}), 400

    print(f"[DEBUG] Received prediction request for symbol: {stock_symbol}")
    result = predict_stock_prices(stock_symbol)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)