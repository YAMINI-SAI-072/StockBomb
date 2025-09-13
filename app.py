from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load pretrained model (you can also train it here if you prefer)
model = load_model('model/stock_model.h5')
scaler = MinMaxScaler(feature_range=(0, 1))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    chart = None
    if request.method == 'POST':
        ticker = request.form['ticker']
        data = yf.download(ticker, period="2y", interval="1d")
        close_prices = data['Close'].values.reshape(-1,1)
        
        scaled_data = scaler.fit_transform(close_prices)
        
        # Prepare last 60 days for prediction
        x_input = scaled_data[-60:].reshape(1, 60, 1)
        pred_scaled = model.predict(x_input)
        
        pred_price = scaler.inverse_transform(pred_scaled)[0][0]
        prediction = round(pred_price, 2)
        
        # Plotting
        plt.figure(figsize=(10,5))
        plt.plot(data['Close'], label='Historical Price')
        plt.axhline(y=prediction, color='r', linestyle='--', label=f'Predicted Price: {prediction}')
        plt.title(f'Stock Price Prediction for {ticker}')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        
        # Save plot to string buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        chart = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
    
    return render_template('index.html', prediction=prediction, chart=chart)

if __name__ == "__main__":
    app.run(debug=True)
