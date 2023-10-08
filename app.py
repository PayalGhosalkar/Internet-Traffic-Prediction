import pickle
import pandas as pd
from flask import Flask, render_template, request, jsonify
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__, template_folder='.')

# Load data and model
df = pd.read_csv("C:/Users/Gujar/Desktop/Projects/Data Science/Internet Traffic Prediction/data.csv")
loaded_model = pickle.load(open('C:/Users/Gujar/Desktop/Projects/Data Science/Internet Traffic Prediction/Internet_Traffic_Trained_Model.sav', 'rb'))
df['Date'] = pd.to_datetime(df['Date']).dt.date
data_index = df['Date'].tolist()

# ARIMA model class
class ARIMAModel:
    def __init__(self, model, data_index):
        self.model = model
        self.data_index = data_index

    def forecast(self, input_date):
        last_record_date = self.data_index[-1]
        steps_to_forecast = (input_date - last_record_date).days

        if steps_to_forecast < 0:
            steps_to_forecast = 1

        arima_forecast = self.model.forecast(steps=steps_to_forecast)
        formatted_forecast = (arima_forecast ** 2).tolist()
        return formatted_forecast

# Create ARIMAModel instance
arima_model = ARIMAModel(loaded_model, data_index)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/forecast', methods=['GET'])
def forecast():
    input_date_str = request.args.get('input_date')

    try:
        input_date = datetime.strptime(input_date_str, '%Y-%m-%d').date()
        forecast = arima_model.forecast(input_date)

        # Plot actual vs forecasted values
        plt.figure(figsize=(10, 6))
        plt.plot(df['Date'], df['visitors'], label='Actual', marker='o')
        forecast_dates = [df['Date'].iloc[-1] + pd.DateOffset(days=i) for i in range(1, len(forecast) + 1)]
        plt.plot(forecast_dates, forecast, label='Forecasted', marker='x', color='red')
        plt.xlabel('Date')
        plt.ylabel('Traffic')
        plt.title('Actual vs Forecasted Traffic')
        plt.legend()

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
        plt.close()

        # Create a list of forecast dates as strings
        forecast_date_strings = [str(date) for date in forecast_dates]

        return jsonify({'forecast': forecast, 'forecast_dates': forecast_date_strings, 'plot': img_base64})
    except ValueError:
        return jsonify({'error': 'Invalid date format. Please use YYYY-MM-DD.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
