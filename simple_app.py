from flask import Flask, jsonify
import os

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        "message": "🏆 Gold Price Forecasting API",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "forecast": "/forecast",
            "about": "/about"
        }
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "service": "Gold Forecasting",
        "message": "Simple Flask app working!"
    })

@app.route('/forecast')
def forecast():
    # Simple mock forecast
    return jsonify({
        "current_price": 2650.0,
        "forecast_7_days": [
            {"date": "2025-09-29", "price": 2655.0},
            {"date": "2025-09-30", "price": 2660.0},
            {"date": "2025-10-01", "price": 2658.0},
            {"date": "2025-10-02", "price": 2665.0},
            {"date": "2025-10-03", "price": 2670.0},
            {"date": "2025-10-04", "price": 2668.0},
            {"date": "2025-10-05", "price": 2675.0}
        ],
        "confidence": "medium"
    })

@app.route('/about')
def about():
    return jsonify({
        "name": "Gold Price Forecasting API",
        "version": "1.0",
        "description": "Simple, reliable gold price forecasting service"
    })

if __name__ == '__main__':
    # For Azure Web App
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)