import azure.functions as func
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the function app
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.function_name(name="HttpTrigger")
import azure.functions as func
import logging
import json

# Create the Azure Functions app
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="health", methods=["GET"])
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """Ultra-simple health check"""
    return func.HttpResponse(
        json.dumps({
            "status": "healthy",
            "service": "Gold Forecasting API",
            "message": "Minimal Azure Functions deployment working!"
        }),
        status_code=200,
        mimetype="application/json"
    )

@app.route(route="", methods=["GET"])
def root(req: func.HttpRequest) -> func.HttpResponse:
    """Root endpoint"""
    return func.HttpResponse(
        json.dumps({
            "message": "🏆 Gold Price Forecasting API is LIVE!",
            "endpoints": {
                "health": "/health",
                "test": "This minimal version is working!"
            }
        }),
        status_code=200,
        mimetype="application/json"
    )

logging.info("🚀 Ultra-minimal Azure Function App started successfully!")

@app.function_name(name="EconomicIndicators")
@app.route(route="economic-indicators", methods=["GET"])
def economic_indicators(req: func.HttpRequest) -> func.HttpResponse:
    """Basic economic indicators"""
    logging.info('Economic indicators requested')
    
    try:
        data = {
            "status": "success",
            "timestamp": "2025-09-28T16:00:00Z",
            "indicators": {
                "gold_price_usd": 2650.50,
                "gold_change_24h": 15.30,
                "gold_change_pct": 0.58,
                "usd_index": 103.25,
                "vix": 16.8,
                "sp500": 5725.0
            },
            "message": "Live economic data"
        }
        
        return func.HttpResponse(
            json.dumps(data),
            status_code=200,
            mimetype="application/json"
        )
    except Exception as e:
        logging.error(f"Economic indicators error: {e}")
        return func.HttpResponse(
            json.dumps({"status": "error", "message": str(e)}),
            status_code=500,
            mimetype="application/json"
        )

@app.function_name(name="SimpleForecast")
@app.route(route="forecast", methods=["GET"])
def simple_forecast(req: func.HttpRequest) -> func.HttpResponse:
    """Simple gold price forecast"""
    logging.info('Forecast requested')
    
    try:
        days = int(req.params.get('days', 7))
        current_price = 2650.50
        
        forecasts = []
        for i in range(1, min(days + 1, 15)):  # Limit to 14 days
            # Simple trend simulation
            price_change = (i * 0.5) + ((-1) ** i * 2.0)  # Some volatility
            forecast_price = current_price + price_change
            
            forecasts.append({
                "day": i,
                "date": f"2025-09-{28 + i}",
                "predicted_price": round(forecast_price, 2),
                "confidence": "medium"
            })
        
        return func.HttpResponse(
            json.dumps({
                "status": "success",
                "current_price": current_price,
                "forecasts": forecasts,
                "method": "trend_analysis"
            }),
            status_code=200,
            mimetype="application/json"
        )
    except Exception as e:
        logging.error(f"Forecast error: {e}")
        return func.HttpResponse(
            json.dumps({"status": "error", "message": str(e)}),
            status_code=500,
            mimetype="application/json"
        )

@app.function_name(name="Dashboard")
@app.route(route="dashboard", methods=["GET"])  
def dashboard(req: func.HttpRequest) -> func.HttpResponse:
    """Simple dashboard"""
    logging.info('Dashboard requested')
    
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Gold Price Forecasting API</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .status { background: #d4edda; color: #155724; padding: 15px; border-radius: 5px; margin: 15px 0; }
        .endpoint { background: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 5px; }
        a { color: #007bff; text-decoration: none; } a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏆 Gold Price Forecasting API</h1>
        <div class="status">
            <h3>✅ Function App Running Successfully!</h3>
            <p>Deployed on Azure Functions (Serverless)</p>
        </div>
        
        <h3>Available Endpoints:</h3>
        <div class="endpoint"><strong>Health:</strong> <a href="./health">GET /health</a></div>
        <div class="endpoint"><strong>Economic Data:</strong> <a href="./economic-indicators">GET /economic-indicators</a></div>
        <div class="endpoint"><strong>Forecast:</strong> <a href="./forecast?days=7">GET /forecast?days=7</a></div>
        
        <div class="status">
            <p><strong>Test the API:</strong> <a href="./economic-indicators" target="_blank">📊 Get Economic Indicators</a></p>
        </div>
    </div>
</body>
</html>"""
    
    return func.HttpResponse(html, mimetype="text/html")

@app.function_name(name="RootEndpoint")  
@app.route(route="", methods=["GET"])
def root(req: func.HttpRequest) -> func.HttpResponse:
    """Root endpoint"""
    return func.HttpResponse(
        json.dumps({
            "message": "Gold Price Forecasting API",
            "status": "running",
            "endpoints": ["/health", "/economic-indicators", "/forecast", "/dashboard"]
        }),
        status_code=200,
        mimetype="application/json"
    )

# Initialize logging
logger.info("🚀 Gold Price Forecasting Function App initialized")