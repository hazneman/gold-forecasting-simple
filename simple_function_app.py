import azure.functions as func
import logging
import json
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Azure Functions app
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="health", methods=["GET"])
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """Simple health check endpoint"""
    try:
        return func.HttpResponse(
            json.dumps({
                "status": "healthy",
                "service": "Gold Price Forecasting API",
                "version": "1.0.0",
                "environment": "Azure Functions"
            }),
            status_code=200,
            mimetype="application/json"
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return func.HttpResponse(
            json.dumps({"status": "error", "message": str(e)}),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="economic-indicators", methods=["GET"])
def economic_indicators(req: func.HttpRequest) -> func.HttpResponse:
    """Get basic economic indicators"""
    try:
        # Simple mock data for now - will be replaced with real data
        indicators = {
            "gold_price": 2650.0,
            "usd_index": 103.2,
            "vix": 16.8,
            "sp500": 5720.0,
            "timestamp": "2025-09-28T12:00:00Z",
            "status": "success"
        }
        
        return func.HttpResponse(
            json.dumps(indicators),
            status_code=200,
            mimetype="application/json"
        )
    except Exception as e:
        logger.error(f"Economic indicators error: {e}")
        return func.HttpResponse(
            json.dumps({"status": "error", "message": str(e)}),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="simple-forecast", methods=["GET"])
def simple_forecast(req: func.HttpRequest) -> func.HttpResponse:
    """Get simple gold price forecast"""
    try:
        days = int(req.params.get('days', 7))
        
        # Simple forecast data
        forecasts = []
        base_price = 2650.0
        
        for i in range(1, days + 1):
            forecast_price = base_price * (1 + 0.001 * i)  # Simple trend
            forecasts.append({
                "date": f"2025-09-{28 + i}",
                "predicted_price": round(forecast_price, 2),
                "confidence": "medium"
            })
        
        return func.HttpResponse(
            json.dumps({
                "status": "success",
                "forecasts": forecasts,
                "current_price": base_price,
                "forecast_method": "simple_trend"
            }),
            status_code=200,
            mimetype="application/json"
        )
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        return func.HttpResponse(
            json.dumps({"status": "error", "message": str(e)}),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="dashboard", methods=["GET"])
def dashboard(req: func.HttpRequest) -> func.HttpResponse:
    """Serve a simple dashboard page"""
    try:
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>🏆 Gold Price Forecasting API</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                .container { max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                .status { padding: 20px; border-radius: 10px; margin: 20px 0; }
                .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
                .endpoint { background: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0; }
                a { color: #007bff; text-decoration: none; }
                a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🏆 Gold Price Forecasting API</h1>
                
                <div class="status success">
                    <h3>✅ Azure Functions API Running!</h3>
                    <p><strong>Status:</strong> Deployed Successfully</p>
                    <p><strong>Platform:</strong> Azure Functions (Serverless)</p>
                    <p><strong>Version:</strong> 1.0.0</p>
                </div>
                
                <h3>🔗 Available Endpoints:</h3>
                <div class="endpoint">
                    <strong>Health Check:</strong> 
                    <a href="./health" target="_blank">GET /health</a>
                </div>
                <div class="endpoint">
                    <strong>Economic Indicators:</strong> 
                    <a href="./economic-indicators" target="_blank">GET /economic-indicators</a>
                </div>
                <div class="endpoint">
                    <strong>Simple Forecast:</strong> 
                    <a href="./simple-forecast?days=7" target="_blank">GET /simple-forecast?days=7</a>
                </div>
                
                <div class="status success">
                    <h3>🚀 Quick Test</h3>
                    <p><a href="./economic-indicators" target="_blank" style="font-size: 18px;">📊 Test Economic Indicators</a></p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return func.HttpResponse(
            html_content,
            status_code=200,
            mimetype="text/html"
        )
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return func.HttpResponse(
            f"<h1>Dashboard Error</h1><p>{str(e)}</p>", 
            status_code=500,
            mimetype="text/html"
        )

# Catch-all route for root
@app.route(route="", methods=["GET"])
def root(req: func.HttpRequest) -> func.HttpResponse:
    """Root endpoint - redirect to dashboard"""
    return func.HttpResponse(
        json.dumps({
            "message": "Gold Price Forecasting API",
            "version": "1.0.0",
            "endpoints": {
                "health": "/health",
                "dashboard": "/dashboard", 
                "economic_indicators": "/economic-indicators",
                "forecast": "/simple-forecast?days=7"
            }
        }),
        status_code=200,
        mimetype="application/json"
    )

logger.info("🚀 Azure Functions Gold Price Forecasting API initialized")