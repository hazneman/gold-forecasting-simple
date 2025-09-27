import azure.functions as func
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import sys
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src directory to Python path
current_dir = Path(__file__).parent
src_path = current_dir / "src"
api_path = current_dir / "api"
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(api_path))
sys.path.insert(0, str(current_dir))

# Import your existing components
try:
    from enhanced_auto_trainer import enhanced_trainer
    from simple_economic_data import SimpleEconomicDataCollector
    ML_AVAILABLE = True
    logger.info("✅ ML components loaded successfully")
except ImportError as e:
    logger.error(f"❌ ML imports failed: {e}")
    ML_AVAILABLE = False

# Create FastAPI app
app = FastAPI(
    title="Gold Price Forecasting API", 
    version="2.0.0",
    description="Professional serverless gold price forecasting with 92 ML features"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/")
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "Gold Price Forecasting API",
        "version": "1.0.0",
        "environment": "Azure Functions"
    }

# Economic indicators
@app.get("/api/economic-indicators")
async def get_economic_indicators():
    try:
        collector = SimpleEconomicDataCollector()
        data = collector.get_latest_indicators()
        return {"status": "success", "data": data}
    except Exception as e:
        logging.error(f"Economic indicators error: {e}")
        return {"status": "error", "message": str(e)}

# Extended predictions endpoint - main feature!
@app.get("/api/extended-predictions")
async def get_extended_predictions():
    if not ML_AVAILABLE:
        return {"status": "error", "message": "ML libraries not available"}
    
    try:
        logger.info("🔮 Generating extended predictions for all 14 horizons...")
        predictions = enhanced_trainer.get_extended_predictions()
        logger.info(f"✅ Generated predictions for {len(predictions.get('predictions', {}))} horizons")
        return predictions
    except Exception as e:
        logger.error(f"Extended predictions error: {e}")
        return {"status": "error", "message": str(e)}

# Training endpoints
@app.post("/api/start-extended-training")
async def start_extended_training(force_retrain: bool = False, period: str = "5y"):
    try:
        result = enhanced_trainer.train_extended_models_with_progress(
            period=period, 
            force_retrain=force_retrain
        )
        return result
    except Exception as e:
        logging.error(f"Training error: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/api/training-progress")
async def get_training_progress():
    try:
        return enhanced_trainer.get_training_progress()
    except Exception as e:
        logging.error(f"Training progress error: {e}")
        return {"status": "error", "message": str(e)}

# Cache status
@app.get("/api/cache-status")
async def get_cache_status():
    try:
        model_status = enhanced_trainer.check_models_exist()
        cache_metadata = enhanced_trainer.load_cache_metadata()
        
        fresh_models = sum(1 for status in model_status.values() if status['fresh'])
        total_models = len(model_status)
        
        return {
            "status": "success",
            "cache_valid": fresh_models == total_models,
            "fresh_models": fresh_models,
            "total_models": total_models,
            "model_status": model_status,
            "cache_metadata": cache_metadata
        }
    except Exception as e:
        logging.error(f"Cache status error: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/dashboard", response_class=HTMLResponse)
@app.get("/api/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    try:
        dashboard_file = current_dir / "economic_dashboard.html"
        if dashboard_file.exists():
            content = dashboard_file.read_text()
            # Update API URLs for Azure Functions
            content = content.replace("http://localhost:8000", "")
            return HTMLResponse(content=content)
        else:
            # Return a simple status page
            return HTMLResponse(
                content=f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>🏆 Gold Price Forecasting API</title>
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                        .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                        .status {{ padding: 20px; border-radius: 10px; margin: 20px 0; }}
                        .success {{ background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }}
                        .info {{ background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }}
                        .endpoint {{ background: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0; }}
                        a {{ color: #007bff; text-decoration: none; }}
                        a:hover {{ text-decoration: underline; }}
                        code {{ background: #f1f1f1; padding: 2px 6px; border-radius: 3px; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>🏆 Gold Price Forecasting API</h1>
                        
                        <div class="status success">
                            <h3>✅ Serverless API Running Successfully!</h3>
                            <p><strong>Platform:</strong> Azure Functions</p>
                            <p><strong>Features:</strong> 92 ML features (Technical + Economic)</p>
                            <p><strong>Economic Data:</strong> Fed funds rate, inflation, DXY, VIX</p>
                            <p><strong>Prediction Horizons:</strong> 14 timeframes (1 day to 1 year)</p>
                            <p><strong>ML Available:</strong> {"✅ Yes" if ML_AVAILABLE else "❌ Loading..."}</p>
                        </div>
                        
                        <div class="status info">
                            <h3>🔗 API Endpoints</h3>
                            <div class="endpoint">
                                <strong>Health Check:</strong> 
                                <a href="/api/health" target="_blank"><code>GET /api/health</code></a>
                            </div>
                            <div class="endpoint">
                                <strong>Extended Predictions (Main Feature):</strong> 
                                <a href="/api/extended-predictions" target="_blank"><code>GET /api/extended-predictions</code></a>
                            </div>
                            <div class="endpoint">
                                <strong>Economic Indicators:</strong> 
                                <a href="/api/economic-indicators" target="_blank"><code>GET /api/economic-indicators</code></a>
                            </div>
                        </div>
                        
                        <div class="status info">
                            <h3>🚀 Quick Test</h3>
                            <p>Try the main prediction endpoint:</p>
                            <p><a href="/api/extended-predictions" target="_blank" style="font-size: 18px; font-weight: bold;">📊 Get Gold Price Predictions</a></a></p>
                        </div>
                    </div>
                </body>
                </html>
                """,
                status_code=200
            )
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return HTMLResponse(
            content=f"<h1>Dashboard Error</h1><p>{str(e)}</p>", 
            status_code=500
        )

# Create Azure Functions app
azure_app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# Main HTTP trigger - routes all requests to FastAPI
@azure_app.route(route="{*route}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def main_trigger(req: func.HttpRequest) -> func.HttpResponse:
    """Main Azure Functions HTTP trigger that routes all requests to FastAPI"""
    try:
        # Create ASGI middleware and handle the request
        asgi_middleware = func.AsgiMiddleware(app)
        return await asgi_middleware.handle_async(req)
    except Exception as e:
        logging.error(f"Azure Functions error: {e}")
        return func.HttpResponse(
            body=f'{{"error": "Internal server error: {str(e)}"}}',
            status_code=500,
            mimetype="application/json"
        )

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("🚀 Gold Price Forecasting Azure Functions app initialized")