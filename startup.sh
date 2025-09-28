# Azure Web App startup command
gunicorn --bind 0.0.0.0:$PORT simple_app:app