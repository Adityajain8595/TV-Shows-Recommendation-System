from flask import Flask
from flask_cors import CORS
from config import Config

app = None

def create_app(config_class=Config):
    global app
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    CORS(app)
    
    # Initialize data - The model initializes itself when imported
    from app.models import recommendation_model
    
    # Register routes
    from app.routes import bp
    app.register_blueprint(bp)
    
    return app