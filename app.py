from flask import Flask
from flask_cors import CORS
from core.api_routes import api_bp
from config import Config

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # เปิด CORS
    CORS(app)
    
    # ลงทะเบียน Blueprint
    app.register_blueprint(api_bp, url_prefix='/api')
    
    return app

app = create_app()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)