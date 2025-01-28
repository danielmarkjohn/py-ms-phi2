from flask import Flask, request, jsonify
from model_utils import ModelManager
from config import Config
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
model_manager = ModelManager()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model_manager.model is not None,
        "offline_mode": Config.OFFLINE_MODE
    })

@app.route('/chat', methods=['POST'])
def chat():
    """Chat endpoint for text generation"""
    try:
        data = request.json
        if not data or 'message' not in data:
            return jsonify({"error": "Missing 'message' in request"}), 400
            
        prompt = data.get("message", "")
        response = model_manager.generate_response(prompt)
        return jsonify({"response": response})
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

def initialize_app():
    """Initialize the application and load model"""
    try:
        logger.info("Initializing application in offline mode...")
        model_manager.load_model()
        logger.info("Application initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise

if __name__ == '__main__':
    initialize_app()
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
