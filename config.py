import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    # Base directory for all model files
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    
    # Model settings
    MODEL_NAME: str = "microsoft/phi-2"  # CPU-friendly model
    CACHE_DIR: str = os.path.join(BASE_DIR, "model_cache")
    OFFLINE_MODE: bool = True  # Offline mode by default
    LOCAL_FILES_ONLY: bool = True  # Use only local files
    
    # Generation settings
    MAX_LENGTH: int = 150  # Maximum tokens in the generated response
    NUM_RETURN_SEQUENCES: int = 1  # Number of responses to generate
    TEMPERATURE: float = 0.7  # Sampling temperature
    TOP_P: float = 0.9  # Nucleus sampling
    TOP_K: int = 50  # Top-k sampling
    REPETITION_PENALTY: float = 1.2  # Penalty for repeated tokens
    
    # Performance settings
    USE_8BIT: bool = False  # Disable 8-bit quantization for CPU
    USE_4BIT: bool = False  # Disable 4-bit quantization for CPU
    USE_CACHE: bool = True  # Enable model caching
    DEVICE: str = "cpu"  # Force CPU usage
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 5000
    DEBUG: bool = False
    MAX_REQUEST_LENGTH: int = 1000  # Maximum characters in the input prompt
    RATE_LIMIT: str = "10 per minute"  # Rate limiting for API endpoints
    
    # Warm-up settings
    WARM_UP_PROMPT: str = "Hello, how can I assist you today?"
    WARM_UP_MAX_LENGTH: int = 20  # Shorter warm-up prompt
    
    @classmethod
    def ensure_cache_dir(cls) -> None:
        """Ensure the cache directory exists."""
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        
    @classmethod
    def get_model_path(cls) -> str:
        """Get the full path to the cached model."""
        return os.path.join(cls.CACHE_DIR, cls.MODEL_NAME.replace("/", "_"))
    
    @classmethod
    def validate_config(cls) -> None:
        """Validate configuration values."""
        if cls.TEMPERATURE <= 0 or cls.TEMPERATURE > 2.0:
            raise ValueError("Temperature must be between 0 and 2.0")
        if cls.MAX_LENGTH <= 0 or cls.MAX_LENGTH > 1024:
            raise ValueError("MAX_LENGTH must be between 1 and 1024")
        if cls.TOP_P <= 0 or cls.TOP_P > 1.0:
            raise ValueError("TOP_P must be between 0 and 1.0")
        if cls.TOP_K <= 0:
            raise ValueError("TOP_K must be greater than 0")