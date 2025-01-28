import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import Config
import logging
from typing import Tuple, Optional
from dataclasses import dataclass

# Structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

@dataclass
class GenerationResult:
    response: str
    tokens_generated: int
    inference_time: float

class ModelManager:
    def __init__(self):
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        Config.ensure_cache_dir()
        
    def _download_model(self) -> None:
        """Download model from HuggingFace Hub."""
        logger.info(f"Downloading model {Config.MODEL_NAME}...")
        
        try:
            # Download and save tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                Config.MODEL_NAME,
                trust_remote_code=True,
                cache_dir=Config.CACHE_DIR,
                local_files_only=Config.LOCAL_FILES_ONLY,
            )
            
            # Download and save model
            self.model = AutoModelForCausalLM.from_pretrained(
                Config.MODEL_NAME,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                cache_dir=Config.CACHE_DIR,
                local_files_only=Config.LOCAL_FILES_ONLY,
            )
            
            logger.info("Model downloaded successfully!")
        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}", exc_info=True)
            raise RuntimeError("Failed to download model") from e
        
    def load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load model with offline mode enforced."""
        logger.info("Loading model and tokenizer...")
        
        try:
            if Config.OFFLINE_MODE:
                # Try to load from cache
                cache_path = os.path.join(Config.CACHE_DIR, 'models--microsoft--phi-2', 'snapshots')
                if not os.path.exists(cache_path):
                    raise RuntimeError(
                        f"Model not found in cache directory: {cache_path}\n"
                        "Please run the application in online mode first to download the model."
                    )
                
                # Get the hash directory (should be the only directory in snapshots)
                hash_dirs = [d for d in os.listdir(cache_path) if os.path.isdir(os.path.join(cache_path, d))]
                if not hash_dirs:
                    raise RuntimeError("No model snapshot found in cache")
                
                model_path = os.path.join(cache_path, hash_dirs[0])
                logger.info(f"Loading model from cache: {model_path}")
                
                # Load tokenizer and model from cache
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    local_files_only=True,
                    trust_remote_code=True,
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    local_files_only=True,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                )
            else:
                # Download model if not in offline mode
                self._download_model()
            
            # Move model to the specified device
            self.model.to(Config.DEVICE)
            logger.info(f"Model loaded successfully on device: {Config.DEVICE}")
            return self.model, self.tokenizer
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            raise RuntimeError("Failed to load model") from e
    
    def generate_response(self, prompt: str) -> GenerationResult:
        """Generate response with optimized settings."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded before generating responses")
            
        try:
            # Encode input prompt
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(Config.DEVICE)
            
            # Generate response with timing
            start_time = torch.cuda.Event(enable_timing=True) if Config.DEVICE == "cuda" else None
            end_time = torch.cuda.Event(enable_timing=True) if Config.DEVICE == "cuda" else None
            
            if start_time:
                start_time.record()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=Config.MAX_LENGTH,
                    num_return_sequences=Config.NUM_RETURN_SEQUENCES,
                    temperature=Config.TEMPERATURE,
                    top_p=Config.TOP_P,
                    top_k=Config.TOP_K,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=Config.REPETITION_PENALTY,
                )
                
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                inference_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
            else:
                inference_time = None
            
            # Decode and return response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            tokens_generated = len(outputs[0])
            
            return GenerationResult(
                response=response,
                tokens_generated=tokens_generated,
                inference_time=inference_time,
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            raise RuntimeError("Failed to generate response") from e
    
    def warm_up_model(self) -> None:
        """Warm up the model with a test prompt."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded before warming up")
        
        logger.info("Warming up model...")
        try:
            self.generate_response(Config.WARM_UP_PROMPT)
            logger.info("Model warm-up completed successfully!")
        except Exception as e:
            logger.error(f"Error during model warm-up: {str(e)}", exc_info=True)
            raise RuntimeError("Failed to warm up model") from e