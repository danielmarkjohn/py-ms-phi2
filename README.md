# Local GPT API Server

A Flask-based API server for running the DeepSeek-R1 language model locally with optimized performance.

## Features

- Local model caching for faster loading
- 8-bit quantization support for reduced memory usage
- Configurable model parameters
- RESTful API interface

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure settings in `config.py`:
- Adjust model parameters
- Configure hardware acceleration
- Set server options

## Usage

1. Start the server:
```bash
python app.py
```

2. Send requests to the chat endpoint:
```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Your prompt here"}'
```

## Performance Optimizations

- Model caching for faster loading
- 8-bit quantization support (enable in config.py)
- Optional 4-bit quantization for even lower memory usage
- Automatic CUDA detection and utilization
- Optimized generation parameters

## Configuration

Edit `config.py` to customize:
- Model settings (cache directory, model name)
- Generation parameters (max length, temperature, etc.)
- Hardware acceleration options
- Server settings
