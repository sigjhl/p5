# P5

The codebase for the P5 paper. 

## Installation

### Prerequisites
- Python 3.8+
- Google Gemini API key
- OpenAI API key
- ffmpeg (optional, for MP3 conversion)

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install ffmpeg for MP3 conversion
# On macOS: brew install ffmpeg
# On Ubuntu: sudo apt install ffmpeg
# On Windows: Download from https://ffmpeg.org/
```

### Configuration
1. Obtain a Google AI API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Obtain an OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys)
3. Set the environment variables:
   ```bash
   export GEMINI_API_KEY=your_gemini_api_key_here
   export OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

```bash
# Set API keys (once per session)
export GEMINI_API_KEY=your_gemini_api_key_here
export OPENAI_API_KEY=your_openai_api_key_here

# Basic usage (full pipeline: PDF → script → audio)
python run_pipeline.py path/to/paper.pdf

# With custom word count
python run_pipeline.py path/to/paper.pdf 1500

# Script generation only (no audio)
python run_pipeline.py path/to/paper.pdf 1500 --script-only

# With custom output directory
python run_pipeline.py path/to/paper.pdf 1500 --output-dir ./my_output
```

