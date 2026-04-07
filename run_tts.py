#!/usr/bin/env python3
"""
Generate audio from an existing script file.
Simple entry point that doesn't require PYTHONPATH.
"""

import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.pipeline import PodcastPipeline

def main():
    """Main entry point for TTS generation from script."""
    if len(sys.argv) < 2:
        print("Usage: python run_tts.py <script_path> [output_dir]")
        print("Note: Set GEMINI_API_KEY environment variable")
        print("\nExample:")
        print("  export GEMINI_API_KEY=your_api_key")
        print("  python run_tts.py ./script_final.txt")
        print("  python run_tts.py ./script_final.txt ./audio_output")
        sys.exit(1)
    
    script_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(script_path):
        print(f"Error: Script file does not exist: {script_path}")
        sys.exit(1)
    
    try:
        pipeline = PodcastPipeline()  # Will use GEMINI_API_KEY from environment
        
        wav_path, mp3_path = pipeline.generate_audio_from_script(script_path, output_dir)
        
        print(f"\nAudio generated successfully!")
        print(f"WAV file: {wav_path}")
        if mp3_path != wav_path:
            print(f"MP3 file: {mp3_path}")
    except Exception as e:
        print(f"TTS generation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()