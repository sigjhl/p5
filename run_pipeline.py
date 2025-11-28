#!/usr/bin/env python3
"""
Entry point script for the podcast generation pipeline.
This script can be run without setting PYTHONPATH.
"""

import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.pipeline import PodcastPipeline
from src.config import DEFAULT_TARGET_WORD_LENGTH

def main():
    """Main entry point for pipeline execution."""
    if len(sys.argv) < 2:
        print("Usage: python run_pipeline.py <pdf_path> [target_word_length] [--script-only] [--output-dir <path>]")
        print("Note: Set GEMINI_API_KEY environment variable")
        print("\nFlags:")
        print("  --script-only      Generate script only (no audio)")
        print("  --output-dir <path> Specify output directory")
        print("\nExamples:")
        print("  export GEMINI_API_KEY=your_api_key")
        print("  python run_pipeline.py ./paper.pdf 1500")
        print("  python run_pipeline.py ./paper.pdf 1500 --script-only")
        print("  python run_pipeline.py ./paper.pdf 1500 --output-dir ./my_output")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    script_only = False
    target_length = DEFAULT_TARGET_WORD_LENGTH
    output_dir = None
    
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--script-only":
            script_only = True
            i += 1
        elif arg == "--output-dir" and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
            i += 2
        elif arg.isdigit():
            target_length = int(arg)
            i += 1
        else:
            i += 1
    
    try:
        pipeline = PodcastPipeline()  # Will use GEMINI_API_KEY from environment
        
        if script_only:
            result_output_dir, script_path = pipeline.run_script_only(pdf_path, target_length, output_dir)
            print(f"\nScript generated successfully!")
            print(f"Script saved to: {script_path}")
        else:
            result_output_dir, script_path, wav_path, mp3_path = pipeline.run_pipeline(
                pdf_path, target_length, output_dir
            )
            print(f"\nPipeline completed successfully!")
            print(f"Script: {script_path}")
            if wav_path:
                print(f"Audio: {wav_path}")
        
        print(f"Output directory: {result_output_dir}")
        print("Using structured outputs with explicit caching")
    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()