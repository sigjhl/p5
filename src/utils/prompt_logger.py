"""
Utility for logging all LLM prompts and responses to files.
Creates a detailed audit trail of all AI interactions.
"""

import os
import json
from datetime import datetime
from typing import Any, Dict, Optional

from ..config import DEFAULT_MODEL

class PromptLogger:
    """Logs prompts and responses for debugging and audit purposes."""
    
    def __init__(self, output_dir: str):
        """Initialize logger with output directory."""
        self.output_dir = output_dir
        self.logs_dir = os.path.join(output_dir, "llm_logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        self.interaction_count = 0
    
    def log_interaction(
        self, 
        stage: str, 
        prompt: str, 
        response: Any, 
        model: str = DEFAULT_MODEL,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Log a single LLM interaction.
        
        Args:
            stage: The pipeline stage (e.g., "script_generation", "qa_judge")
            prompt: The full prompt sent to the LLM
            response: The response object from the LLM
            model: The model used
            metadata: Additional metadata about the interaction
        """
        self.interaction_count += 1
        timestamp = datetime.now().strftime("%H%M%S")
        
        filename = f"{self.interaction_count:02d}_{timestamp}_{stage}.json"
        filepath = os.path.join(self.logs_dir, filename)
        
        response_text = ""
        if hasattr(response, 'text'):
            response_text = response.text
        elif hasattr(response, 'parsed'):
            response_text = str(response.parsed)
        else:
            response_text = str(response)
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "interaction_id": self.interaction_count,
            "stage": stage,
            "model": model,
            "prompt": prompt,
            "response": response_text,
            "metadata": metadata or {}
        }
        
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            prompt_details = getattr(usage, 'prompt_tokens_details', None)
            if prompt_details:
                prompt_details = [
                    {"modality": str(detail.modality), "token_count": detail.token_count}
                    for detail in prompt_details
                ]
            
            log_entry["usage"] = {
                "prompt_tokens": usage.prompt_token_count,
                "candidate_tokens": usage.candidates_token_count,
                "thoughts_tokens": getattr(usage, 'thoughts_token_count', None),
                "total_tokens": usage.total_token_count,
                "cached_content_tokens": getattr(usage, 'cached_content_token_count', None),
                "prompt_tokens_details": prompt_details,
                "traffic_type": str(getattr(usage, 'traffic_type', None)) if getattr(usage, 'traffic_type', None) else None
            }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)
        
        print(f"    [LOG] Saved interaction to: llm_logs/{filename}")
    
    def log_debug_text(self, stage: str, content: str, suffix: str = "") -> None:
        """
        Log plain text content for debugging purposes.
        
        Args:
            stage: The pipeline stage (e.g., "script_revision", "fact_verification")
            content: The text content to log
            suffix: Optional suffix for the filename (e.g., "before", "after")
        """
        timestamp = datetime.now().strftime("%H%M%S")
        
        suffix_part = f"_{suffix}" if suffix else ""
        filename = f"{self.interaction_count:02d}_{timestamp}_{stage}{suffix_part}.txt"
        filepath = os.path.join(self.logs_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"    [LOG] Saved debug text to: llm_logs/{filename}")
    
    def get_logs_summary(self) -> str:
        """Get a summary of logged interactions."""
        return f"Logged {self.interaction_count} LLM interactions to {self.logs_dir}"