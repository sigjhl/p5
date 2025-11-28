"""
Logging utilities for pipeline execution tracking and cost calculation.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import uuid

from ..config import MODEL_PRICING
from ..models.schemas import StageMetrics, TokenUsage, PipelineLog


class LoggingManager:
    """Manages logging and cost tracking for pipeline execution."""
    
    def __init__(self, pipeline_id: str = None, source_pdf: str = None):
        self.pipeline_id = pipeline_id or str(uuid.uuid4())
        self.source_pdf = source_pdf or "unknown"
        self.stage_metrics: List[StageMetrics] = []
        self.start_time: Optional[datetime] = None  # Will be set when first stage is logged
        self.end_time: Optional[datetime] = None    # Will be set when each stage is logged
        self.cache_name: Optional[str] = None
        
    def set_cache_name(self, cache_name: str) -> None:
        """Set the cache name used for this pipeline run."""
        self.cache_name = cache_name
    
    def _calculate_cost(self, token_usage: TokenUsage, model_name: str, is_large_prompt: bool = False, is_audio: bool = False) -> float:
        """Calculate cost for a given token usage and model."""
        if model_name not in MODEL_PRICING:
            return 0.0
        
        pricing = MODEL_PRICING[model_name]
        
        if is_large_prompt and "input_cost_per_token_large" in pricing:
            input_rate = pricing["input_cost_per_token_large"]
            cached_rate = pricing.get("cached_input_cost_per_token_large", pricing.get("cached_input_cost_per_token", 0))
        elif is_audio and "input_cost_per_token_audio" in pricing:
            input_rate = pricing["input_cost_per_token_audio"]
            cached_rate = pricing.get("cached_input_cost_per_token_audio", pricing.get("cached_input_cost_per_token", 0))
        else:
            input_rate = pricing["input_cost_per_token"]
            cached_rate = pricing.get("cached_input_cost_per_token", 0)
        
        reasoning_tokens = token_usage.total_tokens - token_usage.prompt_tokens - token_usage.candidate_tokens
        reasoning_tokens = max(0, reasoning_tokens)  # Ensure non-negative
        
        input_cost = (token_usage.prompt_tokens - token_usage.cached_tokens) * input_rate
        cached_cost = token_usage.cached_tokens * cached_rate
        
        if is_large_prompt and "output_cost_per_token_large" in pricing:
            output_rate = pricing["output_cost_per_token_large"]
        elif is_audio and "output_cost_per_token_audio" in pricing:
            output_rate = pricing["output_cost_per_token_audio"]
        else:
            output_rate = pricing["output_cost_per_token"]
        
        total_output_tokens = token_usage.candidate_tokens + reasoning_tokens
        output_cost = total_output_tokens * output_rate
        
        total_cost = input_cost + cached_cost + output_cost
        return round(total_cost, 6)
    
    def log_stage_metrics(
        self, 
        stage_name: str, 
        token_usage: TokenUsage, 
        model_name: str,
        execution_time: float = None,
        cache_hit: bool = False,
        is_large_prompt: bool = False,
        is_audio: bool = False
    ) -> None:
        """Log metrics for a pipeline stage."""
        if execution_time is None:
            execution_time = 0.0
        
        if self.start_time is None:
            self.start_time = datetime.now()
        
        stage_completion_time = datetime.now()
        self.end_time = stage_completion_time
        
        cost = self._calculate_cost(token_usage, model_name, is_large_prompt, is_audio)
        
        stage_metric = StageMetrics(
            stage_name=stage_name,
            execution_time_seconds=execution_time,
            token_usage=token_usage,
            cost_usd=cost,
            model_used=model_name,
            cache_hit=cache_hit
        )
        
        self.stage_metrics.append(stage_metric)
    
    def merge_metrics_from_logger(self, other_logger: 'LoggingManager') -> None:
        """
        Merge stage metrics from another LoggingManager instance.
        
        Args:
            other_logger: Another LoggingManager instance to merge metrics from
        """
        for metric in other_logger.stage_metrics:
            self.stage_metrics.append(metric)
    
    def get_total_cost(self) -> float:
        """Calculate total cost across all stages."""
        return sum(stage.cost_usd for stage in self.stage_metrics)
    
    def get_total_execution_time(self) -> float:
        """Calculate total execution time accounting for parallel TTS processing."""
        script_time = self.get_script_generation_time()
        
        tts_time = self._calculate_total_tts_time()
        
        return script_time + tts_time
    
    def get_category_execution_time(self, category_keywords: List[str]) -> float:
        """Calculate execution time for stages matching category keywords."""
        total_time = 0.0
        for stage in self.stage_metrics:
            if any(keyword in stage.stage_name.lower() for keyword in category_keywords):
                total_time += stage.execution_time_seconds
        return total_time
    
    def get_script_generation_time(self) -> float:
        """Get total time for script generation stages."""
        script_keywords = ["script_generation", "script_length_correction", "script_evaluation", "script_revision", "claim_extraction", "fact_verification", "content_appropriateness", "phoneme_identification", "phoneme_modification"]
        return self.get_category_execution_time(script_keywords)
    
    def _calculate_total_tts_time(self) -> float:
        """Calculate total TTS time accounting for parallel processing (max chunk time, not sum)."""
        return self.get_tts_generation_time()
    
    def get_tts_generation_time(self) -> float:
        """Get time for TTS generation stages (max total chunk time including all attempts)."""
        import re
        from collections import defaultdict
        
        chunk_times = defaultdict(float)
        
        for stage in self.stage_metrics:
            stage_name = stage.stage_name.lower()
            if "tts_chunk" in stage_name:
                match = re.match(r'tts_chunk_(\d+)', stage_name)
                if match:
                    chunk_id = match.group(1)
                    chunk_times[chunk_id] += stage.execution_time_seconds
        
        return max(chunk_times.values()) if chunk_times else 0.0
    
    def create_pipeline_log(self, status: str = "completed", error_message: str = None) -> PipelineLog:
        """Create a complete pipeline log."""
        actual_start_time = self.start_time or datetime.now()
        actual_end_time = self.end_time or datetime.now()
        total_time = self.get_total_execution_time()
        
        return PipelineLog(
            pipeline_id=self.pipeline_id,
            source_pdf=self.source_pdf,
            cache_name=self.cache_name or "unknown",
            start_time=actual_start_time,
            end_time=actual_end_time,
            total_execution_time_seconds=total_time,
            stage_metrics=self.stage_metrics,
            total_cost_usd=self.get_total_cost(),
            status=status,
            error_message=error_message
        )
    
    def save_pipeline_log(self, output_directory: str) -> str:
        """Save pipeline log to file."""
        log = self.create_pipeline_log()
        log_path = os.path.join(output_directory, "pipeline_log.json")
        
        os.makedirs(output_directory, exist_ok=True)
        
        with open(log_path, 'w') as f:
            json.dump(log.model_dump(), f, indent=2, default=str)
        
        return log_path
    
    def print_summary(self) -> None:
        """Print a summary of the pipeline execution."""
        total_cost = self.get_total_cost()
        total_time = self.get_total_execution_time()
        script_time = self.get_script_generation_time()
        tts_time = self.get_tts_generation_time()
        
        print(f"\n=== Pipeline Summary ===")
        print(f"Pipeline ID: {self.pipeline_id}")
        print(f"Source PDF: {self.source_pdf}")
        print(f"Total Stages: {len(self.stage_metrics)}")
        print(f"Total Cost: ${total_cost:.6f}")
        print(f"Total Time: {total_time:.2f} seconds")
        
        print(f"\n--- Timing Breakdown ---")
        print(f"Script Generation: {script_time:.2f}s ({script_time/total_time*100:.1f}%)")
        print(f"TTS Generation: {tts_time:.2f}s ({tts_time/total_time*100:.1f}%)")
        other_time = total_time - script_time - tts_time
        if other_time > 0:
            print(f"Other Operations: {other_time:.2f}s ({other_time/total_time*100:.1f}%)")
        
        print(f"\n--- Stage Breakdown ---")
        for stage in self.stage_metrics:
            print(f"{stage.stage_name}:")
            print(f"  Time: {stage.execution_time_seconds:.2f}s")
            print(f"  Cost: ${stage.cost_usd:.6f}")
            print(f"  Tokens: {stage.token_usage.total_tokens} total")
            print(f"  Model: {stage.model_used}")
            if stage.cache_hit:
                print(f"  Cache: HIT ({stage.token_usage.cached_tokens} cached tokens)")
            print()

    def log_info(self, message: str) -> None:
        """Log an info message."""
        print(f"ℹ️  {message}")
    
    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        print(f"⚠️  {message}")
    
    def log_error(self, message: str) -> None:
        """Log an error message."""
        print(f"❌ {message}")