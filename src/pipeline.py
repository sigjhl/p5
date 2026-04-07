"""
Main pipeline orchestration for podcast generation.
Coordinates all modules to convert PDF to final audio podcast.
"""

import os
import re
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import uuid

from google import genai

from .config import DEFAULT_TARGET_WORD_LENGTH, SCRIPT_LENGTH_TOLERANCE
from .utils.cache_manager import CacheManager
from .utils.logging_utils import LoggingManager
from .utils.prompt_logger import PromptLogger
from .modules.script_generator import ScriptGenerator
from .modules.quality_assurance import QualityAssurance
from .modules.phoneme_enhancer import PhonemeEnhancer
from .modules.tts_generator import TTSGenerator


class PodcastPipeline:
    """Main pipeline for converting PDF to podcast audio."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the pipeline with API key.
        
        Args:
            api_key: Google AI API key (if None, will use GEMINI_API_KEY environment variable)
        """
        if api_key is None:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("API key must be provided either as parameter or GEMINI_API_KEY environment variable")
        
        self.client = genai.Client(api_key=api_key)
        
        self.cache_manager = CacheManager()
        self.logger: Optional[LoggingManager] = None
        self.prompt_logger: Optional[PromptLogger] = None
        
    
    def _create_output_directory(self, pdf_path: str, base_output_dir: str = "output", user_id: str = None) -> str:
        """
        Create output directory with prominent PDF filename for this pipeline run.
        
        Args:
            pdf_path: Source PDF path
            base_output_dir: Base output directory
            user_id: Optional user ID to prefix the directory name
            
        Returns:
            Path to created output directory
        """
        filename_base = os.path.basename(pdf_path).replace('.pdf', '')
        filename_safe = re.sub(r'[^\w\-_]', '_', filename_base)
        filename_short = filename_safe[:40].rstrip('_')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if user_id:
            output_dir = os.path.join(base_output_dir, f"{filename_short}_{user_id}_{timestamp}")
        else:
            output_dir = os.path.join(base_output_dir, f"{filename_short}_{timestamp}")
        
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def _save_script(self, script: str, output_dir: str, stage: str = "final") -> str:
        """Save script to output directory."""
        script_path = os.path.join(output_dir, f"script_{stage}.txt")
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script)
        return script_path
    
    
    def run_pipeline(
        self,
        pdf_path: str,
        target_word_length: int = DEFAULT_TARGET_WORD_LENGTH,
        output_dir: str = None,
        skip_audio: bool = False,
        user_id: str = None,
        cancellation_check: callable = None,
        custom_instructions: str = None,
        host_a_persona: str = None,
        host_b_persona: str = None
    ) -> Tuple[str, str, Optional[str], Optional[str]]:
        """
        Run the complete pipeline from PDF to audio.
        
        Args:
            pdf_path: Path to source PDF file
            target_word_length: Target word count for script
            output_dir: Output directory (created if None)
            skip_audio: If True, skip TTS generation
            user_id: Optional user ID to prefix output directory name
            cancellation_check: Optional callable that returns True if pipeline should be cancelled
            
        Returns:
            Tuple of (output_directory, final_script_path, wav_path, mp3_path)
        """
        def check_cancellation():
            """Check if pipeline should be cancelled."""
            if cancellation_check and cancellation_check():
                print("⚠️ Pipeline cancellation requested - STOPPING NOW")
                raise InterruptedError("Pipeline was cancelled by user")
        
        pipeline_id = str(uuid.uuid4())
        pdf_filename = os.path.basename(pdf_path)
        self.logger = LoggingManager(pipeline_id, pdf_filename)
        
        if output_dir is None:
            output_dir = self._create_output_directory(pdf_path, user_id=user_id)
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        self.prompt_logger = PromptLogger(output_dir)
        
        try:
            
            print(f"Starting pipeline for {pdf_filename}")
            print(f"Output directory: {output_dir}")
            
            print("Initializing document cache...")
            cache_name = self.cache_manager.get_or_create_cache(pdf_path, self.client)
            self.logger.set_cache_name(cache_name)
            
            check_cancellation()
            
            script_generator = ScriptGenerator(self.client, self.logger, self.prompt_logger)
            quality_assurance = QualityAssurance(self.client, self.logger, self.prompt_logger)
            phoneme_enhancer = PhonemeEnhancer(self.client, self.logger, self.prompt_logger)
            
            print("Generating podcast script...")
            initial_script = script_generator.generate_script_with_length_correction(
                cache_name, target_word_length, 
                additional_instructions=None,
                host_a_persona=host_a_persona,
                host_b_persona=host_b_persona,
                custom_instructions=custom_instructions
            )
            print(f"✓ Script generated: {initial_script.word_count} words (target: {target_word_length})")
            
            check_cancellation()
            
            if initial_script.word_count != target_word_length:
                target_min = int(target_word_length * (1 - SCRIPT_LENGTH_TOLERANCE))
                target_max = int(target_word_length * (1 + SCRIPT_LENGTH_TOLERANCE))
                tolerance_percent = int(SCRIPT_LENGTH_TOLERANCE * 100)
                if target_min <= initial_script.word_count <= target_max:
                    print(f"✓ Script length is within acceptable range (±{tolerance_percent}%)")
                else:
                    print("⚠ Script length correction may have been applied")
            
            self._save_script(
                initial_script.script_text, output_dir, "initial"
            )
            
            print("Running quality assurance...")
            final_script_text = quality_assurance.run_full_qa_cycle(
                initial_script.script_text, cache_name, target_word_length
            )
            print("✓ Quality assurance completed")
            
            check_cancellation()
            
            human_script_path = self._save_script(final_script_text, output_dir, "human_readable")
            
            print("Enhancing script with phoneme tags...")
            enhanced_script = phoneme_enhancer.enhance_script_with_phonemes(final_script_text)
            print("✓ Phoneme enhancement completed")
            
            check_cancellation()
            
            
            tts_script = phoneme_enhancer.remove_words_from_phoneme_tags_for_tts(enhanced_script)
            
            final_script_path = self._save_script(tts_script, output_dir, "final")
            
            wav_path = None
            mp3_path = None
            
            if not skip_audio:
                check_cancellation()
                
                print("Generating audio...")
                tts_generator = TTSGenerator(self.client, self.logger, self.prompt_logger, phoneme_dict=phoneme_enhancer.get_phoneme_dictionary())
                
                if tts_generator.enable_quality_check:
                    print("  → TTS quality checking enabled with automatic retry for critical issues")
                
                tts_issues = tts_generator.validate_script_for_tts(tts_script)
                if tts_issues:
                    print("TTS validation warnings:")
                    for issue in tts_issues:
                        print(f"  - {issue}")
                
                wav_path, mp3_path = tts_generator.generate_audio_from_script(
                    tts_script, output_dir, "podcast_final", final_script_text
                )
                
                print(f"Audio generated: {wav_path}")
                if mp3_path != wav_path:
                    print(f"MP3 version: {mp3_path}")
                
                check_cancellation()
            
            log_path = self.logger.save_pipeline_log(output_dir)
            
            phoneme_enhancer.restore_original_dict_if_temporary()
            
            self.logger.print_summary()
            
            print(f"\nPipeline completed successfully!")
            print(f"Human-readable script: {human_script_path}")
            print(f"TTS script: {final_script_path}")
            if wav_path:
                print(f"Audio files: {wav_path}")
            print(f"Pipeline log: {log_path}")
            if self.prompt_logger:
                print(f"LLM interactions: {self.prompt_logger.get_logs_summary()}")
            
            return output_dir, final_script_path, wav_path, mp3_path
            
        except Exception as e:
            error_msg = str(e)
            print(f"Pipeline failed: {error_msg}")
            
            if self.logger:
                self.logger.save_pipeline_log(output_dir or "output/failed")
            
            raise
    
    def run_pipeline_without_phoneme(
        self,
        pdf_path: str,
        target_word_length: int = DEFAULT_TARGET_WORD_LENGTH,
        output_dir: str = None,
        user_id: str = None,
        custom_instructions: str = None,
        host_a_persona: str = None,
        host_b_persona: str = None
    ) -> Tuple[str, str]:
        """
        Run pipeline up to quality assurance only (no phoneme enhancement or audio).
        
        Args:
            pdf_path: Path to source PDF file
            target_word_length: Target word count for script
            output_dir: Output directory (created if None)
            user_id: Optional user ID to prefix output directory name
            
        Returns:
            Tuple of (output_directory, human_readable_script_path)
        """
        pipeline_id = str(uuid.uuid4())
        pdf_filename = os.path.basename(pdf_path)
        self.logger = LoggingManager(pipeline_id, pdf_filename)
        
        if output_dir is None:
            output_dir = self._create_output_directory(pdf_path, user_id=user_id)
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        self.prompt_logger = PromptLogger(output_dir)
        
        try:
            
            print(f"Starting pipeline for {pdf_filename}")
            print(f"Output directory: {output_dir}")
            
            print("Initializing document cache...")
            cache_name = self.cache_manager.get_or_create_cache(pdf_path, self.client)
            self.logger.set_cache_name(cache_name)
            
            
            script_generator = ScriptGenerator(self.client, self.logger, self.prompt_logger)
            quality_assurance = QualityAssurance(self.client, self.logger, self.prompt_logger)
            
            print("Generating podcast script...")
            initial_script = script_generator.generate_script_with_length_correction(
                cache_name, target_word_length, 
                additional_instructions=None,
                host_a_persona=host_a_persona,
                host_b_persona=host_b_persona,
                custom_instructions=custom_instructions
            )
            print(f"✓ Script generated: {initial_script.word_count} words (target: {target_word_length})")
            
            if initial_script.word_count != target_word_length:
                target_min = int(target_word_length * (1 - SCRIPT_LENGTH_TOLERANCE))
                target_max = int(target_word_length * (1 + SCRIPT_LENGTH_TOLERANCE))
                tolerance_percent = int(SCRIPT_LENGTH_TOLERANCE * 100)
                if target_min <= initial_script.word_count <= target_max:
                    print(f"✓ Script length is within acceptable range (±{tolerance_percent}%)")
                else:
                    print("⚠ Script length correction may have been applied")
            
            self._save_script(
                initial_script.script_text, output_dir, "initial"
            )
            
            print("Running quality assurance...")
            final_script_text = quality_assurance.run_full_qa_cycle(
                initial_script.script_text, cache_name, target_word_length
            )
            print("✓ Quality assurance completed")
            
            human_script_path = self._save_script(final_script_text, output_dir, "human_readable")
            
            log_path = self.logger.save_pipeline_log(output_dir)
            
            self.logger.print_summary()
            
            print(f"\nScript generation completed successfully!")
            print(f"Human-readable script: {human_script_path}")
            print(f"Pipeline log: {log_path}")
            if self.prompt_logger:
                print(f"LLM interactions: {self.prompt_logger.get_logs_summary()}")
            
            return output_dir, human_script_path
            
        except Exception as e:
            error_msg = str(e)
            print(f"Pipeline failed: {error_msg}")
            
            if self.logger:
                self.logger.save_pipeline_log(output_dir or "output/failed")
            
            raise
    
    def run_script_only(
        self,
        pdf_path: str,
        target_word_length: int = DEFAULT_TARGET_WORD_LENGTH,
        output_dir: str = None,
        user_id: str = None
    ) -> Tuple[str, str]:
        """
        Run pipeline up to script generation only (no audio).
        
        Args:
            pdf_path: Path to source PDF file
            target_word_length: Target word count for script
            output_dir: Output directory (created if None)
            user_id: Optional user ID to prefix output directory name
            
        Returns:
            Tuple of (output_directory, final_script_path)
        """
        output_dir, script_path, _, _ = self.run_pipeline(
            pdf_path, target_word_length, output_dir, skip_audio=True, user_id=user_id
        )
        return output_dir, script_path
    
    def generate_audio_from_script(
        self,
        script_path: str,
        output_dir: str = None
    ) -> Tuple[str, str]:
        """
        Generate audio from an existing script file.
        
        Args:
            script_path: Path to script file
            output_dir: Output directory (uses script directory if None)
            
        Returns:
            Tuple of (wav_path, mp3_path)
        """
        with open(script_path, 'r', encoding='utf-8') as f:
            script_text = f.read()
        
        script_dir = os.path.dirname(script_path)
        human_readable_script = None
        
        for filename in os.listdir(script_dir):
            if "human_readable" in filename and filename.endswith('.txt'):
                human_readable_path = os.path.join(script_dir, filename)
                try:
                    with open(human_readable_path, 'r', encoding='utf-8') as f:
                        human_readable_script = f.read()
                    print(f"  → Found human-readable script: {filename}")
                    break
                except Exception as e:
                    print(f"  → Warning: Could not read {filename}: {e}")
        
        if not human_readable_script:
            print("  → No human-readable script found, using phoneme tag removal for quality checking")
        
        if output_dir is None:
            output_dir = os.path.dirname(script_path)
        
        logger = LoggingManager(str(uuid.uuid4()), os.path.basename(script_path))
        prompt_logger = PromptLogger(output_dir)
        tts_generator = TTSGenerator(self.client, logger, prompt_logger)
        
        wav_path, mp3_path = tts_generator.generate_audio_from_script(
            script_text, output_dir, "podcast_from_script", human_readable_script
        )
        
        return wav_path, mp3_path, logger
    
    def _consolidate_pipeline_log(self, output_dir: str, new_logger: LoggingManager, operation_name: str) -> None:
        """
        Consolidate metrics from a new LoggingManager into the existing pipeline log.
        
        Args:
            output_dir: Directory containing the pipeline log
            new_logger: LoggingManager containing new metrics to merge
            operation_name: Description of the operation for logging
        """
        try:
            existing_log_path = os.path.join(output_dir, "pipeline_log.json")
            if os.path.exists(existing_log_path):
                import json
                with open(existing_log_path, 'r') as f:
                    existing_log = json.load(f)
                
                source_pdf = existing_log.get('source_pdf', 'unknown')
                pipeline_id = existing_log.get('pipeline_id', str(uuid.uuid4()))
                consolidated_logger = LoggingManager(pipeline_id=pipeline_id, source_pdf=source_pdf)
                consolidated_logger.stage_metrics = []
                
                if 'stage_metrics' in existing_log:
                    from .models.schemas import StageMetrics, TokenUsage
                    for stage_data in existing_log['stage_metrics']:
                        token_usage = TokenUsage(
                            prompt_tokens=stage_data['token_usage']['prompt_tokens'],
                            cached_tokens=stage_data['token_usage']['cached_tokens'],
                            candidate_tokens=stage_data['token_usage']['candidate_tokens'],
                            total_tokens=stage_data['token_usage']['total_tokens']
                        )
                        stage_metric = StageMetrics(
                            stage_name=stage_data['stage_name'],
                            execution_time_seconds=stage_data['execution_time_seconds'],
                            token_usage=token_usage,
                            cost_usd=stage_data['cost_usd'],
                            model_used=stage_data['model_used'],
                            cache_hit=stage_data['cache_hit']
                        )
                        consolidated_logger.stage_metrics.append(stage_metric)
                
                consolidated_logger.merge_metrics_from_logger(new_logger)
                
                log_path = consolidated_logger.save_pipeline_log(output_dir)
                print(f"✅ {operation_name} metrics merged into pipeline log: {log_path}")
            else:
                log_path = new_logger.save_pipeline_log(output_dir)
                print(f"✅ {operation_name} metrics saved to: {log_path}")
        except Exception as e:
            print(f"⚠ Could not save {operation_name} metrics: {e}")
    
    def validate_pipeline_setup(self) -> Dict[str, bool]:
        """
        Validate that the pipeline is properly set up.
        
        Returns:
            Dictionary with validation results
        """
        results = {}
        
        try:
            test_response = self.client.generate_content("Test")
            results["api_key"] = bool(test_response and test_response.text)
        except Exception as e:
            print(f"API key validation failed: {e}")
            results["api_key"] = False
        
        results["output_dir"] = True  # Can be created as needed
        
        try:
            cache_log = self.cache_manager.list_active_caches()
            results["cache_manager"] = True
        except Exception:
            results["cache_manager"] = False
        
        try:
            enhancer = PhonemeEnhancer(self.client, LoggingManager(source_pdf="validation_check"))
            phoneme_dict = enhancer.get_phoneme_dictionary()
            results["phoneme_dict"] = len(phoneme_dict) > 0
        except Exception:
            results["phoneme_dict"] = False

        return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <pdf_path> [target_word_length] [output_dir]")
        print("Note: Set GEMINI_API_KEY environment variable or use PodcastPipeline(api_key='your_key') in Python")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    target_length = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TARGET_WORD_LENGTH
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None
    
    try:
        pipeline = PodcastPipeline()  # Will use GEMINI_API_KEY from environment
        
        output_dir, script_path, wav_path, mp3_path = pipeline.run_pipeline(
            pdf_path, target_length, output_dir
        )
        print(f"\nSuccess! Check output in: {output_dir}")
    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)