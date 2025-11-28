"""
Text-to-Speech generation module using Gemini TTS API.
Handles script splitting, multi-speaker audio generation, and audio stitching.
"""

import os
import base64
import subprocess
import wave
import uuid
import time
import concurrent.futures
import asyncio
from typing import List, Tuple, Dict
import re

from google import genai
from google.genai import types

from ..config import AUDIO_CHUNK_SIZE, PAUSE_DURATION, PAUSE_DURATION_FINAL, TTS_VOICE_CONFIG, GEMINI_TTS_MODEL, THINKING_BUDGET, ENABLE_TTS_QUALITY_CHECK, TTS_MAX_RETRIES
from ..models.schemas import AudioChunk, AudioGeneration, TokenUsage, TTSQualityAssessment
from ..utils.logging_utils import LoggingManager
from .tts_checker import TTSChecker


class TTSGenerator:
    """Handles text-to-speech generation for podcast scripts."""
    
    def __init__(self, client, logger: LoggingManager, prompt_logger=None, enable_quality_check=None, phoneme_dict=None):
        self.client = client
        self.logger = logger
        self.prompt_logger = prompt_logger
        self.enable_quality_check = enable_quality_check if enable_quality_check is not None else ENABLE_TTS_QUALITY_CHECK
        self.tts_checker = TTSChecker(client, logger, phoneme_dict=phoneme_dict) if self.enable_quality_check else None
    
    def _count_tokens_estimate(self, text: str) -> int:
        """Rough estimate of token count for text."""
        return len(text) // 4
    
    def _split_script_into_chunks(self, script: str, max_tokens: int = AUDIO_CHUNK_SIZE) -> List[AudioChunk]:
        """
        Split script into chunks with even distribution, respecting speaker boundaries.
        
        Args:
            script: The complete script text
            max_tokens: Target tokens per chunk (default 1000)
            
        Returns:
            List of AudioChunk objects
        """
        lines = script.strip().split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        
        total_tokens = sum(self._count_tokens_estimate(line) for line in lines)
        ideal_chunks = max(1, (total_tokens + max_tokens - 1) // max_tokens)  # Round up
        target_tokens_per_chunk = total_tokens / ideal_chunks
        
        chunks = []
        current_chunk_lines = []
        current_chunk_tokens = 0
        chunk_id = 0
        
        for line in lines:
            line_tokens = self._count_tokens_estimate(line)
            
            should_break = (
                current_chunk_lines and  # Has content
                chunk_id < ideal_chunks - 1 and  # Not the last chunk
                current_chunk_tokens + line_tokens > target_tokens_per_chunk and  # Would exceed target
                (line.startswith('Host A:') or line.startswith('Host B:'))  # Speaker boundary
            )
            
            if should_break:
                chunk_text = '\n'.join(current_chunk_lines)
                speaker_starts = self._extract_speaker_starts(chunk_text)
                
                chunks.append(AudioChunk(
                    chunk_id=chunk_id,
                    text_content=chunk_text,
                    speaker_starts=speaker_starts,
                    word_count=len(chunk_text.split())
                ))
                
                current_chunk_lines = []
                current_chunk_tokens = 0
                chunk_id += 1
            
            current_chunk_lines.append(line)
            current_chunk_tokens += line_tokens
        
        if current_chunk_lines:
            chunk_text = '\n'.join(current_chunk_lines)
            speaker_starts = self._extract_speaker_starts(chunk_text)
            
            chunks.append(AudioChunk(
                chunk_id=chunk_id,
                text_content=chunk_text,
                speaker_starts=speaker_starts,
                word_count=len(chunk_text.split())
            ))
        
        chunks = self._ensure_speaker_starts(chunks)
        
        return chunks
    
    def _extract_speaker_starts(self, text: str) -> List[str]:
        """Extract speakers that start speaking in this chunk."""
        speakers = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('Host A:'):
                if 'Host A' not in speakers:
                    speakers.append('Host A')
            elif line.startswith('Host B:'):
                if 'Host B' not in speakers:
                    speakers.append('Host B')
        
        return speakers
    
    def _ensure_speaker_starts(self, chunks: List[AudioChunk]) -> List[AudioChunk]:
        """
        Ensure each chunk begins with a speaker line for proper context.
        
        Args:
            chunks: List of audio chunks
            
        Returns:
            Modified chunks with proper speaker starts
        """
        if not chunks:
            return chunks
        
        for i in range(1, len(chunks)):
            chunk = chunks[i]
            lines = chunk.text_content.split('\n')
            
            first_line = lines[0].strip()
            if not (first_line.startswith('Host A:') or first_line.startswith('Host B:')):
                prev_chunk = chunks[i - 1]
                prev_lines = prev_chunk.text_content.split('\n')
                
                last_speaker = None
                for line in reversed(prev_lines):
                    line = line.strip()
                    if line.startswith('Host A:') or line.startswith('Host B:'):
                        last_speaker = line.split(':')[0]
                        break
                
                if last_speaker:
                    continuation = f"{last_speaker}: [continuing]"
                    chunk.text_content = f"{continuation}\n{chunk.text_content}"
                    chunk.speaker_starts = self._extract_speaker_starts(chunk.text_content)
        
        return chunks
    
    def _create_tts_config(self) -> types.GenerateContentConfig:
        """Create TTS configuration for multi-speaker audio."""
        return types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=[
                        types.SpeakerVoiceConfig(
                            speaker="Host A",
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=TTS_VOICE_CONFIG['speaker_a']
                                )
                            )
                        ),
                        types.SpeakerVoiceConfig(
                            speaker="Host B",
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=TTS_VOICE_CONFIG['speaker_b']
                                )
                            )
                        )
                    ]
                )
            )
        )
    
    def _generate_audio_chunk(self, chunk: AudioChunk, is_final_chunk: bool = False, debug_dir: str = None) -> Tuple[int, bytes, float]:
        """
        Generate audio for a single chunk.
        
        Args:
            chunk: AudioChunk to generate audio for
            is_final_chunk: Whether this is the final chunk of the script
            debug_dir: Directory to save debug files (optional)
            
        Returns:
            Tuple of (chunk_id, audio_data, execution_time)
        """
        start_time = time.time()
        
        prompt_header = "Please read the following aloud in an authentic podcast style—lively and engaging. Read slightly quickly, like in a true human conversation. When phoneme tags are provided, use the specified IPA pronunciation. You *MUST* read *exactly* as it is written.\n\n--\n"
        pause_duration = PAUSE_DURATION_FINAL if is_final_chunk else PAUSE_DURATION
        full_prompt = prompt_header + chunk.text_content + pause_duration
        
        if self.prompt_logger and debug_dir:
            chunk_text_file = os.path.join(debug_dir, f"tts_chunk_{chunk.chunk_id + 1}.txt")
            with open(chunk_text_file, 'w', encoding='utf-8') as f:
                f.write(f"Chunk {chunk.chunk_id + 1} Content:\n")
                f.write(f"Word count: {chunk.word_count}\n")
                f.write(f"Speaker starts: {', '.join(chunk.speaker_starts)}\n")
                f.write(f"Is final chunk: {is_final_chunk}\n\n")
                f.write("=== CHUNK TEXT ===\n")
                f.write(chunk.text_content)
                f.write("\n\n=== FULL PROMPT ===\n")
                f.write(full_prompt)
        
        tts_config = self._create_tts_config()
        
        try:
            response = self.client.models.generate_content(
                model=GEMINI_TTS_MODEL,
                contents=[full_prompt],
                config=tts_config
            )
            
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        audio_data = part.inline_data.data
                        
                        if isinstance(audio_data, str):
                            audio_data = base64.b64decode(audio_data)
                        
                        execution_time = time.time() - start_time
                        
                        if debug_dir:
                            chunk_wav_file = os.path.join(debug_dir, f"tts_chunk_{chunk.chunk_id + 1}.wav")
                            self._save_wave_file(chunk_wav_file, audio_data)
                        
                        if hasattr(response, 'usage_metadata'):
                            usage = response.usage_metadata
                            token_usage = TokenUsage(
                                prompt_tokens=usage.prompt_token_count or 0,
                                cached_tokens=0,
                                candidate_tokens=usage.candidates_token_count or 0,
                                total_tokens=usage.total_token_count or 0
                            )
                            self.logger.log_stage_metrics(
                                f"tts_chunk_{chunk.chunk_id}", 
                                token_usage, 
                                GEMINI_TTS_MODEL,
                                execution_time=execution_time,
                                is_audio=True
                            )
                        
                        return chunk.chunk_id, audio_data, execution_time
            
            error_msg = f"No audio data found in response for chunk {chunk.chunk_id + 1}"
            if debug_dir:
                error_file = os.path.join(debug_dir, f"tts_chunk_{chunk.chunk_id + 1}_error.txt")
                with open(error_file, 'w', encoding='utf-8') as f:
                    f.write(f"Error: {error_msg}\n")
                    f.write(f"Response: {response}\n")
            raise ValueError(error_msg)
            
        except Exception as e:
            if debug_dir:
                error_file = os.path.join(debug_dir, f"tts_chunk_{chunk.chunk_id + 1}_error.txt")
                with open(error_file, 'w', encoding='utf-8') as f:
                    f.write(f"Exception during TTS generation: {str(e)}\n")
                    f.write(f"Chunk ID: {chunk.chunk_id + 1}\n")
                    f.write(f"Word count: {chunk.word_count}\n")
                    f.write(f"Text content:\n{chunk.text_content}\n")
            raise ValueError(f"Failed to generate audio for chunk {chunk.chunk_id + 1}: {str(e)}")
    
    def _generate_and_validate_chunk(self, chunk: AudioChunk, is_final_chunk: bool = False, debug_dir: str = None, max_retries: int = None) -> Tuple[int, bytes, float]:
        """
        Generate audio chunk with quality validation and retry logic.
        
        Args:
            chunk: AudioChunk to generate audio for
            is_final_chunk: Whether this is the final chunk of the script
            debug_dir: Directory to save debug files (optional)
            max_retries: Maximum number of retry attempts for critical issues
            
        Returns:
            Tuple of (chunk_id, audio_data, execution_time)
        """
        total_start_time = time.time()
        max_retries = max_retries if max_retries is not None else TTS_MAX_RETRIES
        
        for attempt in range(max_retries):
            try:
                chunk_id, audio_data, generation_time = self._generate_audio_chunk(chunk, is_final_chunk, debug_dir)
                
                if not self.enable_quality_check or not self.tts_checker:
                    return chunk_id, audio_data, generation_time
                
                if debug_dir:
                    temp_audio_path = os.path.join(debug_dir, f"tts_chunk_{chunk.chunk_id + 1}.wav")
                    
                    assessment = self.tts_checker.check_audio_chunk(
                        chunk_id=chunk.chunk_id,
                        audio_file_path=temp_audio_path,
                        expected_text=chunk.text_content
                    )
                    
                    if assessment.has_critical_issues and attempt < max_retries - 1:
                        print(f"  → Chunk {chunk_id + 1}: Retrying due to quality issues (attempt {attempt + 2})")
                        self.logger.log_warning(f"TTS Chunk {chunk_id + 1} attempt {attempt + 1}: Quality issues detected, retrying...")
                        self.logger.log_warning(f"  Retrying - Issues found: {len([i for i in assessment.issues if i.severity == 'critical'])} critical, {len([i for i in assessment.issues if i.severity == 'minor'])} minor")
                        continue
                    elif assessment.has_critical_issues and attempt == max_retries - 1:
                        print(f"  ⚠️  Chunk {chunk_id + 1}: Max retries reached, using best attempt")
                        self.logger.log_error(f"TTS Chunk {chunk_id + 1}: Max retries reached, proceeding with potentially flawed audio")
                        self.logger.log_error(f"  Critical issues remain: {len([i for i in assessment.issues if i.severity == 'critical'])}")
                    else:
                        print(f"  ✓ Chunk {chunk_id + 1}: Quality acceptable (score: {assessment.overall_quality_score:.2f})")
                        self.logger.log_info(f"TTS Chunk {chunk_id + 1}: Quality acceptable (score: {assessment.overall_quality_score:.2f})")
                
                total_time = time.time() - total_start_time
                return chunk_id, audio_data, total_time
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise ValueError(f"Failed to generate valid audio for chunk {chunk.chunk_id + 1} after {max_retries} attempts: {str(e)}")
                
                self.logger.log_warning(f"TTS Chunk {chunk.chunk_id + 1} attempt {attempt + 1} failed: {str(e)}, retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff
        
        raise ValueError(f"Unexpected error in chunk generation for {chunk.chunk_id + 1}")
    
    def _generate_audio_chunk_with_attempt(self, chunk: AudioChunk, is_final_chunk: bool = False, debug_dir: str = None, attempt_number: int = 1) -> Tuple[int, bytes, float]:
        """
        Generate audio chunk with attempt-specific debug file naming.
        
        Args:
            chunk: AudioChunk to generate audio for
            is_final_chunk: Whether this is the final chunk of the script
            debug_dir: Directory to save debug files (optional)
            attempt_number: Attempt number for unique file naming
            
        Returns:
            Tuple of (chunk_id, audio_data, execution_time)
        """
        start_time = time.time()
        
        prompt_header = "Please read the following aloud in an authentic podcast style—lively and engaging. Read slightly quickly, like in a true human conversation. When phoneme tags are provided, use the specified IPA pronunciation. You *MUST* read *exactly* as it is written.\n\n--\n"
        pause_duration = PAUSE_DURATION_FINAL if is_final_chunk else PAUSE_DURATION
        full_prompt = prompt_header + chunk.text_content + pause_duration
        
        if self.prompt_logger and debug_dir:
            chunk_text_file = os.path.join(debug_dir, f"tts_chunk_{chunk.chunk_id + 1}_attempt_{attempt_number}.txt")
            with open(chunk_text_file, 'w', encoding='utf-8') as f:
                f.write(f"Chunk {chunk.chunk_id + 1} Content - Attempt {attempt_number}:\n")
                f.write(f"Word count: {chunk.word_count}\n")
                f.write(f"Speaker starts: {', '.join(chunk.speaker_starts)}\n")
                f.write(f"Is final chunk: {is_final_chunk}\n")
                f.write(f"Attempt number: {attempt_number}\n\n")
                f.write("=== CHUNK TEXT ===\n")
                f.write(chunk.text_content)
                f.write("\n\n=== FULL PROMPT ===\n")
                f.write(full_prompt)
        
        tts_config = self._create_tts_config()
        
        try:
            response = self.client.models.generate_content(
                model=GEMINI_TTS_MODEL,
                contents=[full_prompt],
                config=tts_config
            )
            
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        audio_data = part.inline_data.data
                        
                        if isinstance(audio_data, str):
                            audio_data = base64.b64decode(audio_data)
                        
                        execution_time = time.time() - start_time
                        
                        if debug_dir:
                            chunk_wav_file = os.path.join(debug_dir, f"tts_chunk_{chunk.chunk_id + 1}_attempt_{attempt_number}.wav")
                            self._save_wave_file(chunk_wav_file, audio_data)
                        
                        if hasattr(response, 'usage_metadata'):
                            usage = response.usage_metadata
                            token_usage = TokenUsage(
                                prompt_tokens=usage.prompt_token_count or 0,
                                cached_tokens=0,
                                candidate_tokens=usage.candidates_token_count or 0,
                                total_tokens=usage.total_token_count or 0
                            )
                            self.logger.log_stage_metrics(
                                f"tts_chunk_{chunk.chunk_id}_attempt_{attempt_number}", 
                                token_usage, 
                                GEMINI_TTS_MODEL,
                                execution_time=execution_time,
                                is_audio=True
                            )
                        
                        return chunk.chunk_id, audio_data, execution_time
            
            error_msg = f"No audio data found in response for chunk {chunk.chunk_id + 1} attempt {attempt_number}"
            if debug_dir:
                error_file = os.path.join(debug_dir, f"tts_chunk_{chunk.chunk_id + 1}_attempt_{attempt_number}_error.txt")
                with open(error_file, 'w', encoding='utf-8') as f:
                    f.write(f"Error: {error_msg}\n")
                    f.write(f"Response: {response}\n")
            raise ValueError(error_msg)
            
        except Exception as e:
            if debug_dir:
                error_file = os.path.join(debug_dir, f"tts_chunk_{chunk.chunk_id + 1}_attempt_{attempt_number}_error.txt")
                with open(error_file, 'w', encoding='utf-8') as f:
                    f.write(f"Exception during TTS generation: {str(e)}\n")
                    f.write(f"Chunk ID: {chunk.chunk_id + 1}\n")
                    f.write(f"Attempt number: {attempt_number}\n")
                    f.write(f"Word count: {chunk.word_count}\n")
                    f.write(f"Text content:\n{chunk.text_content}\n")
            raise ValueError(f"Failed to generate audio for chunk {chunk.chunk_id + 1} attempt {attempt_number}: {str(e)}")
    
    async def _generate_and_validate_chunk_async(self, chunk: AudioChunk, is_final_chunk: bool = False, debug_dir: str = None, human_readable_script: str = None, max_retries: int = None) -> Tuple[int, bytes, float]:
        """
        Async version: Generate audio chunk with immediate quality validation pipeline.
        
        Args:
            chunk: AudioChunk to generate audio for
            is_final_chunk: Whether this is the final chunk of the script
            debug_dir: Directory to save debug files (optional)
            max_retries: Maximum number of retry attempts for critical issues
            
        Returns:
            Tuple of (chunk_id, audio_data, execution_time)
        """
        total_start_time = time.time()
        max_retries = max_retries if max_retries is not None else TTS_MAX_RETRIES
        
        attempts = []  # List of (audio_data, assessment, generation_time, attempt_number)
        
        quality_retries = 0  # Retries due to quality issues (count against max_retries)
        server_retries = 0   # Retries due to server errors (don't count against max_retries)
        total_attempts = 0   # Total attempts made
        
        def _is_server_error(error_msg: str) -> bool:
            """Check if error is a server-side error that shouldn't count against max_retries."""
            server_error_indicators = [
                "500 INTERNAL",
                "503 SERVICE_UNAVAILABLE", 
                "502 BAD_GATEWAY",
                "504 GATEWAY_TIMEOUT",
                "429 RESOURCE_EXHAUSTED",
                "DEADLINE_EXCEEDED",
                "UNAVAILABLE",
                "INTERNAL"
            ]
            return any(indicator in error_msg for indicator in server_error_indicators)
        
        while quality_retries < max_retries:
            total_attempts += 1
            try:
                loop = asyncio.get_event_loop()
                attempt_debug_dir = debug_dir
                attempt_suffix = f"_attempt_{total_attempts}" if debug_dir else None
                
                chunk_id, audio_data, generation_time = await loop.run_in_executor(
                    None, self._generate_audio_chunk_with_attempt, chunk, is_final_chunk, debug_dir, total_attempts
                )
                
                if not self.enable_quality_check or not self.tts_checker:
                    return chunk_id, audio_data, generation_time
                
                assessment = None
                if debug_dir:
                    temp_audio_path = os.path.join(debug_dir, f"tts_chunk_{chunk.chunk_id + 1}_attempt_{total_attempts}.wav")
                    
                    if hasattr(chunk, 'human_readable_text') and chunk.human_readable_text:
                        human_readable_text = chunk.human_readable_text
                    else:
                        human_readable_text = re.sub(r'<[^>]*>', '', chunk.text_content)
                        human_readable_text = re.sub(r'\s+', ' ', human_readable_text).strip()
                    
                    assessment = await loop.run_in_executor(
                        None, 
                        self.tts_checker.check_audio_chunk,
                        chunk.chunk_id,
                        temp_audio_path,
                        human_readable_text
                    )
                    
                    if self.prompt_logger:
                        assessment_file = os.path.join(debug_dir, f"tts_chunk_{chunk.chunk_id + 1}_attempt_{total_attempts}_quality_assessment.txt")
                        with open(assessment_file, 'w', encoding='utf-8') as f:
                            f.write(f"TTS Quality Assessment - Chunk {chunk.chunk_id + 1} - Attempt {total_attempts}\n")
                            f.write(f"{'=' * 60}\n\n")
                            f.write(f"Overall Quality Score: {assessment.overall_quality_score:.2f}\n")
                            f.write(f"Has Critical Issues: {assessment.has_critical_issues}\n")
                            critical_count = len([i for i in assessment.issues if i.severity == "critical"])
                            minor_count = len([i for i in assessment.issues if i.severity == "minor"])
                            f.write(f"Critical Issues: {critical_count}, Minor Issues: {minor_count}\n\n")
                            
                            if assessment.issues:
                                f.write("Issues Found:\n")
                                for issue in assessment.issues:
                                    f.write(f"- {issue.severity.upper()}: {issue.issue_type}\n")
                                    f.write(f"  Description: {issue.description}\n\n")
                            else:
                                f.write("No issues found.\n")
                
                attempts.append((audio_data, assessment, generation_time, total_attempts))
                
                if not assessment or not assessment.has_critical_issues:
                    if assessment:
                        self.logger.log_info(f"TTS Chunk {chunk_id + 1}: Quality acceptable (score: {assessment.overall_quality_score:.2f})")
                    total_time = time.time() - total_start_time
                    return chunk_id, audio_data, total_time
                
                else:
                    quality_retries += 1
                    if quality_retries < max_retries:
                        self.logger.log_warning(f"TTS Chunk {chunk_id + 1} attempt {total_attempts}: Quality issues detected, retrying... (quality retry {quality_retries}/{max_retries})")
                        self.logger.log_warning(f"  Retrying - Issues found: {len([i for i in assessment.issues if i.severity == 'critical'])} critical, {len([i for i in assessment.issues if i.severity == 'minor'])} minor")
                        continue
                    else:
                        break
                    
            except Exception as e:
                error_str = str(e)
                
                error_assessment = None
                if self.tts_checker:
                    error_assessment = self.tts_checker._create_error_assessment(chunk.chunk_id, error_str)
                attempts.append((None, error_assessment, 0, total_attempts))
                
                if _is_server_error(error_str):
                    server_retries += 1
                    self.logger.log_warning(f"TTS Chunk {chunk.chunk_id + 1} attempt {total_attempts} failed due to server error: {error_str}")
                    self.logger.log_warning(f"  Server retry {server_retries} (not counted against quality limit), retrying...")
                    await asyncio.sleep(min(2 ** server_retries, 30))  # Cap at 30 seconds
                    continue
                else:
                    quality_retries += 1
                    if quality_retries < max_retries:
                        self.logger.log_warning(f"TTS Chunk {chunk.chunk_id + 1} attempt {total_attempts} failed: {error_str}, retrying... (quality retry {quality_retries}/{max_retries})")
                        await asyncio.sleep(2 ** quality_retries)
                        continue
                    else:
                        break
        
        if not attempts:
            raise ValueError(f"No valid attempts generated for chunk {chunk.chunk_id + 1}")
        
        valid_attempts = [(audio, assess, gen_time, num) for audio, assess, gen_time, num in attempts if audio is not None]
        
        if not valid_attempts:
            raise ValueError(f"All attempts failed for chunk {chunk.chunk_id + 1}")
        
        best_attempt = max(valid_attempts, key=lambda x: x[1].overall_quality_score if x[1] else 0.0)
        best_audio, best_assessment, best_time, best_attempt_num = best_attempt
        
        self.logger.log_error(f"TTS Chunk {chunk_id + 1}: Max quality retries reached ({quality_retries}/{max_retries}), using best attempt #{best_attempt_num}")
        self.logger.log_error(f"  Total attempts made: {total_attempts} (including {server_retries} server error retries)")
        if best_assessment:
            self.logger.log_error(f"  Best attempt quality score: {best_assessment.overall_quality_score:.2f}")
            critical_count = len([i for i in best_assessment.issues if i.severity == "critical"])
            self.logger.log_error(f"  Best attempt issues: {critical_count} critical, {len(best_assessment.issues) - critical_count} minor")
        
        if debug_dir and self.prompt_logger:
            selection_file = os.path.join(debug_dir, f"tts_chunk_{chunk.chunk_id + 1}_best_attempt_selection.txt")
            with open(selection_file, 'w', encoding='utf-8') as f:
                f.write(f"Best Attempt Selection - Chunk {chunk.chunk_id + 1}\n")
                f.write(f"{'=' * 50}\n\n")
                f.write(f"Total attempts: {total_attempts}\n")
                f.write(f"Quality retries: {quality_retries}/{max_retries}\n")
                f.write(f"Server error retries: {server_retries} (not counted against limit)\n")
                f.write(f"Valid attempts: {len(valid_attempts)}\n")
                f.write(f"Selected attempt: #{best_attempt_num}\n")
                f.write(f"Selected quality score: {best_assessment.overall_quality_score:.2f}\n\n")
                
                f.write("All attempts:\n")
                for audio, assess, gen_time, num in attempts:
                    status = "FAILED" if audio is None else "SUCCESS"
                    score = assess.overall_quality_score if assess else 0.0
                    f.write(f"  Attempt #{num}: {status} (score: {score:.2f})\n")
                    if assess and assess.issues:
                        critical_count = len([i for i in assess.issues if i.severity == "critical"])
                        f.write(f"    Issues: {critical_count} critical, {len(assess.issues) - critical_count} minor\n")
        
        total_time = time.time() - total_start_time
        return chunk_id, best_audio, total_time
    
    def _stitch_audio_chunks(self, audio_chunks: List[bytes]) -> bytes:
        """
        Stitch multiple audio chunks into a single audio stream.
        
        Args:
            audio_chunks: List of raw audio data chunks
            
        Returns:
            Combined audio data
        """
        combined_audio = bytearray()
        
        for chunk in audio_chunks:
            combined_audio.extend(chunk)
        
        return bytes(combined_audio)
    
    def _save_wave_file(self, filename: str, pcm_data: bytes, channels: int = 1, rate: int = 24000, sample_width: int = 2) -> None:
        """Save PCM data to WAV file with proper headers"""
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)  # 24kHz to match Gemini TTS output
            wf.writeframes(pcm_data)
    
    def _save_audio_file(self, audio_data: bytes, output_path: str, format: str = "wav") -> str:
        """
        Save audio data to file.
        
        Args:
            audio_data: Raw audio data
            output_path: Output file path
            format: Audio format (wav or mp3)
            
        Returns:
            Path to saved file
        """
        if format.lower() == "wav":
            self._save_wave_file(output_path, audio_data)
            return output_path
        
        elif format.lower() == "mp3":
            temp_wav_path = output_path.replace('.mp3', f'_temp_{uuid.uuid4().hex[:8]}.wav')
            self._save_wave_file(temp_wav_path, audio_data)
            
            mp3_result = self._convert_wav_to_mp3(temp_wav_path, output_path)
            if mp3_result:
                os.remove(temp_wav_path)
                return mp3_result
            else:
                print(f"  → Keeping WAV file: {temp_wav_path}")
                return temp_wav_path
        
        else:
            raise ValueError(f"Unsupported audio format: {format}")
    
    def _convert_wav_to_mp3(self, wav_filename: str, mp3_filename: str = None) -> str:
        """Convert WAV file to MP3 using ffmpeg"""
        if mp3_filename is None:
            mp3_filename = wav_filename.replace('.wav', '.mp3')
        
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            
            print(f"  → Converting {os.path.basename(wav_filename)} to MP3...")
            cmd = [
                'ffmpeg', '-i', wav_filename,
                '-codec:a', 'libmp3lame',
                '-b:a', '320k',  # 320 kbps bitrate for maximum quality
                '-q:a', '0',     # Use best quality VBR mode
                '-y',  # Overwrite output file if it exists
                mp3_filename
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  ✓ MP3 conversion successful: {os.path.basename(mp3_filename)}")
                return mp3_filename
            else:
                print(f"  ✗ Error during MP3 conversion: {result.stderr}")
                return None
                
        except FileNotFoundError:
            print("  ✗ Error: ffmpeg not found. Please install ffmpeg to enable MP3 conversion.")
            print("    Install instructions:")
            print("      - Ubuntu/Debian: sudo apt install ffmpeg")
            print("      - macOS: brew install ffmpeg")
            print("      - Windows: Download from https://ffmpeg.org/download.html")
            return None
        except subprocess.CalledProcessError:
            print("  ✗ Error: ffmpeg check failed")
            return None
    
    def generate_audio_from_script(
        self, 
        script: str, 
        output_directory: str,
        filename_base: str = "podcast_final",
        human_readable_script: str = None
    ) -> Tuple[str, str]:
        """
        Generate complete audio from script.
        
        Args:
            script: Final script with phoneme tags
            output_directory: Directory to save audio files
            filename_base: Base filename for output files
            human_readable_script: Optional human-readable version for quality checking
            
        Returns:
            Tuple of (wav_path, mp3_path)
        """
        os.makedirs(output_directory, exist_ok=True)
        
        debug_dir = None
        if self.prompt_logger:
            debug_dir = os.path.join(output_directory, "llm_logs")
            os.makedirs(debug_dir, exist_ok=True)
        
        chunks = self._split_script_into_chunks(script)
        
        if human_readable_script:
            human_lines = [line for line in human_readable_script.strip().split('\n') if line.strip()]
            line_index = 0
            
            for chunk in chunks:
                chunk_lines = [line for line in chunk.text_content.strip().split('\n') if line.strip()]
                chunk_line_count = len(chunk_lines)
                
                human_chunk_lines = human_lines[line_index:line_index + chunk_line_count]
                chunk.human_readable_text = '\n'.join(human_chunk_lines)
                
                line_index += chunk_line_count
        
        total_tokens = sum(self._count_tokens_estimate(chunk.text_content) for chunk in chunks)
        total_words = sum(chunk.word_count for chunk in chunks)
        
        print(f"  → Script split into {len(chunks)} chunks for equal distribution")
        print(f"  → Total estimated tokens: {total_tokens}")
        print(f"  → Total words: {total_words}")
        if debug_dir:
            print(f"  → TTS debugging files will be saved to: {debug_dir}")
        
        for i, chunk in enumerate(chunks):
            chunk_tokens = self._count_tokens_estimate(chunk.text_content)
            print(f"    Chunk {i+1}: {chunk_tokens} tokens (~{chunk_tokens/total_tokens*100:.1f}%)")
        
        print(f"  → Processing {len(chunks)} chunks with async streaming validation...")
        audio_results = {}
        failed_chunks = []
        total_tts_time = 0.0
        
        async def process_all_chunks():
            tasks = [
                self._generate_and_validate_chunk_async(
                    chunk, 
                    chunk.chunk_id == len(chunks) - 1, 
                    debug_dir,
                    human_readable_script
                )
                for chunk in chunks
            ]
            
            for completed_task in asyncio.as_completed(tasks):
                try:
                    chunk_id, audio_data, execution_time = await completed_task
                    audio_results[chunk_id] = audio_data
                    nonlocal total_tts_time
                    total_tts_time += execution_time
                    print(f"  ✓ Chunk {chunk_id + 1} completed and validated ({execution_time:.1f}s)")
                except Exception as e:
                    chunk_num = str(e).split("chunk ")[1].split(" ")[0] if "chunk " in str(e) else "unknown"
                    failed_chunks.append(chunk_num)
                    print(f"  ✗ Failed to generate/validate chunk {chunk_num}: {e}")
                    continue
        
        try:
            asyncio.run(process_all_chunks())
        except RuntimeError:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(process_all_chunks())
        
        if failed_chunks:
            print(f"  ⚠ {len(failed_chunks)} chunk(s) failed: {', '.join(map(str, failed_chunks))}")
            if debug_dir:
                print(f"    Check error files in {debug_dir} for details")
        
        audio_chunks = []
        missing_chunks = []
        for i in range(len(chunks)):
            if i in audio_results:
                audio_chunks.append(audio_results[i])
            else:
                missing_chunks.append(i + 1)
        
        if not audio_chunks:
            raise ValueError("Failed to generate any audio chunks")
        
        if missing_chunks:
            print(f"  ⚠ Missing audio for chunks: {', '.join(map(str, missing_chunks))}")
        
        print(f"  → Stitching {len(audio_chunks)} audio chunks together...")
        print(f"  → Total TTS generation time: {total_tts_time:.1f}s (parallelized)")
        
        combined_audio = self._stitch_audio_chunks(audio_chunks)
        
        print(f"  → Saving WAV file...")
        wav_path = os.path.join(output_directory, f"{filename_base}.wav")
        wav_file = self._save_audio_file(combined_audio, wav_path, "wav")
        
        if filename_base.startswith("streamlit_"):
            print(f"  → Skipping MP3 conversion (WAV-only mode)")
            return wav_file, wav_file
        
        print(f"  → Converting to MP3...")
        mp3_path = os.path.join(output_directory, f"{filename_base}.mp3")
        mp3_file = self._save_audio_file(combined_audio, mp3_path, "mp3")
        
        return wav_file, mp3_file
    
    def create_audio_generation_plan(self, script: str) -> AudioGeneration:
        """
        Create an audio generation plan without actually generating audio.
        
        Args:
            script: Script to analyze
            
        Returns:
            AudioGeneration plan with chunk information
        """
        chunks = self._split_script_into_chunks(script)
        
        total_words = sum(chunk.word_count for chunk in chunks)
        estimated_duration = total_words / 150.0
        
        return AudioGeneration(
            chunks=chunks,
            total_chunks=len(chunks),
            estimated_duration_minutes=estimated_duration
        )
    
    def validate_script_for_tts(self, script: str) -> List[str]:
        """
        Validate script for TTS compatibility.
        
        Args:
            script: Script to validate
            
        Returns:
            List of validation issues
        """
        issues = []
        lines = script.split('\n')
        
        speaker_lines = 0
        for line in lines:
            line = line.strip()
            if line.startswith(('Host A:', 'Host B:')):
                speaker_lines += 1
            elif line and not line.startswith('<') and ':' in line:
                issues.append(f"Potential malformed speaker line: {line[:50]}...")
        
        if speaker_lines == 0:
            issues.append("No speaker lines found (Host A: or Host B:)")
        
        ssml_tags = ['phoneme', 'break', 'emphasis', 'prosody', 'say-as']
        for tag in ssml_tags:
            open_tags = len(re.findall(f'<{tag}[^>]*>', script, re.IGNORECASE))
            close_tags = len(re.findall(f'</{tag}>', script, re.IGNORECASE))
            
            if open_tags != close_tags:
                issues.append(f"Unmatched {tag} tags: {open_tags} open, {close_tags} close")
        
        word_count = len(script.split())
        if word_count < 100:
            issues.append(f"Script is very short ({word_count} words)")
        elif word_count > 10000:
            issues.append(f"Script is very long ({word_count} words) - may hit API limits")
        
        return issues