"""
TTS Quality Checker module using Gemini for audio validation.
Evaluates generated TTS audio against expected script content to identify critical issues.
"""

import os
import json
import re
import time
from typing import List, Optional, Tuple, Dict
from pathlib import Path

from google import genai
from google.genai import types

from ..config import GEMINI_TTS_MODEL, THINKING_BUDGET, TTS_MAX_RETRIES, DEFAULT_MODEL
from ..models.schemas import TTSQualityAssessment, TTSIssue
from ..utils.logging_utils import LoggingManager


class TTSChecker:
    """Validates TTS audio quality using Gemini multimodal analysis."""
    
    def __init__(self, client, logger: LoggingManager, model: str = DEFAULT_MODEL, phoneme_dict: Dict[str, str] = None):
        self.client = client
        self.logger = logger
        self.model = model
        self.phoneme_dict = phoneme_dict if phoneme_dict is not None else self._load_phoneme_dict()
    
    def check_audio_chunk(self, 
                         chunk_id: int,
                         audio_file_path: str,
                         expected_text: str,
                         max_retries: int = TTS_MAX_RETRIES) -> TTSQualityAssessment:
        """
        Check a single audio chunk for TTS quality issues.
        
        Args:
            chunk_id: ID of the audio chunk
            audio_file_path: Path to the generated .wav file
            expected_text: Script text that should be spoken
            max_retries: Maximum number of API retry attempts
            
        Returns:
            TTSQualityAssessment with evaluation results
        """
        if not os.path.exists(audio_file_path):
            self.logger.log_error(f"Audio file not found: {audio_file_path}")
            return self._create_error_assessment(chunk_id, "Audio file not found")
        
        try:
            audio_file = self.client.files.upload(file=audio_file_path)
            
            self._wait_for_file_processing(audio_file)
            
            prompt = self._create_evaluation_prompt(expected_text)
            
            if hasattr(self, 'logger') and self.logger:
                debug_file = audio_file_path.replace('.wav', '_gemini_prompt.txt')
                with open(debug_file, 'w', encoding='utf-8') as f:
                    f.write(f"TTS Quality Check - Chunk {chunk_id + 1}\n")
                    f.write(f"{'=' * 50}\n\n")
                    f.write("EXACT PROMPT SENT TO GEMINI:\n")
                    f.write(f"{prompt}\n\n")
                    f.write("AUDIO FILE:\n")
                    f.write(f"{audio_file_path}\n\n")
                    f.write("EXPECTED TEXT SENT TO TTS CHECKER:\n")
                    f.write(f"{expected_text}\n")
            
            for attempt in range(max_retries):
                try:
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=[prompt, audio_file],
                        config=types.GenerateContentConfig(
                            response_modalities=["TEXT"],
                            response_mime_type="application/json",
                            response_schema=TTSQualityAssessment,
                            thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET)
                        )
                    )
                    
                    if response and response.parsed:
                        assessment = response.parsed
                        
                        self._cleanup_uploaded_file(audio_file)
                        
                        self._log_assessment(chunk_id, assessment)
                        
                        return assessment
                    
                except Exception as e:
                    self.logger.log_error(f"TTS check attempt {attempt + 1} failed for chunk {chunk_id}: {str(e)}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(2 ** attempt)  # Exponential backoff
            
        except Exception as e:
            error_msg = str(e) if str(e) else f"Unknown error of type {type(e).__name__}"
            print(f"  ❌ TTS quality check completely failed for chunk {chunk_id + 1}: {error_msg}")
            self.logger.log_error(f"TTS quality check failed for chunk {chunk_id}: {error_msg}")
            return self._create_error_assessment(chunk_id, error_msg)
    
    def _load_phoneme_dict(self) -> Dict[str, str]:
        """Load the phoneme dictionary from phoneme_dict.json."""
        phoneme_file = "phoneme_dict.json"
        if os.path.exists(phoneme_file):
            try:
                with open(phoneme_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.log_warning(f"Could not load phoneme dictionary: {e}")
        return {}
    
    def _identify_phoneme_terms(self, text: str) -> List[Tuple[str, str]]:
        """Identify terms in text that should have special pronunciations."""
        phoneme_terms = []
        
        for term, ipa in self.phoneme_dict.items():
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, text):
                phoneme_terms.append((term, ipa))
        
        return phoneme_terms

    def _create_evaluation_prompt(self, expected_text: str) -> str:
        """Create the evaluation prompt for Gemini with phoneme guidance."""
        phoneme_terms = self._identify_phoneme_terms(expected_text)

        return f"""You are a text-to-speech quality assessor. Analyze this audio file and compare it to the expected script text.

EXPECTED SCRIPT TEXT:
{expected_text}

PRONUNCIATION RULES FOR SELECTED TERMS (these terms should be pronounced according to the IPA) **TERMS ARE CASE SENSITIVE**
*******DO NOT APPLY TO WORDS THAT HAVE DIFFERENT CASING *********:
{phoneme_terms}

CRITICAL ISSUES (require retry):
- Missing content: Entire sentences, phrases, or significant words are skipped
- Clipping at the end of the file CUTTING OFF A SPOKEN WORD OR PHRASE.
- Wrong speaker: Audio uses wrong voice for Host A/Host B sections
- Major mispronunciations: Technical terms, acronyms, or key concepts pronounced incorrectly
    Example: APHE, pronounced as "APHC"

MINOR ISSUES (log only, no retry): All other issues INCLUDING electronical sounds, buzzing, humming, noise, tone. Acceptable pronunciation variation of the IPA-tagged words. Minor variations of pronunciation.
**Abrupt ending is a *MINOR ISSUE* if no words are cut-off.**
"""

    def _wait_for_file_processing(self, uploaded_file, timeout: int = 60) -> None:
        """Wait for uploaded file to be processed by Gemini."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            file_info = self.client.files.get(name=uploaded_file.name)
            if file_info.state == "ACTIVE":
                return
            elif file_info.state == "FAILED":
                raise Exception(f"File processing failed: {uploaded_file.name}")
            
            time.sleep(2)
        
        raise Exception(f"File processing timeout: {uploaded_file.name}")
    
    def _cleanup_uploaded_file(self, uploaded_file) -> None:
        """Clean up uploaded file from Gemini storage."""
        try:
            self.client.files.delete(name=uploaded_file.name)
        except Exception as e:
            self.logger.log_warning(f"Failed to cleanup uploaded file {uploaded_file.name}: {str(e)}")
    
    def _create_error_assessment(self, chunk_id: int, error_message: str) -> TTSQualityAssessment:
        """Create an error assessment when TTS checking fails."""
        return TTSQualityAssessment(
            has_critical_issues=True,
            issues=[TTSIssue(
                issue_type="audio_corruption",
                severity="critical",
                description=f"TTS quality check failed: {error_message}"
            )],
            overall_quality_score=0.0
        )
    
    def _log_assessment(self, chunk_id: int, assessment: TTSQualityAssessment) -> None:
        """Log TTS assessment results."""
        display_chunk_id = chunk_id + 1  # Convert to 1-based for display consistency
        if assessment.has_critical_issues:
            critical_count = len([i for i in assessment.issues if i.severity == "critical"])
            self.logger.log_warning(f"TTS Chunk {display_chunk_id}: Critical issues found - {critical_count} critical issues")
            for issue in assessment.issues:
                if issue.severity == "critical":
                    self.logger.log_warning(f"  - {issue.issue_type}: {issue.description}")
        else:
            self.logger.log_info(f"TTS Chunk {display_chunk_id}: Quality OK (score: {assessment.overall_quality_score:.2f})")
            
        minor_issues = [issue for issue in assessment.issues if issue.severity == "minor"]
        if minor_issues:
            self.logger.log_info(f"TTS Chunk {display_chunk_id}: {len(minor_issues)} minor issues logged")