"""
Phoneme enhancement module for adding SSML phoneme tags to scripts.
Uses persistent dictionary for memory and learning across runs.
"""

import json
import os
import re
import time
from typing import Dict, List

from google import genai
from google.genai import types

from ..config import COMMON_TERMS_TO_SKIP, THINKING_BUDGET, DEFAULT_MODEL, SAVE_NEW_PHONEMES_PERMANENTLY
from ..models.schemas import IdentifiedTerms, TermIPA, TokenUsage
from ..utils.logging_utils import LoggingManager

import openai


class PhonemeEnhancer:
    """Manages phoneme tagging for TTS enhancement with persistent memory."""
    
    def __init__(self, client, logger: LoggingManager, prompt_logger=None,
                 phoneme_dict_path: str = "phoneme_dict.json"):
        self.client = client  # Gemini client for other operations
        self.logger = logger
        self.prompt_logger = prompt_logger
        self.phoneme_dict_path = phoneme_dict_path
        self.phoneme_dict = self._load_phoneme_dictionary()
        
        openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openai_client = openai.OpenAI(api_key=openai_api_key) if openai_api_key else None
    
    def _load_phoneme_dictionary(self) -> Dict[str, str]:
        """Load the phoneme dictionary from disk."""
        if os.path.exists(self.phoneme_dict_path):
            try:
                with open(self.phoneme_dict_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return {}
        else:
            initial_dict = {
                "TACE": "teɪs",
                "Ktrans": "keɪ træns",
                "DWI": "diː dʌbəl juː aɪ",
                "PCNSL": "piː siː ɛn ɛs ɛl",
                "ADC": "eɪ diː siː",
                "ROI": "ɑːr oʊ aɪ",
                "PACS": "pæks",
                "DICOM": "daɪkɒm",
                "SUV": "ɛs juː viː",
                "FLAIR": "flɛər",
                "STIR": "stɜːr",
                "T1WI": "tiː wʌn dʌbəl juː aɪ",
                "T2WI": "tiː tuː dʌbəl juː aɪ"
            }
            self._save_phoneme_dictionary(initial_dict)
            return initial_dict
    
    def _save_phoneme_dictionary(self, dictionary: Dict[str, str]) -> None:
        """Save the phoneme dictionary to disk."""
        with open(self.phoneme_dict_path, 'w', encoding='utf-8') as f:
            json.dump(dictionary, f, indent=2, ensure_ascii=False)
    
    def _would_term_be_applied(self, term: str, script: str) -> bool:
        """
        Check if a term would actually be applied to the script (avoiding contractions).
        
        Args:
            term: The term to check
            script: Script text to check against
            
        Returns:
            True if the term would be applied (found as standalone word, not in contraction)
        """
        flags = 0
        
        if len(term) <= 2:
            pattern = r'\b' + re.escape(term) + r'\b'
            matches = list(re.finditer(pattern, script, flags=flags))
            
            for match in matches:
                start_pos = match.start()
                if start_pos >= 2 and script[start_pos-1] == "'" and script[start_pos-2].isalnum():
                    continue  # Skip this match - it's part of a contraction
                else:
                    return True  # Found at least one valid standalone occurrence
            return False  # All matches were in contractions
        else:
            pattern = r'\b' + re.escape(term) + r'\b'
            return bool(re.search(pattern, script, flags=flags))
    
    def _apply_known_phonemes(self, script: str) -> str:
        """Apply all known phoneme tags to the script."""
        enhanced_script = script
        
        for term, ipa in self.phoneme_dict.items():
            flags = 0
            
            phoneme_tag = f'<phoneme alphabet="ipa" ph="{ipa}">{term}</phoneme>'
            
            if len(term) <= 2:
                pattern = r'\b' + re.escape(term) + r'\b'
                matches = list(re.finditer(pattern, enhanced_script, flags=flags))
                
                for match in reversed(matches):
                    start_pos = match.start()
                    if start_pos >= 2 and enhanced_script[start_pos-1] == "'" and enhanced_script[start_pos-2].isalnum():
                        continue  # Skip this match - it's part of a contraction
                    else:
                        enhanced_script = enhanced_script[:start_pos] + phoneme_tag + enhanced_script[match.end():]
            else:
                pattern = r'\b' + re.escape(term) + r'\b'
                enhanced_script = re.sub(pattern, phoneme_tag, enhanced_script, flags=flags)
        
        return enhanced_script
    
    def _extract_already_tagged_terms(self, script: str) -> List[str]:
        """Extract terms that are already tagged with phoneme tags."""
        pattern = r'<phoneme[^>]*>([^<]+)</phoneme>'
        matches = re.findall(pattern, script, re.IGNORECASE)
        return matches

    def _build_identification_prompt_for_o3(self, script: str, already_tagged: List[str]) -> str:
        """Build JSON-structured prompt for o3 phoneme identification."""
        
        existing_terms = list(self.phoneme_dict.keys())
        
        return f"""[TASK] Identify and provide the IPA pronunciation of medical/technical acronyms, in a way an American professional radiologist or doctor would read it. Target extremely long words, Roman numerals, and ambiguous abbreviations in a script text that an educated layperson would have trouble pronouncing.
         
Do NOT tag normal english words you can find in the dictionary. Provide their IPA pronunciations, as if an American doctor would pronounce them.
Only VERY STRANGE words should be tagged. Only WORDS or uniquely compound terms are okay (e.g. De Quervain, Bethesda IV).
Do NOT tag common chemical, biological terms (e.g., "alanine", "glucose").
Skip long words that are made up entirely of familiar English/medical roots (even if multi-part or compound), as long as each component is common, dictionary-found, or its pronunciation is predictable by an educated layperson. Only tag terms that have unpredictable pronunciations, unique/rare root forms, Roman numerals, or ambiguous abbreviations.
The terms are case-sensitive. Address by making separate tags for upper- and lower-case words. 

See this list to get a feel of what constitutes the criteria for NOT tagging:
==== TERMS LIKE THESE, SKIP ====
{', '.join(COMMON_TERMS_TO_SKIP)}

--- SCRIPT TO ANALYZE ---
{script}

--- EXISTING DICTIONARY TERMS ---
{', '.join(existing_terms)}

--- RESPONSE FORMAT ---
IMPORTANT: You MUST return your response as a valid JSON object with this exact structure:
{{
  "terms": [
    {{"term": "exact_term_from_script", "ipa": "aɪ_pi_eɪ_pronunciation"}},
    {{"term": "another_term", "ipa": "pronunciation"}}
  ]
}}

If you find NO terms that need phoneme tags, return:
{{
  "terms": []
}}

NEVER return plain text. ALWAYS return valid JSON. Only include terms that truly need phoneme tags. The "term" field must match the exact casing and form as it appears in the script."""
    
    def identify_additional_terms(self, script: str) -> IdentifiedTerms:
        """
        Identify new terms that need phoneme tags using OpenAI o3.
        
        Args:
            script: Script text to analyze
            
        Returns:
            IdentifiedTerms with new terms requiring phoneme tags
        """
        try:
            already_tagged = self._extract_already_tagged_terms(script)
            
            prompt = self._build_identification_prompt_for_o3(script, already_tagged)
            
            start_time = time.time()
            response = None
            response_text = None
            
            try:
                response = self.openai_client.chat.completions.create(
                    model="o3",
                    messages=[
                        {"role": "system", "content": "You are an expert medical pronunciation specialist. Your task is to identify technical terms that need phoneme tags for text-to-speech accuracy."},
                        {"role": "user", "content": prompt}
                    ],
                    max_completion_tokens=25000  # o3 needs lots of tokens for reasoning
                )
                execution_time = time.time() - start_time
                
                response_text = response.choices[0].message.content
                if response_text is None:
                    response_text = ""
                else:
                    response_text = response_text.strip()
                
            except Exception as api_error:
                execution_time = time.time() - start_time
                error_msg = f"o3 API call failed: {type(api_error).__name__}: {str(api_error)}"
                
                if self.prompt_logger:
                    self.prompt_logger.log_interaction(
                        stage="phoneme_identification_o3_failed",
                        prompt=prompt,
                        response=f"ERROR: {error_msg}",
                        metadata={
                            "script_length": len(script),
                            "already_tagged_count": len(already_tagged),
                            "model": "o3",
                            "error": error_msg,
                            "execution_time": execution_time
                        }
                    )
                
                raise ValueError(error_msg)
            
            if response_text.startswith("```json"):
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif response_text.startswith("```"):
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            try:
                parsed_data = json.loads(response_text)
            except json.JSONDecodeError as json_error:
                if any(phrase in response_text.lower() for phrase in [
                    "no terms", "no additional terms", "none", "no phoneme", 
                    "no words", "nothing to tag", "no medical terms"
                ]):
                    parsed_data = {"terms": []}
                else:
                    debug_info = {
                        "response_length": len(response_text),
                        "response_is_empty": response_text == "",
                        "response_repr": repr(response_text),
                        "raw_response_choices": len(response.choices) if response else "NO_RESPONSE",
                        "raw_response_content": response.choices[0].message.content if response and response.choices else "NO_CONTENT"
                    }
                    
                    if self.prompt_logger:
                        self.prompt_logger.log_interaction(
                            stage="phoneme_identification_o3_invalid_json",
                            prompt=prompt,
                            response=response_text if response_text else "[EMPTY_RESPONSE]",
                            metadata={
                                "script_length": len(script),
                                "already_tagged_count": len(already_tagged),
                                "model": "o3",
                                "error": f"JSON parsing failed: {str(json_error)}",
                                "raw_o3_response": response_text,
                                "debug_info": debug_info,
                                "execution_time": execution_time
                            }
                        )
                    
                    if response_text == "":
                        raise ValueError(f"o3 returned completely empty response (no content)")
                    else:
                        raise ValueError(f"o3 returned invalid JSON: {response_text}")
            
            terms = []
            for term_data in parsed_data.get("terms", []):
                terms.append(TermIPA(
                    term=term_data["term"],
                    ipa=term_data["ipa"]
                ))
            
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            total_tokens = prompt_tokens + completion_tokens
            
            token_usage = TokenUsage(
                prompt_tokens=prompt_tokens,
                cached_tokens=0,
                candidate_tokens=completion_tokens,
                total_tokens=total_tokens
            )
            self.logger.log_stage_metrics("phoneme_identification_o3", token_usage, "o3", execution_time=execution_time)
            
            if self.prompt_logger:
                self.prompt_logger.log_interaction(
                    stage="phoneme_identification_o3",
                    prompt=prompt,
                    response=response_text,
                    metadata={
                        "script_length": len(script),
                        "already_tagged_count": len(already_tagged),
                        "terms_found": len(parsed_data.get("terms", [])),
                        "model": "o3",
                        "raw_o3_response": response_text,
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens
                        }
                    }
                )
            
            filtered_terms = []
            
            for term_ipa in terms:
                term = term_ipa.term
                if not any(existing_term.lower() == term.lower() for existing_term in self.phoneme_dict.keys()):
                    filtered_terms.append(term_ipa)
                else:
                    print(f"  → Skipping duplicate term: {term} (already in dictionary)")
            
            filtered_response = IdentifiedTerms(terms=filtered_terms)
            
            original_count = len(terms)
            filtered_count = len(filtered_terms)
            if original_count > filtered_count:
                print(f"  → Filtered out {original_count - filtered_count} duplicate terms from o3 suggestions")
            
            return filtered_response
            
        except Exception as e:
            self.logger.log_error(f"Exception in identify_additional_terms: {type(e).__name__}: {str(e)}")
            raise
    
    def apply_new_phoneme_tags(self, script: str, identified_terms: IdentifiedTerms) -> str:
        """
        Apply newly identified phoneme tags to the script.
        
        Args:
            script: Current script text
            identified_terms: New terms to tag
            
        Returns:
            Script with new phoneme tags applied
        """
        enhanced_script = script
        
        for term_ipa in identified_terms.terms:
            term = term_ipa.term
            ipa = term_ipa.ipa
            
            flags = 0
            
            phoneme_tag = f'<phoneme alphabet="ipa" ph="{ipa}">{term}</phoneme>'
            
            if len(term) <= 2:
                pattern = r'\b' + re.escape(term) + r'\b'
                matches = list(re.finditer(pattern, enhanced_script, flags=flags))
                
                for match in reversed(matches):
                    start_pos = match.start()
                    if start_pos >= 2 and enhanced_script[start_pos-1] == "'" and enhanced_script[start_pos-2].isalnum():
                        continue  # Skip this match - it's part of a contraction
                    else:
                        enhanced_script = enhanced_script[:start_pos] + phoneme_tag + enhanced_script[match.end():]
            else:
                pattern = r'\b' + re.escape(term) + r'\b'
                enhanced_script = re.sub(pattern, phoneme_tag, enhanced_script, flags=flags)
            
            self.phoneme_dict[term] = ipa
        
        if SAVE_NEW_PHONEMES_PERMANENTLY:
            self._save_phoneme_dictionary(self.phoneme_dict)
        
        return enhanced_script
    
    def enhance_script_with_phonemes(self, script: str) -> str:
        """
        Complete phoneme enhancement process for a script.
        
        Args:
            script: Input script text
            
        Returns:
            Script enhanced with SSML phoneme tags
        """
        initial_dict_size = len(self.phoneme_dict)
        print(f"  → Starting phoneme enhancement (dictionary has {initial_dict_size} known terms)")
        
        original_dict = None
        if not SAVE_NEW_PHONEMES_PERMANENTLY:
            original_dict = self.phoneme_dict.copy()
        
        script_with_known = self._apply_known_phonemes(script)
        known_applications = script_with_known.count('<phoneme')
        if known_applications > 0:
            print(f"  → Applied {known_applications} known phoneme tags")
        
        identified_terms = self.identify_additional_terms(script_with_known)
        
        if identified_terms.terms:
            print(f"  → Found {len(identified_terms.terms)} new terms needing phonemes")
            
            final_script = self.apply_new_phoneme_tags(script_with_known, identified_terms)
            
            if SAVE_NEW_PHONEMES_PERMANENTLY:
                new_dict_size = len(self.phoneme_dict)
                new_terms_added = new_dict_size - initial_dict_size
                if new_terms_added > 0:
                    print(f"  → Added {new_terms_added} new terms to phoneme dictionary")
            else:
                print(f"  → Applied {len(identified_terms.terms)} new phoneme tags (temporary - not saved to dictionary)")
        else:
            print("  → No new terms requiring phonemes found")
            final_script = script_with_known
        
        total_phoneme_tags = final_script.count('<phoneme')
        print(f"  ✓ Phoneme enhancement complete: {total_phoneme_tags} total phoneme tags applied")
        
        return final_script
    
    def restore_original_dict_if_temporary(self):
        """Restore original dictionary after TTS generation if in temporary mode."""
        if not SAVE_NEW_PHONEMES_PERMANENTLY:
            self.phoneme_dict = self._load_phoneme_dictionary()
            print(f"  → Restored original phoneme dictionary for next document")
    
    def get_phoneme_dictionary(self) -> Dict[str, str]:
        """Get the current phoneme dictionary."""
        return self.phoneme_dict.copy()
    
    def add_manual_phoneme(self, term: str, ipa: str) -> None:
        """
        Manually add a phoneme mapping to the dictionary.
        
        Args:
            term: The term to add
            ipa: The IPA pronunciation
        """
        self.phoneme_dict[term] = ipa
        self._save_phoneme_dictionary(self.phoneme_dict)
    
    def remove_phoneme(self, term: str) -> bool:
        """
        Remove a phoneme mapping from the dictionary.
        
        Args:
            term: The term to remove
            
        Returns:
            True if term was removed, False if not found
        """
        if term in self.phoneme_dict:
            del self.phoneme_dict[term]
            self._save_phoneme_dictionary(self.phoneme_dict)
            return True
        return False
    
    def export_phoneme_dictionary(self, export_path: str) -> None:
        """
        Export the phoneme dictionary to a specified path.
        
        Args:
            export_path: Path to export the dictionary
        """
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(self.phoneme_dict, f, indent=2, ensure_ascii=False)
    
    def import_phoneme_dictionary(self, import_path: str, merge: bool = True) -> None:
        """
        Import phoneme dictionary from a file.
        
        Args:
            import_path: Path to import from
            merge: If True, merge with existing dictionary. If False, replace entirely.
        """
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                imported_dict = json.load(f)
            
            if merge:
                self.phoneme_dict.update(imported_dict)
            else:
                self.phoneme_dict = imported_dict
            
            self._save_phoneme_dictionary(self.phoneme_dict)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            raise ValueError(f"Failed to import phoneme dictionary: {e}")
    
    def remove_words_from_phoneme_tags_for_tts(self, script: str) -> str:
        """
        Remove the words inside phoneme tags for TTS generation, keeping only the pronunciation.
        
        Args:
            script: Script with phoneme tags like <phoneme alphabet="ipa" ph="teɪs">TACE</phoneme>
            
        Returns:
            Script with empty phoneme tags like <phoneme alphabet="ipa" ph="teɪs"></phoneme>
        """
        phoneme_pattern = r'(<phoneme[^>]*>)[^<]+(</phoneme>)'
        tts_script = re.sub(phoneme_pattern, r'\1\2', script, flags=re.IGNORECASE)
        
        return tts_script
    
    def validate_phoneme_tags(self, script: str) -> List[str]:
        """
        Validate SSML phoneme tags in a script.
        
        Args:
            script: Script to validate
            
        Returns:
            List of validation errors, empty if no errors
        """
        errors = []
        
        phoneme_pattern = r'<phoneme[^>]*>([^<]+)</phoneme>'
        matches = re.finditer(phoneme_pattern, script, re.IGNORECASE)
        
        for match in matches:
            full_tag = match.group(0)
            term = match.group(1)
            
            if 'alphabet="ipa"' not in full_tag:
                errors.append(f"Missing alphabet attribute in tag: {full_tag}")
            
            if 'ph="' not in full_tag:
                errors.append(f"Missing ph attribute in tag: {full_tag}")
            
            ph_match = re.search(r'ph="([^"]*)"', full_tag)
            if ph_match:
                ipa = ph_match.group(1)
                if not ipa.strip():
                    errors.append(f"Empty IPA pronunciation in tag: {full_tag}")
            else:
                errors.append(f"Could not extract IPA from tag: {full_tag}")
        
        return errors
    
    def verify_and_enhance_script(self, original_script: str, edited_script: str) -> str:
        """
        Verify script changes and apply phoneme enhancement using existing dictionary.
        
        This function compares the original script with the user-edited version and 
        applies phoneme tags from the existing dictionary without making LLM calls.
        
        Args:
            original_script: The original human-readable script
            edited_script: The user-edited script from the frontend
            
        Returns:
            Script with phoneme enhancements applied from existing dictionary
        """
        self.logger.log_info(f"Verifying script changes (original: {len(original_script)} chars, edited: {len(edited_script)} chars)")
        
        enhanced_script = self._apply_existing_phoneme_tags(edited_script)
        
        self.logger.log_info(f"Applied {len(self.phoneme_dict)} phoneme mappings from existing dictionary")
        
        return enhanced_script
    
    def _apply_existing_phoneme_tags(self, script: str) -> str:
        """
        Apply phoneme tags from existing dictionary without LLM calls.
        
        Args:
            script: Script text to enhance
            
        Returns:
            Script with phoneme tags applied
        """
        enhanced_script = script
        
        for term, ipa in self.phoneme_dict.items():
            flags = 0
            
            phoneme_tag = f'<phoneme alphabet="ipa" ph="{ipa}">{term}</phoneme>'
            
            pattern = r'\b' + re.escape(term) + r'\b'
            enhanced_script = re.sub(
                pattern, 
                phoneme_tag, 
                enhanced_script, 
                flags=flags
            )
        
        return enhanced_script