"""
Phoneme Processor for handling user requests to modify phoneme dictionary.
Uses Google GenAI with function calling to process natural language requests.
"""

import time
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from google import genai
from google.genai import types

from ..modules.phoneme_enhancer import PhonemeEnhancer
from ..modules.phoneme_tools import PhonemeTools
from ..models.schemas import TokenUsage
from ..utils.logging_utils import LoggingManager

from ..config import THINKING_BUDGET_EDITOR, DEFAULT_MODEL


class PhonemeProcessor:
    """
    Processes user requests to modify phoneme dictionary using LLM function calling.
    """
    
    def __init__(self, client: genai.Client, logger: LoggingManager, 
                 phoneme_enhancer: PhonemeEnhancer, prompt_logger=None):
        self.client = client
        self.logger = logger
        self.phoneme_enhancer = phoneme_enhancer
        self.prompt_logger = prompt_logger
        self.tools = PhonemeTools(phoneme_enhancer)
    
    
    def _build_unified_prompt(self, user_request: str, current_dictionary: Dict[str, str], 
                             identified_new_terms: List[Dict] = None) -> str:
        """
        Build unified prompt for phoneme modification with current dictionary state.
        
        Args:
            user_request: User's natural language request
            current_dictionary: Current phoneme dictionary
            identified_new_terms: Optional list of newly identified terms
            
        Returns:
            Formatted prompt for processing modifications
        """
        dict_items = []
        for word, ipa in current_dictionary.items():
            dict_items.append(f"- {word}: {ipa}")
        
        current_dict_text = "\n".join(dict_items) if dict_items else "No existing phoneme mappings"
        
        new_terms_section = ""
        if identified_new_terms:
            new_terms_items = []
            for term in identified_new_terms:
                word = term.get('term', '')
                ipa = term.get('ipa', '')
                new_terms_items.append(f"- {word}: {ipa}")
            
            if new_terms_items:
                new_terms_text = "\n".join(new_terms_items)
                new_terms_section = f"\n\nNEWLY IDENTIFIED TERMS THAT WERE ADDED:\n{new_terms_text}"
        
        prompt = f"""You are helping a user modify phoneme pronunciations for a podcast text-to-speech system.

CURRENT PHONEME DICTIONARY:
{current_dict_text}{new_terms_section}

Please process the user's request using the available tools:
- Use add_phoneme(word, phoneme) to add new words or modify existing pronunciations
- Use remove_phoneme(word) to remove words from the dictionary

For pronunciation descriptions, convert them to accurate IPA (International Phonetic Alphabet) notation.

Examples of user requests and how to handle them:
- "I want Taewoong to be pronounced Tae (as in tae kwon do) - oong (long o like in dragoon followed by ng)" 
  → add_phoneme("Taewoong", "teɪˈwʊŋ")
- "Remove the word p-value, that isn't a hard word"
  → remove_phoneme("p-value")
- "Add the word BERT pronounced as 'bert'"
  → add_phoneme("BERT", "bɜːrt")

Process all requested changes and when complete, respond with exactly: "All tasks are done."

User request: {user_request}
"""
        
        return prompt
    
    
    def process_user_modifications(self, user_request: str, identified_new_terms: List[Dict] = None) -> Tuple[str, str]:
        """
        Process user's natural language request to modify phoneme dictionary.
        
        Args:
            user_request: User's request in natural language
            identified_new_terms: Optional list of newly identified terms
            
        Returns:
            Tuple of (LLM response, conversation log file path)
        """
        try:
            dictionary_before = self.phoneme_enhancer.get_phoneme_dictionary().copy()
            
            prompt = self._build_unified_prompt(user_request, dictionary_before, identified_new_terms)
            
            start_time = time.time()
            response = self.client.models.generate_content(
                model=DEFAULT_MODEL,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    tools=self.tools.get_callable_tools(),
                    thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET_EDITOR)
                )
            )
            execution_time = time.time() - start_time
            
            response_text = response.text if response.text else "No response generated"
            
            dictionary_after = self.phoneme_enhancer.get_phoneme_dictionary().copy()
            
            conversation_log = self._create_conversation_log(
                user_request=user_request,
                prompt=prompt,
                response=response,
                response_text=response_text,
                dictionary_before=dictionary_before,
                dictionary_after=dictionary_after,
                execution_time=execution_time
            )
            
            log_file_path = self._save_conversation_log(conversation_log)
            
            
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                token_usage = TokenUsage(
                    prompt_tokens=usage.prompt_token_count,
                    cached_tokens=0,
                    candidate_tokens=usage.candidates_token_count,
                    total_tokens=usage.total_token_count
                )
                self.logger.log_stage_metrics("phoneme_modification", token_usage, DEFAULT_MODEL, execution_time=execution_time)
            
            return response_text.strip(), log_file_path
            
        except Exception as e:
            self.logger.log_error(f"Error processing user modifications: {str(e)}")
            raise
    
    def _analyze_dictionary_changes(self, before: Dict[str, str], after: Dict[str, str]) -> Dict:
        """
        Analyze what changes were made to the dictionary.
        
        Args:
            before: Dictionary before changes
            after: Dictionary after changes
            
        Returns:
            Dictionary describing the changes made
        """
        changes = {
            "added": {},
            "removed": {},
            "modified": {}
        }
        
        for word, ipa in after.items():
            if word not in before:
                changes["added"][word] = ipa
        
        for word, ipa in before.items():
            if word not in after:
                changes["removed"][word] = ipa
        
        for word, ipa in after.items():
            if word in before and before[word] != ipa:
                changes["modified"][word] = {
                    "from": before[word],
                    "to": ipa
                }
        
        return changes
    
    def _create_conversation_log(self, user_request: str, prompt: str, response, response_text: str,
                                dictionary_before: Dict[str, str], dictionary_after: Dict[str, str],
                                execution_time: float) -> Dict:
        """
        Create a conversation log in chat format.
        
        Returns:
            Dictionary containing the conversation log
        """
        conversation = {
            "timestamp": datetime.now().isoformat(),
            "model": DEFAULT_MODEL,
            "execution_time_seconds": execution_time,
            "conversation": []
        }
        
        conversation["conversation"].append({
            "role": "user",
            "content": prompt
        })
        
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            conversation["token_usage"] = {
                "prompt_tokens": usage.prompt_token_count,
                "candidate_tokens": usage.candidates_token_count,
                "total_tokens": usage.total_token_count
            }
        
        if response_text and response_text.strip():
            conversation["conversation"].append({
                "role": "assistant",
                "content": response_text
            })
        
        changes = self._analyze_dictionary_changes(dictionary_before, dictionary_after)
        if any(changes.values()):
            change_summary = []
            if changes["added"]:
                change_summary.append(f"Added: {list(changes['added'].keys())}")
            if changes["removed"]:
                change_summary.append(f"Removed: {list(changes['removed'].keys())}")
            if changes["modified"]:
                change_summary.append(f"Modified: {list(changes['modified'].keys())}")
            
            conversation["dictionary_changes"] = {
                "summary": ", ".join(change_summary),
                "details": changes
            }
        
        return conversation
    
    def _save_conversation_log(self, conversation_log: Dict) -> str:
        """
        Save conversation log to file in the correct location as phoneme_modification log.
        
        Returns:
            Path to saved log file
        """
        if self.prompt_logger and hasattr(self.prompt_logger, 'logs_dir'):
            llm_logs_dir = self.prompt_logger.logs_dir
            
            timestamp = datetime.now().strftime("%H%M%S")
            
            log_number = f"{self.prompt_logger.interaction_count + 1:02d}"
            
            log_filename = f"{log_number}_{timestamp}_phoneme_modification.json"
            log_file_path = os.path.join(llm_logs_dir, log_filename)
            
            self.prompt_logger.interaction_count += 1
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"phoneme_conversation_{timestamp}.json"
            log_file_path = os.path.join(os.getcwd(), log_filename)
        
        with open(log_file_path, 'w', encoding='utf-8') as f:
            json.dump(conversation_log, f, indent=2, ensure_ascii=False)
        
        return log_file_path