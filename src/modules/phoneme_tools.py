"""
Phoneme Tools for LLM interaction with phoneme dictionary management.
Implements function calling tools for Google GenAI to add/remove phoneme mappings.
"""

import json
import os
from typing import Dict, List, Optional

from ..modules.phoneme_enhancer import PhonemeEnhancer


class PhonemeTools:
    """
    Tools for managing phoneme dictionary through LLM function calls.
    Provides add_phoneme and remove_phoneme functions for Google GenAI.
    """
    
    def __init__(self, phoneme_enhancer: PhonemeEnhancer):
        self.enhancer = phoneme_enhancer
    
    def add_phoneme(self, word: str, phoneme: str) -> str:
        """Add a new word and its IPA phoneme pronunciation to the phoneme dictionary.
        
        Use this when the user wants to add a new word pronunciation or modify an existing one.
        
        Args:
            word: The word to add phoneme mapping for (exact case and spelling as it appears in the script)
            phoneme: The IPA phoneme pronunciation for the word (International Phonetic Alphabet)
            
        Returns:
            Success message confirming the addition
        """
        try:
            if not word or not word.strip():
                return "Error: Word cannot be empty"
            
            if not phoneme or not phoneme.strip():
                return "Error: Phoneme cannot be empty"
            
            word = word.strip()
            phoneme = phoneme.strip()
            
            self.enhancer.add_manual_phoneme(word, phoneme)
            
            return f"Success: Added '{word}' with pronunciation '{phoneme}'"
            
        except Exception as e:
            return f"Error adding phoneme: {str(e)}"
    
    def remove_phoneme(self, word: str) -> str:
        """Remove a word from the phoneme dictionary.
        
        Use this when the user wants to remove a word from special pronunciation handling.
        
        Args:
            word: The exact word to remove from the phoneme dictionary
            
        Returns:
            Success or error message
        """
        try:
            if not word or not word.strip():
                return "Error: Word cannot be empty"
            
            word = word.strip()
            
            removed = self.enhancer.remove_phoneme(word)
            
            if removed:
                return f"Success: Removed '{word}' from phoneme dictionary"
            else:
                return f"Error: Word '{word}' not found in phoneme dictionary"
                
        except Exception as e:
            return f"Error removing phoneme: {str(e)}"
    
    def get_callable_tools(self) -> List:
        """
        Get callable tools for Google GenAI function calling.
        
        Returns:
            List of callable functions for LLM use
        """
        return [self.add_phoneme, self.remove_phoneme]