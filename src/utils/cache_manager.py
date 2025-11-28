"""
Cache management system for PDF document caching with Gemini API.
Handles cache creation, retrieval, and cleanup with TTL support.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from google import genai
from google.genai import types

from ..config import CACHE_TTL_SECONDS, CACHE_LOG_FILE, DEFAULT_MODEL


class CacheManager:
    """Manages PDF document caches for the Gemini API."""
    
    def __init__(self, cache_log_path: str = CACHE_LOG_FILE):
        self.cache_log_path = cache_log_path
        self._ensure_cache_log_exists()
    
    def _ensure_cache_log_exists(self) -> None:
        """Create cache log file if it doesn't exist."""
        if not os.path.exists(self.cache_log_path):
            with open(self.cache_log_path, 'w') as f:
                json.dump({}, f, indent=2)
    
    def _load_cache_log(self) -> Dict:
        """Load the cache log from disk."""
        try:
            with open(self.cache_log_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_cache_log(self, cache_log: Dict) -> None:
        """Save the cache log to disk."""
        with open(self.cache_log_path, 'w') as f:
            json.dump(cache_log, f, indent=2)
    
    def _is_cache_expired(self, expire_time_str: str) -> bool:
        """Check if a cache entry has expired."""
        try:
            expire_time = datetime.fromisoformat(expire_time_str)
            return datetime.now() > expire_time
        except ValueError:
            return True  # Treat invalid timestamps as expired
    
    def _cleanup_expired_caches(self) -> None:
        """Remove expired cache entries from the log."""
        cache_log = self._load_cache_log()
        active_caches = {}
        
        for filename, cache_info in cache_log.items():
            if not self._is_cache_expired(cache_info['expire_time']):
                active_caches[filename] = cache_info
        
        if len(active_caches) != len(cache_log):
            self._save_cache_log(active_caches)
    
    def get_cached_content(self, pdf_filename: str, client=None) -> Optional[str]:
        """
        Get cached content for a PDF file if it exists and hasn't expired.
        
        Args:
            pdf_filename: Name of the source PDF file
            client: Optional Gemini client to validate cache exists
            
        Returns:
            Cache name if valid cache exists, None otherwise
        """
        self._cleanup_expired_caches()
        cache_log = self._load_cache_log()
        
        if pdf_filename in cache_log:
            cache_info = cache_log[pdf_filename]
            if not self._is_cache_expired(cache_info['expire_time']):
                cache_name = cache_info['cache_name']
                
                if client:
                    try:
                        client.models.get(DEFAULT_MODEL).generate_content(
                            "test", 
                            config=types.GenerateContentConfig(
                                cached_content=cache_name,
                                max_output_tokens=1
                            )
                        )
                        return cache_name
                    except Exception as e:
                        print(f"Cache validation failed for {cache_name}, removing from log: {e}")
                        del cache_log[pdf_filename]
                        self._save_cache_log(cache_log)
                        return None
                else:
                    return cache_name
        
        return None
    
    def create_cache(self, pdf_path: str, client) -> Tuple[str, str]:
        """
        Create a new cache for a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            client: Configured Gemini client
            
        Returns:
            Tuple of (cache_name, pdf_filename)
        """
        pdf_filename = os.path.basename(pdf_path)
        
        max_filename_for_upload = 128 - 5  # 123 chars max for upload display_name
        max_filename_for_cache = 128 - 10   # 118 chars max for cache display_name
        safe_filename = pdf_filename[:min(max_filename_for_upload, max_filename_for_cache)]
        uploaded_file = client.files.upload(
            file=pdf_path,
            config=types.UploadFileConfig(display_name=f"PDF: {safe_filename}")
        )
        
        system_instruction = (
            "You are an expert document analyzer. Your job is to answer the user's "
            "query and perform requested tasks based on the document you have access to."
        )
        
        try:
            cache = client.caches.create(
                model=DEFAULT_MODEL,
                config=types.CreateCachedContentConfig(
                    display_name=f"Cache for {safe_filename}",
                    system_instruction=system_instruction,
                    contents=[uploaded_file],
                    ttl=f"{CACHE_TTL_SECONDS}s"
                )
            )
            
            cache_log = self._load_cache_log()
            expire_time = datetime.now() + timedelta(seconds=CACHE_TTL_SECONDS)
            
            cache_log[pdf_filename] = {
                "cache_name": cache.name,
                "expire_time": expire_time.isoformat(),
                "uploaded_file_name": uploaded_file.name
            }
            
            self._save_cache_log(cache_log)
            
            return cache.name, pdf_filename
            
        except Exception as e:
            if "too small" in str(e) or "min_total_token_count" in str(e):
                token_count = "unknown"
                error_str = str(e)
                if "total_token_count=" in error_str:
                    import re
                    match = re.search(r"total_token_count=(\d+)", error_str)
                    if match:
                        token_count = match.group(1)
                
                print(f"Document too small for caching ({pdf_filename}): {token_count} tokens (minimum 2048 required)")
                print("Using direct file access instead")
                return uploaded_file.name, pdf_filename
            else:
                raise
    
    def get_or_create_cache(self, pdf_path: str, client) -> str:
        """
        Get existing cache or create new one for a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            client: Configured Gemini client
            
        Returns:
            Cache name for use in API calls
        """
        pdf_filename = os.path.basename(pdf_path)
        
        cache_name = self.get_cached_content(pdf_filename, client)
        if cache_name:
            return cache_name
        
        cache_name, _ = self.create_cache(pdf_path, client)
        return cache_name
    
    def list_active_caches(self) -> Dict:
        """
        List all active (non-expired) caches.
        
        Returns:
            Dictionary of active cache entries
        """
        self._cleanup_expired_caches()
        return self._load_cache_log()
    
    def delete_cache(self, pdf_filename: str) -> bool:
        """
        Delete a specific cache entry.
        
        Args:
            pdf_filename: Name of the source PDF file
            
        Returns:
            True if cache was deleted, False if not found
        """
        cache_log = self._load_cache_log()
        
        if pdf_filename in cache_log:
            del cache_log[pdf_filename]
            self._save_cache_log(cache_log)
            return True
        
        return False
    
    def clear_all_caches(self) -> None:
        """Clear all cache entries from the log."""
        self._save_cache_log({})