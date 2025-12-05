"""
Dedicated Gemini API Key Manager
Mỗi task/file sẽ được assign 1 key riêng để tránh rate limit conflict
"""

import os
import time
from google import genai
from threading import Lock
import random

from logger_ import get_logger

logger = get_logger("dedicated_key_manager", log_file="logs/dedicated_key_manager.log")


class DedicatedKeyManager:
    """
    Manager for dedicated Gemini API keys
    Each task/file gets its own key to avoid rate limit conflicts
    """
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.api_keys = []
        self.key_assignments = {}
        self.key_in_use = set()
        
        i = 1
        while True:
            key = os.getenv(f"GEMINI_API_KEY_{i}")
            if key:
                self.api_keys.append(key)
                i += 1
            else:
                break
        
        if not self.api_keys:
            raise ValueError("No GEMINI_API_KEY found in environment")
        
        logger.info(f"[DEDICATED] Loaded {len(self.api_keys)} Gemini API keys")
        logger.info(f"[DEDICATED] Each task will get 1 dedicated key")
        logger.info(f"[DEDICATED] Rate limit: 15 RPM = 1 request every 4 seconds per key")
        
        self._initialized = True
    
    def assign_key(self, task_id=None, exclude_keys=None):
        """
        Assign 1 dedicated key cho task
        Args:
            task_id: Task identifier
            exclude_keys: Set of key indices to EXCLUDE (for rotation after failure)
        Returns: (key_index, api_key_string)
        """
        with self._lock:
            if task_id is None:
                import uuid
                task_id = str(uuid.uuid4())
            
            exclude_keys = exclude_keys or set()
            
            if task_id in self.key_assignments:
                key_idx = self.key_assignments[task_id]
                if key_idx not in exclude_keys:
                    logger.info(f"[DEDICATED] Task {task_id[:8]} reusing key #{key_idx + 1}")
                    return key_idx, self.api_keys[key_idx]
                else:
                    logger.warning(f"[DEDICATED] Key #{key_idx + 1} excluded, rotating for task {task_id[:8]}")
                    self.key_in_use.discard(key_idx)
                    del self.key_assignments[task_id]
            
            available_keys = [i for i in range(len(self.api_keys)) 
                            if i not in self.key_in_use and i not in exclude_keys]
            
            if not available_keys:
                non_excluded = [i for i in range(len(self.api_keys)) if i not in exclude_keys]
                if not non_excluded:
                    raise Exception(f"All {len(self.api_keys)} keys have been tried and failed!")
                
                key_idx = random.choice(non_excluded)
                logger.warning(f"[DEDICATED] All keys in use. Sharing key #{key_idx + 1} for task {task_id[:8]}")
            else:
                key_idx = random.choice(available_keys)
                self.key_in_use.add(key_idx)
                logger.info(f"[DEDICATED] Assigned key #{key_idx + 1} to task {task_id[:8]}")
            
            self.key_assignments[task_id] = key_idx
            return key_idx, self.api_keys[key_idx]
    
    def release_key(self, task_id):
        """
        Release key when task is completed
        """
        with self._lock:
            if task_id in self.key_assignments:
                key_idx = self.key_assignments[task_id]
                self.key_in_use.discard(key_idx)
                del self.key_assignments[task_id]
                logger.info(f"[DEDICATED] Released key #{key_idx + 1} from task {task_id[:8]}")
    
    def get_stats(self):
        """Statistics about key usage"""
        with self._lock:
            return {
                'total_keys': len(self.api_keys),
                'keys_in_use': len(self.key_in_use),
                'available_keys': len(self.api_keys) - len(self.key_in_use),
                'active_tasks': len(self.key_assignments)
            }


class DedicatedKeyClient:
    """
    Client uses 1 dedicated key with rate limiting + auto rotation
    Rate limit: 15 requests/minute = 1 request every 4 seconds
    Auto-rotates to new key if current key fails after max retries
    """
    
    def __init__(self, task_id=None):
        self.manager = DedicatedKeyManager()
        self.task_id = task_id or f"task_{id(self)}"
        self.failed_keys = set()
        
        self.key_index, self.api_key = self.manager.assign_key(self.task_id)
        self.genai_client = genai.Client(api_key=self.api_key)
        
        self.min_delay_between_requests = 4.0
        self.last_request_time = 0
        
        logger.info(f"[CLIENT] Initialized for task {self.task_id[:8]} with key #{self.key_index + 1}")
    
    def _rotate_to_new_key(self):
        """
        Rotate to a new key after current key fails
        """
        self.failed_keys.add(self.key_index)
        logger.warning(f"[CLIENT] Rotating from failed key #{self.key_index + 1}. "
                      f"Failed keys so far: {sorted([k+1 for k in self.failed_keys])}")
        
        try:
            self.key_index, self.api_key = self.manager.assign_key(
                self.task_id, 
                exclude_keys=self.failed_keys
            )
            self.genai_client = genai.Client(api_key=self.api_key)
            self.last_request_time = 0
            logger.info(f"[CLIENT] Rotated to new key #{self.key_index + 1}")
            return True
        except Exception as e:
            logger.error(f"[CLIENT] Cannot rotate: {e}")
            return False
    
    def call_with_retry(self, prompt, max_retries=5, model="models/gemini-2.5-flash-lite", **config):
        """
        Call API with dedicated key and rate limiting + auto rotation
        
        Strategy:
        - Try max_retries times with current key
        - If all retries fail → rotate to new key and try again
        - Repeat until success or all keys exhausted
        
        Args:
            prompt: Text prompt
            max_retries: Max retry attempts PER KEY
            model: Model name (gemini-2.5-flash-lite hoặc gemini-2.5-flash-lite-preview)
            **config: Additional config
        """
        max_key_rotations = len(self.manager.api_keys)  # Try all available keys
        
        for rotation_attempt in range(max_key_rotations):
            current_key_num = self.key_index + 1
            logger.debug(f"[CLIENT] Using key #{current_key_num} (rotation {rotation_attempt + 1}/{max_key_rotations})")
            
            for attempt in range(max_retries):
                try:
                    current_time = time.time()
                    time_since_last = current_time - self.last_request_time
                    
                    if time_since_last < self.min_delay_between_requests:
                        sleep_time = self.min_delay_between_requests - time_since_last
                        logger.debug(f"[CLIENT #{current_key_num}] Rate limiting: sleeping {sleep_time:.1f}s")
                        time.sleep(sleep_time)
                    
                    self.last_request_time = time.time()
                    
                    response_obj = self.genai_client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=config if config else {"temperature": 0}
                    )
                    
                    if response_obj and hasattr(response_obj, "text") and response_obj.text:
                        return response_obj.text.strip()
                    elif response_obj and hasattr(response_obj, "candidates") and response_obj.candidates:
                        candidate = response_obj.candidates[0]
                        if hasattr(candidate, "finish_reason"):
                            finish_reason = str(candidate.finish_reason)
                            logger.warning(f"[CLIENT #{current_key_num}] Response blocked: {finish_reason}")
                            if attempt < max_retries - 1:
                                time.sleep(3)
                                continue
                            else:
                                raise Exception(f"Response blocked after {max_retries} attempts: {finish_reason}")
                    else:
                        logger.warning(f"[CLIENT #{current_key_num}] Empty response")
                        if attempt < max_retries - 1:
                            time.sleep(3)
                            continue
                        else:
                            raise Exception(f"Empty response after {max_retries} attempts")
                    
                except Exception as e:
                    error_str = str(e)
                    
                    if "PERMISSION_DENIED" in error_str and "CONSUMER_SUSPENDED" in error_str:
                        logger.error(f"[CLIENT #{current_key_num}] KEY SUSPENDED! Rotating immediately...")
                        logger.error(f"Suspended key error: {error_str[:200]}...")
                        break  # Exit retry loop immediately, go to rotation
                    
                    elif "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                        wait_time = 10 + (attempt * 5)  # Exponential backoff: 10, 15, 20, 25, 30s
                        logger.warning(f"[CLIENT #{current_key_num}] Rate limit hit. "
                                     f"Waiting {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        
                        if attempt == max_retries - 1:
                            logger.error(f"[CLIENT #{current_key_num}] Rate limit persists after {max_retries} retries")
                            break  # Exit retry loop, go to rotation
                        continue
                    
                    elif "500" in error_str or "503" in error_str or "INTERNAL" in error_str:
                        logger.warning(f"[CLIENT #{current_key_num}] Server error. Retrying in 5s...")
                        time.sleep(5.0)
                        continue
                    
                    elif "disconnected" in error_str.lower() or "connection" in error_str.lower():
                        logger.warning(f"[CLIENT #{current_key_num}] Network error. Retrying in 3s...")
                        time.sleep(3.0)
                        continue
                    
                    else:
                        logger.error(f"[CLIENT #{current_key_num}] Unexpected error: {error_str}")
                        raise
            
            logger.error(f"[CLIENT] Key #{current_key_num} exhausted after {max_retries} retries")
            
            if rotation_attempt < max_key_rotations - 1:
                if not self._rotate_to_new_key():
                    break
                logger.info(f"[CLIENT] Retrying with new key #{self.key_index + 1}...")
                time.sleep(5)
            else:
                logger.error(f"[CLIENT] All {max_key_rotations} keys exhausted!")
        
        raise Exception(f"All keys exhausted! Tried {len(self.failed_keys) + 1} different keys.")
    
    def __del__(self):
        """Release key when client is destroyed"""
        try:
            self.manager.release_key(self.task_id)
        except:
            pass


# Global manager instance
_dedicated_manager = None

def get_dedicated_manager():
    """Get global dedicated manager instance"""
    global _dedicated_manager
    if _dedicated_manager is None:
        _dedicated_manager = DedicatedKeyManager()
    return _dedicated_manager


def create_dedicated_client(task_id=None):
    """
    Create a dedicated client with its own API key
    
    Usage:
        client = create_dedicated_client(task_id="import_file_123")
        response = client.call_with_retry("Your prompt here")
    """
    return DedicatedKeyClient(task_id)