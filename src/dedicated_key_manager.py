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
    Manager phân phối keys độc quyền cho từng task
    Mỗi task nhận 1 key riêng, tránh conflict khi chạy song song
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
        self.key_assignments = {}  # {task_id: key_index}
        self.key_in_use = set()    # Set of key indices currently in use
        
        # Load all keys
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
        
        logger.info(f"🔑 [DEDICATED] Loaded {len(self.api_keys)} Gemini API keys")
        logger.info(f"📌 [DEDICATED] Each task will get 1 dedicated key")
        logger.info(f"⏱️  [DEDICATED] Rate limit: 15 RPM = 1 request every 4 seconds per key")
        
        self._initialized = True
    
    def assign_key(self, task_id=None):
        """
        Assign 1 dedicated key cho task
        Returns: (key_index, api_key_string)
        """
        with self._lock:
            if task_id is None:
                import uuid
                task_id = str(uuid.uuid4())
            
            # Check if task already has a key
            if task_id in self.key_assignments:
                key_idx = self.key_assignments[task_id]
                logger.info(f"♻️  [DEDICATED] Task {task_id[:8]} reusing key #{key_idx + 1}")
                return key_idx, self.api_keys[key_idx]
            
            # Find first available key (not in use)
            available_keys = [i for i in range(len(self.api_keys)) if i not in self.key_in_use]
            
            if not available_keys:
                # All keys in use, pick least-recently-used (round-robin)
                key_idx = len(self.key_assignments) % len(self.api_keys)
                logger.warning(f"⚠️  [DEDICATED] All keys in use. Sharing key #{key_idx + 1} for task {task_id[:8]}")
            else:
                # Pick random from available keys to distribute load
                key_idx = random.choice(available_keys)
                self.key_in_use.add(key_idx)
                logger.info(f"✅ [DEDICATED] Assigned key #{key_idx + 1} to task {task_id[:8]}")
            
            self.key_assignments[task_id] = key_idx
            return key_idx, self.api_keys[key_idx]
    
    def release_key(self, task_id):
        """Release key khi task hoàn thành"""
        with self._lock:
            if task_id in self.key_assignments:
                key_idx = self.key_assignments[task_id]
                self.key_in_use.discard(key_idx)
                del self.key_assignments[task_id]
                logger.info(f"🔓 [DEDICATED] Released key #{key_idx + 1} from task {task_id[:8]}")
    
    def get_stats(self):
        """Thống kê về key usage"""
        with self._lock:
            return {
                'total_keys': len(self.api_keys),
                'keys_in_use': len(self.key_in_use),
                'available_keys': len(self.api_keys) - len(self.key_in_use),
                'active_tasks': len(self.key_assignments)
            }


class DedicatedKeyClient:
    """
    Client sử dụng 1 key độc quyền với rate limiting
    Rate limit: 15 requests/minute = 1 request every 4 seconds
    """
    
    def __init__(self, task_id=None):
        self.manager = DedicatedKeyManager()
        self.task_id = task_id or f"task_{id(self)}"
        self.key_index, self.api_key = self.manager.assign_key(self.task_id)
        self.genai_client = genai.Client(api_key=self.api_key)
        
        # Rate limiting: 15 RPM = 4 seconds between requests
        self.min_delay_between_requests = 4.0
        self.last_request_time = 0
        
        logger.info(f"🎯 [CLIENT] Initialized for task {self.task_id[:8]} with key #{self.key_index + 1}")
    
    def call_with_retry(self, prompt, max_retries=5, model="models/gemini-2.5-flash-lite", **config):
        """
        Call API với dedicated key và rate limiting
        
        Args:
            prompt: Text prompt
            max_retries: Max retry attempts
            model: Model name (gemini-2.5-flash-lite hoặc gemini-2.5-flash-lite-preview)
            **config: Additional config
        """
        for attempt in range(max_retries):
            try:
                # Rate limiting: Đảm bảo ít nhất 4 giây giữa các requests
                current_time = time.time()
                time_since_last = current_time - self.last_request_time
                
                if time_since_last < self.min_delay_between_requests:
                    sleep_time = self.min_delay_between_requests - time_since_last
                    logger.debug(f"⏳ [CLIENT #{self.key_index + 1}] Rate limiting: sleeping {sleep_time:.1f}s")
                    time.sleep(sleep_time)
                
                self.last_request_time = time.time()
                
                # Make API call
                response_obj = self.genai_client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config if config else {"temperature": 0}
                )
                
                # Handle response safely
                if response_obj and hasattr(response_obj, "text") and response_obj.text:
                    return response_obj.text.strip()
                elif response_obj and hasattr(response_obj, "candidates") and response_obj.candidates:
                    candidate = response_obj.candidates[0]
                    if hasattr(candidate, "finish_reason"):
                        logger.warning(f"⚠️ [CLIENT #{self.key_index + 1}] Response blocked: {candidate.finish_reason}")
                        return ""
                else:
                    logger.warning(f"⚠️ [CLIENT #{self.key_index + 1}] Empty response")
                    return ""
                
            except Exception as e:
                error_str = str(e)
                
                # 1. Rate limit - wait longer
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    wait_time = 10 + (attempt * 5)  # Exponential backoff: 10, 15, 20, 25, 30s
                    logger.warning(f"⚠️ [CLIENT #{self.key_index + 1}] Rate limit hit. "
                                 f"Waiting {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                
                # 2. Server errors
                elif "500" in error_str or "503" in error_str or "INTERNAL" in error_str:
                    logger.warning(f"⚠️ [CLIENT #{self.key_index + 1}] Server error. Retrying in 5s...")
                    time.sleep(5.0)
                    continue
                
                # 3. Network errors
                elif "disconnected" in error_str.lower() or "connection" in error_str.lower():
                    logger.warning(f"⚠️ [CLIENT #{self.key_index + 1}] Network error. Retrying in 3s...")
                    time.sleep(3.0)
                    continue
                
                # 4. Other errors - raise
                else:
                    logger.error(f"❌ [CLIENT #{self.key_index + 1}] Unexpected error: {error_str}")
                    raise
        
        raise Exception(f"Max retries ({max_retries}) exceeded for key #{self.key_index + 1}")
    
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
