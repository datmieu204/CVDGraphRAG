import numpy as np
import os
import time
from google import genai
from transformers import AutoTokenizer, AutoModel
import torch

from ._utils import compute_args_hash, wrap_embedding_func_with_attrs
from .base import BaseKVStorage


# Global API key rotation state
_gemini_api_keys = []
_current_key_index = 0
_disabled_key_indices = set()  # Track exhausted keys
_genai_client = None

def _init_gemini_keys():
    """Initialize Gemini API keys for rotation"""
    global _gemini_api_keys, _genai_client, _current_key_index
    
    if not _gemini_api_keys:
        # Dynamically load all available API keys
        i = 1
        while True:
            key = os.getenv(f"GEMINI_API_KEY_{i}")
            if key:
                _gemini_api_keys.append(key)
                i += 1
            else:
                break
        
        if not _gemini_api_keys:
            raise ValueError("No GEMINI_API_KEY found in environment")
        
        _current_key_index = 0
        _genai_client = genai.Client(api_key=_gemini_api_keys[0])
        print(f"🔑 Loaded {len(_gemini_api_keys)} Gemini API keys (nano_graphrag)")
    
    return _genai_client

def _get_active_keys_count():
    """Get number of keys still available (not disabled)"""
    return len(_gemini_api_keys) - len(_disabled_key_indices)

def _disable_current_key():
    """Mark current key as disabled (exhausted)"""
    global _current_key_index, _disabled_key_indices
    if _current_key_index not in _disabled_key_indices:
        _disabled_key_indices.add(_current_key_index)
        print(f"🚫 Key #{_current_key_index + 1} disabled. Active: {_get_active_keys_count()}/{len(_gemini_api_keys)}")

def _rotate_gemini_key():
    """Rotate to next available API key, skipping disabled ones"""
    global _current_key_index, _genai_client, _gemini_api_keys
    
    attempts = 0
    max_attempts = len(_gemini_api_keys)
    
    while attempts < max_attempts:
        _current_key_index = (_current_key_index + 1) % len(_gemini_api_keys)
        attempts += 1
        
        if _current_key_index not in _disabled_key_indices:
            _genai_client = genai.Client(api_key=_gemini_api_keys[_current_key_index])
            print(f"🔄 Rotated to key #{_current_key_index + 1}/{len(_gemini_api_keys)} (Active: {_get_active_keys_count()})")
            return True
    
    print(f"❌ All {len(_gemini_api_keys)} keys disabled")
    return False


async def gemini_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """Use Gemini with API key rotation and rate limiting"""
    global _genai_client
    
    client = _init_gemini_keys()
    
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    
    # Build messages for cache key
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    
    # Combine prompts for Gemini
    full_prompt = ""
    if system_prompt:
        full_prompt += f"{system_prompt}\n\n"
    for msg in history_messages:
        full_prompt += f"{msg.get('role', 'user')}: {msg.get('content', '')}\n"
    full_prompt += prompt
    
    # Retry with key rotation
    max_retries = _get_active_keys_count() * 2
    
    if _get_active_keys_count() == 0:
        raise Exception(f"All {len(_gemini_api_keys)} API keys disabled.")
    
    for attempt in range(max_retries):
        try:
            # Rate limiting delay
            time.sleep(1.0)
            
            response_obj = client.models.generate_content(
                model=model,
                contents=full_prompt,
                config=kwargs
            )
            
            response_text = response_obj.text.strip() if response_obj and hasattr(response_obj, "text") else ""
            
            if hashing_kv is not None:
                await hashing_kv.upsert(
                    {args_hash: {"return": response_text, "model": model}}
                )
            return response_text
            
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                print(f"⚠️ Rate limit on key #{_current_key_index + 1}")
                
                _disable_current_key()
                
                if not _rotate_gemini_key():
                    raise Exception(f"All {len(_gemini_api_keys)} API keys exhausted.")
                
                client = _genai_client
                time.sleep(2.0)
                continue
            else:
                raise
    
    return ""


async def gemini_1_5_flash_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await gemini_complete_if_cache(
        "models/gemini-2.5-flash-lite",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


# Initialize embedding model globally
_bge_tokenizer = None
_bge_model = None

def get_bge_model():
    """Initialize and return bge-small-en-v1.5 model"""
    global _bge_tokenizer, _bge_model
    
    if _bge_tokenizer is None or _bge_model is None:
        hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
        _bge_tokenizer = AutoTokenizer.from_pretrained(
            "BAAI/bge-small-en-v1.5", 
            token=hf_token
        )
        _bge_model = AutoModel.from_pretrained(
            "BAAI/bge-small-en-v1.5",
            token=hf_token
        )
        _bge_model.eval()
    
    return _bge_tokenizer, _bge_model


@wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=512)
async def bge_m3_embedding(texts: list[str]) -> np.ndarray:
    """Use HuggingFace bge-small-en-v1.5 for embeddings"""
    tokenizer, model = get_bge_model()
    
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Use mean pooling
            embedding = outputs.last_hidden_state.mean(dim=1)
        
        embeddings.append(embedding[0].cpu().numpy())
    
    return np.array(embeddings)
