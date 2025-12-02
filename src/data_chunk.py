from pydantic import BaseModel
from agentic_chunker import AgenticChunker
from typing import List

from logger_ import get_logger

logger = get_logger("data_chunk", log_file="logs/data_chunk.log")

class Sentences(BaseModel):
    sentences: List[str]


def run_chunk(essay, client=None):
    """
    Run chunking with optional dedicated client
    
    Args:
        essay: Text to chunk
        client: DedicatedKeyClient instance (if None, will create one)
    """
    if client is None:
        from dedicated_key_manager import create_dedicated_client
        client = create_dedicated_client(task_id="data_chunk_fallback")
        logger.warning("No client provided, created fallback client")

    paragraphs = essay.split("\n\n")
    essay_propositions = []

    for i, para in enumerate(paragraphs):
        try:
            prompt = f"""
            Extract main propositions from the following text. 
            Return strictly in JSON format with a "sentences" field containing an array of strings.

            Text:
            {para}
            
            Example format:
            {{"sentences": ["proposition 1", "proposition 2", ...]}}
            """
            
            response_text = client.call_with_retry(prompt, model="models/gemini-2.5-flash-lite")
            
            import json
            try:
                result = json.loads(response_text)
                if isinstance(result, dict) and "sentences" in result:
                    essay_propositions.extend(result["sentences"])
                else:
                    sentences = [s.strip() for s in response_text.split('\n') if s.strip()]
                    essay_propositions.extend(sentences)
            except json.JSONDecodeError:
                sentences = [s.strip() for s in response_text.split('\n') if s.strip()]
                essay_propositions.extend(sentences)

            logger.info(f"Done paragraph {i+1}")
        except Exception as e:
            logger.warning(f"Skipped paragraph {i+1}: {e}")

    ac = AgenticChunker(client=client)
    ac.add_propositions(essay_propositions)
    ac.pretty_print_chunks()
    chunks = ac.get_chunks(get_type="list_of_strings")

    return chunks
