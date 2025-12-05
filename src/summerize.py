import time
from concurrent.futures import ThreadPoolExecutor
import tiktoken
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

sum_prompt = """
Generate a structured summary from the provided medical source (report, paper, or book), strictly adhering to the following categories. The summary should list key information under each category in a concise format: 'CATEGORY_NAME: Key information'. No additional explanations or detailed descriptions are necessary unless directly related to the categories:

ANATOMICAL_STRUCTURE: Mention any anatomical structures specifically discussed.
BODY_FUNCTION: List any body functions highlighted.
BODY_MEASUREMENT: Include normal measurements like blood pressure or temperature.
BM_RESULT: Results of these measurements.
BM_UNIT: Units for each measurement.
BM_VALUE: Values of these measurements.
LABORATORY_DATA: Outline any laboratory tests mentioned.
LAB_RESULT: Outcomes of these tests (e.g., 'increased', 'decreased').
LAB_VALUE: Specific values from the tests.
LAB_UNIT: Units of measurement for these values.
MEDICINE: Name medications discussed.
MED_DOSE, MED_DURATION, MED_FORM, MED_FREQUENCY, MED_ROUTE, MED_STATUS, MED_STRENGTH, MED_UNIT, MED_TOTALDOSE: Provide concise details for each medication attribute.
PROBLEM: Identify any medical conditions or findings.
PROCEDURE: Describe any procedures.
PROCEDURE_RESULT: Outcomes of these procedures.
PROC_METHOD: Methods used.
SEVERITY: Severity of the conditions mentioned.
MEDICAL_DEVICE: List any medical devices used.
SUBSTANCE_ABUSE: Note any substance abuse mentioned.
Each category should be addressed only if relevant to the content of the medical source. Ensure the summary is clear and direct, suitable for quick reference.
"""

# Note: summerize.py is used by utils.py, run.py, and qna_evaluator.py
# These are for query/evaluation, not import. They can use shared manager safely.
# But for consistency, we'll update to support optional client parameter

def call_openai_api(chunk, client=None):
    """Call Gemini API with optional dedicated client
    
    Args:
        chunk: Text to summarize
        client: DedicatedKeyClient (REQUIRED). Should be passed from process_chunks().
    """
    if client is None:
        raise ValueError("Client must be provided to call_openai_api()")
    
    full_prompt = f"{sum_prompt}\n\n{chunk}"
    return client.call_with_retry(
        full_prompt, 
        model="models/gemini-2.5-flash-lite",
        max_retries=3
    )

def split_into_chunks(text, tokens=500):
    encoding = tiktoken.encoding_for_model('gpt-4-1106-preview')
    words = encoding.encode(text)
    chunks = []
    for i in range(0, len(words), tokens):
        chunks.append(' '.join(encoding.decode(words[i:i + tokens])))
    return chunks   

def process_chunks(content, client=None):
    """Process content chunks with optional dedicated client
    
    Args:
        content: Text to process
        client: Optional DedicatedKeyClient. If None, creates ONE shared client for all chunks.
    """
    from dedicated_key_manager import create_dedicated_client
    
    chunks = split_into_chunks(content)
    
    if client is None:
        client = create_dedicated_client(task_id="summerize_standalone")

    from functools import partial
    call_with_client = partial(call_openai_api, client=client)
    
    with ThreadPoolExecutor() as executor:
        responses = list(executor.map(call_with_client, chunks))
    return responses


if __name__ == "__main__":
    content = " sth you wanna test"
    process_chunks(content)