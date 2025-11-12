import time
import os
from neo4j import GraphDatabase
import numpy as np
from camel.storages import Neo4jGraph
import uuid
from summerize import process_chunks
from transformers import AutoTokenizer, AutoModel
import torch
from google import genai

sys_prompt_one = """
Please answer the question using insights supported by provided graph-based data relevant to medical information.
"""

sys_prompt_two = """
Modify the response to the question using the provided references. Include precise citations relevant to your answer. You may use multiple citations simultaneously, denoting each with the reference index number. For example, cite the first and third documents as [1][3]. If the references do not pertain to the response, simply provide a concise answer to the original question.
"""

# Initialize HuggingFace bge-small-en-v1.5 model for embeddings
_embedding_tokenizer = None
_embedding_model = None

def get_bge_m3_embedding(text):
    """Get embeddings using HuggingFace bge-small-en-v1.5 model"""
    global _embedding_tokenizer, _embedding_model
    
    if _embedding_tokenizer is None or _embedding_model is None:
        hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
        _embedding_tokenizer = AutoTokenizer.from_pretrained(
            "BAAI/bge-small-en-v1.5", 
            token=hf_token
        )
        _embedding_model = AutoModel.from_pretrained(
            "BAAI/bge-small-en-v1.5",
            token=hf_token
        )
        _embedding_model.eval()
    
    # Tokenize and get embeddings
    inputs = _embedding_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = _embedding_model(**inputs)
        # Use mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)
    
    return embeddings[0].cpu().numpy().tolist()

def get_embedding(text, mod="bge-small-en-v1.5"):
    """Get embeddings using bge-small-en-v1.5 from HuggingFace"""
    return get_bge_m3_embedding(text)

def fetch_texts(n4j):
    # Fetch the text for each node
    query = "MATCH (n) RETURN n.id AS id"
    return n4j.query(query)

def add_embeddings(n4j, node_id, embedding):
    # Upload embeddings to Neo4j
    query = "MATCH (n) WHERE n.id = $node_id SET n.embedding = $embedding"
    n4j.query(query, params = {"node_id":node_id, "embedding":embedding})

def add_nodes_emb(n4j):
    nodes = fetch_texts(n4j)

    for node in nodes:
        # Calculate embedding for each node's text
        if node['id']:  # Ensure there is text to process
            embedding = get_embedding(node['id'])
            # Store embedding back in the node
            add_embeddings(n4j, node['id'], embedding)

def add_ge_emb(graph_element):
    for node in graph_element.nodes:
        emb = get_embedding(node.id)
        node.properties['embedding'] = emb
    return graph_element

def add_gid(graph_element, gid):
    for node in graph_element.nodes:
        node.properties['gid'] = gid
    for rel in graph_element.relationships:
        rel.properties['gid'] = gid
    return graph_element

def add_sum(n4j,content,gid):
    sum = process_chunks(content)
    creat_sum_query = """
        CREATE (s:Summary {content: $sum, gid: $gid})
        RETURN s
        """
    s = n4j.query(creat_sum_query, {'sum': sum, 'gid': gid})
    
    link_sum_query = """
        MATCH (s:Summary {gid: $gid}), (n)
        WHERE n.gid = s.gid AND NOT n:Summary
        CREATE (s)-[:SUMMARIZES]->(n)
        RETURN s, n
        """
    n4j.query(link_sum_query, {'gid': gid})

    return s

# Global API key rotator for call_llm
_gemini_rotator = None

def _get_gemini_rotator():
    """Get or create global Gemini API key rotator"""
    global _gemini_rotator
    if _gemini_rotator is None:
        _gemini_rotator = GeminiClientRotator()
    return _gemini_rotator


class GeminiClientRotator:
    """Rotator class for multiple Gemini API keys with blacklist support for exhausted keys"""
    def __init__(self, print_logging=True):
        self.api_keys = []
        # Dynamically load all available API keys
        i = 1
        while True:
            key = os.getenv(f"GEMINI_API_KEY_{i}")
            if key:
                self.api_keys.append(key)
                i += 1
            else:
                break
        
        # Filter out None values
        self.api_keys = [key for key in self.api_keys if key]
        self.current_key_index = 0
        self.disabled_key_indices = set()  # Track disabled keys
        self.key_429_count = {}  # Track 429 errors per key
        self.print_logging = print_logging
        
        # Initialize 429 counters
        for idx in range(len(self.api_keys)):
            self.key_429_count[idx] = 0
        
        if not self.api_keys:
            raise ValueError("No GEMINI_API_KEY found in environment variables")
        
        if self.print_logging:
            print(f"🔑 Loaded {len(self.api_keys)} Gemini API keys")
        
        self.genai_client = genai.Client(api_key=self.api_keys[0])
    
    def _get_active_keys_count(self):
        """Get number of keys still available (not disabled)"""
        return len(self.api_keys) - len(self.disabled_key_indices)
    
    def reset(self):
        """Reset all disabled keys and 429 counters"""
        self.disabled_key_indices.clear()
        self.key_429_count = {i: 0 for i in range(len(self.api_keys))}
        self.current_key_index = 0
        self.genai_client = genai.Client(api_key=self.api_keys[0])
        if self.print_logging:
            print(f"🔄 Reset all keys. Active: {self._get_active_keys_count()}/{len(self.api_keys)}")
    
    def _disable_current_key(self):
        """Mark current key as disabled (exhausted) - only when truly out of quota"""
        if self.current_key_index not in self.disabled_key_indices:
            self.disabled_key_indices.add(self.current_key_index)
            if self.print_logging:
                print(f"🚫 Key #{self.current_key_index + 1} PERMANENTLY disabled (quota exhausted). "
                      f"Active: {self._get_active_keys_count()}/{len(self.api_keys)}")
    
    def _handle_429_error(self):
        """
        Smart 429 handling:
        - First 429: rotate only (might be temporary rate limit)
        - Second+ 429: disable permanently (truly exhausted)
        """
        self.key_429_count[self.current_key_index] = self.key_429_count.get(self.current_key_index, 0) + 1
        
        if self.key_429_count[self.current_key_index] == 1:
            # First 429 - just rotate
            if self.print_logging:
                print(f"⚠️ Key #{self.current_key_index + 1} hit rate limit (first time). Rotating to next key...")
            return self.rotate_key()
        else:
            # Second+ 429 - disable permanently
            if self.print_logging:
                print(f"⚠️ Key #{self.current_key_index + 1} hit rate limit again "
                      f"(#{self.key_429_count[self.current_key_index]} times). Disabling permanently...")
            self._disable_current_key()
            return self.rotate_key()
    
    def rotate_key(self):
        """Rotate to next available API key, skipping disabled ones"""
        initial_index = self.current_key_index
        attempts = 0
        max_attempts = len(self.api_keys)
        
        while attempts < max_attempts:
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            attempts += 1
            
            if self.current_key_index not in self.disabled_key_indices:
                self.genai_client = genai.Client(api_key=self.api_keys[self.current_key_index])
                if self.print_logging:
                    print(f"🔄 Rotated to key #{self.current_key_index + 1}/{len(self.api_keys)} (Active: {self._get_active_keys_count()})")
                return True
        
        if self.print_logging:
            print(f"❌ All {len(self.api_keys)} keys disabled")
        return False
    
    def call_with_retry(self, prompt, max_output_tokens=500, temperature=0.5, max_retries=None):
        """
        Call Gemini API with automatic retry and key rotation
        FIXED: Rotate key BEFORE each request để phân phối đều tải
        """
        if max_retries is None:
            max_retries = self._get_active_keys_count() * 2
        
        if self._get_active_keys_count() == 0:
            raise Exception(f"All {len(self.api_keys)} API keys disabled. No active keys.")
        
        for attempt in range(max_retries):
            try:
                # CRITICAL FIX: Rotate to next available key BEFORE each request
                # This ensures true round-robin distribution
                if attempt > 0:  # Don't rotate on first attempt
                    if not self.rotate_key():
                        raise Exception(f"All {len(self.api_keys)} API keys disabled.")
                
                # Rate limiting delay
                time.sleep(1.0)
                
                response_obj = self.genai_client.models.generate_content(
                    model="models/gemini-2.5-flash-lite",
                    contents=prompt,
                    config={
                        "max_output_tokens": max_output_tokens,
                        "temperature": temperature,
                    }
                )
                
                # Success - reset 429 counter for this key
                if self.current_key_index in self.key_429_count:
                    self.key_429_count[self.current_key_index] = 0
                
                # Rotate to next key after successful request (round-robin)
                self.rotate_key()
                
                # Safe response handling
                if response_obj and hasattr(response_obj, "text") and response_obj.text:
                    return response_obj.text.strip()
                elif response_obj and hasattr(response_obj, "candidates") and response_obj.candidates:
                    # Check if content was blocked by safety filters
                    candidate = response_obj.candidates[0]
                    if hasattr(candidate, "finish_reason"):
                        if self.print_logging:
                            print(f"⚠️ Response blocked by safety filters: {candidate.finish_reason}")
                        return ""
                else:
                    if self.print_logging:
                        print(f"⚠️ Empty response from key #{self.current_key_index + 1}")
                    return ""
                    
            except Exception as e:
                error_str = str(e)
                
                # 1. Rate limit / Quota exhausted - mark and continue
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                    self.key_429_count[self.current_key_index] = self.key_429_count.get(self.current_key_index, 0) + 1
                    
                    if self.key_429_count[self.current_key_index] >= 2:
                        # Second+ 429 on same key = truly exhausted
                        if self.print_logging:
                            print(f"⚠️ Key #{self.current_key_index + 1} hit 429 multiple times. Disabling...")
                        self._disable_current_key()
                    else:
                        # First 429 = might be temporary
                        if self.print_logging:
                            print(f"⚠️ Key #{self.current_key_index + 1} hit 429 (first time). Marked.")
                    
                    # Continue to next iteration (will rotate at start)
                    time.sleep(2.0)
                    continue
                
                # 2. Server errors (500, 503) - retry without disabling key
                elif "500" in error_str or "503" in error_str or "INTERNAL" in error_str:
                    if self.print_logging:
                        print(f"⚠️ Server error on key #{self.current_key_index + 1}: {error_str[:100]}")
                        print(f"   Retrying in 5 seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(5.0)
                    continue
                
                # 3. Network errors - retry
                elif "disconnected" in error_str.lower() or "connection" in error_str.lower():
                    if self.print_logging:
                        print(f"⚠️ Network error: {error_str[:100]}")
                        print(f"   Retrying in 3 seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(3.0)
                    continue
                
                # 4. Other errors - raise immediately (don't rotate)
                else:
                    if self.print_logging:
                        print(f"❌ Unexpected error on key #{self.current_key_index + 1}: {error_str}")
                        print(f"   This is likely a code bug or API issue, not a quota problem.")
                    raise
        
        # Max retries exceeded
        raise Exception(f"Max retries ({max_retries}) exceeded. Last active keys: {self._get_active_keys_count()}")


def call_llm(sys, user):
    """Call Gemini-2.5-flash-lite LLM with API key rotation"""
    rotator = _get_gemini_rotator()
    
    # Combine system and user prompts for Gemini
    full_prompt = f"{sys}\n\n{user}"
    
    return rotator.call_with_retry(full_prompt, max_output_tokens=500, temperature=0.5)

def find_index_of_largest(nums):
    # Handle empty list
    if not nums:
        print("⚠️ Warning: No ratings found. Database may be empty.")
        return -1
    
    # Sorting the list while keeping track of the original indexes
    sorted_with_index = sorted((num, index) for index, num in enumerate(nums))
    
    # Extracting the original index of the largest element
    largest_original_index = sorted_with_index[-1][1]
    
    return largest_original_index

def get_response(n4j, gid, query):
    selfcont = ret_context(n4j, gid)
    linkcont = link_context(n4j, gid)
    user_one = "the question is: " + query + "the provided information is:" +  "".join(selfcont)
    res = call_llm(sys_prompt_one,user_one)
    user_two = "the question is: " + query + "the last response of it is:" +  res + "the references are: " +  "".join(linkcont)
    res = call_llm(sys_prompt_two,user_two)
    return res

def link_context(n4j, gid):
    cont = []
    retrieve_query = """
        // Match all 'n' nodes with a specific gid but not of the "Summary" type
        MATCH (n)
        WHERE n.gid = $gid AND NOT n:Summary

        // Find all 'm' nodes where 'm' is a reference of 'n' via a 'REFERENCES' relationship
        MATCH (n)-[r:REFERENCE]->(m)
        WHERE NOT m:Summary

        // Find all 'o' nodes connected to each 'm', and include the relationship type,
        // while excluding 'Summary' type nodes and 'REFERENCE' relationship
        MATCH (m)-[s]-(o)
        WHERE NOT o:Summary AND TYPE(s) <> 'REFERENCE'

        // Collect and return details in a structured format
        RETURN n.id AS NodeId1, 
            m.id AS Mid, 
            TYPE(r) AS ReferenceType, 
            collect(DISTINCT {RelationType: type(s), Oid: o.id}) AS Connections
    """
    res = n4j.query(retrieve_query, {'gid': gid})
    for r in res:
        # Expand each set of connections into separate entries with n and m
        for ind, connection in enumerate(r["Connections"]):
            cont.append("Reference " + str(ind) + ": " + r["NodeId1"] + "has the reference that" + r['Mid'] + connection['RelationType'] + connection['Oid'])
    return cont

def ret_context(n4j, gid):
    cont = []
    ret_query = """
    // Match all nodes with a specific gid but not of type "Summary" and collect them
    MATCH (n)
    WHERE n.gid = $gid AND NOT n:Summary
    WITH collect(n) AS nodes

    // Unwind the nodes to a pairs and match relationships between them
    UNWIND nodes AS n
    UNWIND nodes AS m
    MATCH (n)-[r]-(m)
    WHERE n.gid = m.gid AND id(n) < id(m) AND NOT n:Summary AND NOT m:Summary // Ensure each pair is processed once and exclude "Summary" nodes in relationships
    WITH n, m, TYPE(r) AS relType

    // Return node IDs and relationship types in structured format
    RETURN n.id AS NodeId1, relType, m.id AS NodeId2
    """
    res = n4j.query(ret_query, {'gid': gid})
    for r in res:
        cont.append(r['NodeId1'] + r['relType'] + r['NodeId2'])
    return cont

def merge_similar_nodes(n4j, gid):
    """
    Merge similar nodes based on embedding similarity using vector.similarity.cosine
    or manual cosine calculation if that's not available
    """
    if gid:
        # Try using vector.similarity.cosine first (Neo4j 5.x+)
        merge_query = """
            WITH 0.5 AS threshold
            MATCH (n), (m)
            WHERE NOT n:Summary AND NOT m:Summary 
                AND n.gid = m.gid AND n.gid = $gid 
                AND n<>m 
                AND apoc.coll.sort(labels(n)) = apoc.coll.sort(labels(m))
                AND n.embedding IS NOT NULL AND m.embedding IS NOT NULL
            WITH n, m, threshold,
                // Manual cosine similarity calculation
                reduce(dot = 0.0, i IN range(0, size(n.embedding)-1) | 
                    dot + n.embedding[i] * m.embedding[i]) / 
                (sqrt(reduce(norm1 = 0.0, x IN n.embedding | norm1 + x * x)) * 
                 sqrt(reduce(norm2 = 0.0, y IN m.embedding | norm2 + y * y))) AS similarity
            WHERE similarity > threshold
            WITH head(collect([n,m])) as nodes
            CALL apoc.refactor.mergeNodes(nodes, {properties: 'overwrite', mergeRels: true})
            YIELD node
            RETURN count(*) as merged_count
        """
        result = n4j.query(merge_query, {'gid': gid})
    else:
        merge_query = """
            WITH 0.5 AS threshold
            MATCH (n), (m)
            WHERE NOT n:Summary AND NOT m:Summary 
                AND n<>m 
                AND apoc.coll.sort(labels(n)) = apoc.coll.sort(labels(m))
                AND n.embedding IS NOT NULL AND m.embedding IS NOT NULL
            WITH n, m, threshold,
                // Manual cosine similarity calculation
                reduce(dot = 0.0, i IN range(0, size(n.embedding)-1) | 
                    dot + n.embedding[i] * m.embedding[i]) / 
                (sqrt(reduce(norm1 = 0.0, x IN n.embedding | norm1 + x * x)) * 
                 sqrt(reduce(norm2 = 0.0, y IN m.embedding | norm2 + y * y))) AS similarity
            WHERE similarity > threshold
            WITH head(collect([n,m])) as nodes
            CALL apoc.refactor.mergeNodes(nodes, {properties: 'overwrite', mergeRels: true})
            YIELD node
            RETURN count(*) as merged_count
        """
        result = n4j.query(merge_query)
    return result

def ref_link(n4j, gid1, gid2):
    """
    Create reference links between similar nodes from different graphs
    using manual cosine similarity calculation
    """
    trinity_query = """
        // Match nodes from Graph A
        MATCH (a)
        WHERE a.gid = $gid1 AND NOT a:Summary AND a.embedding IS NOT NULL
        WITH collect(a) AS GraphA

        // Match nodes from Graph B
        MATCH (b)
        WHERE b.gid = $gid2 AND NOT b:Summary AND b.embedding IS NOT NULL
        WITH GraphA, collect(b) AS GraphB

        // Unwind the nodes to compare each against each
        UNWIND GraphA AS n
        UNWIND GraphB AS m

        // Set the threshold for cosine similarity
        WITH n, m, 0.6 AS threshold

        // Compute cosine similarity and apply the threshold
        WHERE apoc.coll.sort(labels(n)) = apoc.coll.sort(labels(m)) AND n <> m
        WITH n, m, threshold,
            // Manual cosine similarity calculation
            reduce(dot = 0.0, i IN range(0, size(n.embedding)-1) | 
                dot + n.embedding[i] * m.embedding[i]) / 
            (sqrt(reduce(norm1 = 0.0, x IN n.embedding | norm1 + x * x)) * 
             sqrt(reduce(norm2 = 0.0, y IN m.embedding | norm2 + y * y))) AS similarity
        WHERE similarity > threshold

        // Create a relationship based on the condition
        MERGE (m)-[:REFERENCE]->(n)

        // Return results
        RETURN n, m, similarity
"""
    result = n4j.query(trinity_query, {'gid1': gid1, 'gid2': gid2})
    return result


def str_uuid():
    # Generate a random UUID
    generated_uuid = uuid.uuid4()

    # Convert UUID to a string
    return str(generated_uuid)


