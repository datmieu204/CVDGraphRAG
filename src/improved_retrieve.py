"""
Improved U-Retrieval Module
===========================
Enhancements:
1. Hybrid Retrieval: Vector search + LLM Reranking
2. Multi-subgraph aggregation
3. Query-aware context ranking
4. Entity-guided retrieval (optional)

Author: CVDGraphRAG Team
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from utils import get_embedding, get_bge_m3_embedding, find_index_of_largest
from dedicated_key_manager import create_dedicated_client
from logger_ import get_logger

logger = get_logger("improved_retrieve", log_file="logs/improved_retrieve.log")

def vector_search_summaries(n4j, query: str, top_n: int = 20) -> List[Dict]:
    """
    Fast vector-based pre-filtering using BGE embeddings
    
    Args:
        n4j: Neo4j connection
        query: User query
        top_n: Number of candidates to retrieve
    
    Returns:
        List of {gid, content, similarity} sorted by similarity
    """
    logger.info(f"[Vector Search] Finding top {top_n} candidates...")
    
    query_embedding = get_bge_m3_embedding(query)
    
    check_query = """
        MATCH (s:Summary)
        RETURN 
            count(*) AS total,
            sum(CASE WHEN s.embedding IS NOT NULL THEN 1 ELSE 0 END) AS with_emb
    """
    check_result = n4j.query(check_query)
    
    if check_result:
        total = check_result[0]['total']
        with_emb = check_result[0]['with_emb']
        logger.info(f"[Vector Search] {with_emb}/{total} summaries have pre-computed embeddings")
    
    if check_result and check_result[0]['with_emb'] > 0:
        sum_query = """
            MATCH (s:Summary)
            WHERE s.embedding IS NOT NULL
            RETURN s.content AS content, s.gid AS gid, s.embedding AS embedding
        """
        results = n4j.query(sum_query)
        use_precomputed = True
    else:
        logger.warning("No pre-computed embeddings! Run: python add_summary_embeddings.py")
        sum_query = """
            MATCH (s:Summary)
            RETURN s.content AS content, s.gid AS gid
        """
        results = n4j.query(sum_query)
        use_precomputed = False
    
    if not results:
        logger.error("No Summary nodes found in database")
        return []
    
    logger.info(f"[Vector Search] Processing {len(results)} summaries...")
    
    candidates = []
    for r in results:
        try:
            # Get embedding
            if use_precomputed and 'embedding' in r and r['embedding'] is not None:
                emb = r['embedding']
            else:
                # Compute on-the-fly (slow)
                content = r['content']
                if isinstance(content, list):
                    content = content[0] if content else ""
                content = content[:1000] if len(content) > 1000 else content
                emb = get_bge_m3_embedding(content)
            
            # Cosine similarity
            dot_product = np.dot(query_embedding, emb)
            norm_product = np.linalg.norm(query_embedding) * np.linalg.norm(emb)
            similarity = dot_product / (norm_product + 1e-8)
            
            content_str = r['content']
            if isinstance(content_str, list):
                content_str = content_str[0] if content_str else ""
            
            candidates.append({
                'gid': r['gid'],
                'content': content_str,
                'similarity': float(similarity)
            })
        except Exception as e:
            logger.debug(f"Skip candidate: {e}")
            continue
    
    candidates.sort(key=lambda x: x['similarity'], reverse=True)
    
    logger.info(f"[Vector Search] Found {len(candidates)} candidates")
    if candidates:
        logger.info(f"  Top similarity: {candidates[0]['similarity']:.3f}")
        if len(candidates) >= top_n:
            logger.info(f"  #{top_n} similarity: {candidates[top_n-1]['similarity']:.3f}")
    
    return candidates[:top_n]

def llm_rerank(candidates: List[Dict], query: str, client, top_k: int = 5) -> List[str]:
    """
    LLM-based reranking of top candidates
    
    Args:
        candidates: List from vector_search_summaries
        query: User query
        client: DedicatedKeyClient
        top_k: Final number of GIDs to return
    
    Returns:
        List of GIDs (best matches first)
    """
    if len(candidates) <= top_k:
        return [c['gid'] for c in candidates]
    
    logger.info(f"[LLM Rerank] Reranking {len(candidates)} candidates to top {top_k}...")
    
    candidate_texts = []
    for i, c in enumerate(candidates[:min(10, len(candidates))]):  # Max 10 for reranking
        summary_preview = c['content'][:300] + "..." if len(c['content']) > 300 else c['content']
        candidate_texts.append(f"{i+1}. {summary_preview}")
    
    rerank_prompt = f"""Given the query and candidate summaries, rank them by relevance.

Query: {query}

Candidates:
{chr(10).join(candidate_texts)}

Return ONLY the numbers in order of relevance (most relevant first), separated by commas.
Example: 3,1,5,2,4

Your ranking:"""

    try:
        response = client.call_with_retry(
            rerank_prompt, 
            model="models/gemini-2.5-flash-lite",
            max_retries=3
        )
        
        numbers = [int(n.strip()) for n in response.split(',') if n.strip().isdigit()]
        
        ranked_gids = []
        for idx in numbers:
            if 1 <= idx <= len(candidates):
                gid = candidates[idx-1]['gid']
                if gid not in ranked_gids:
                    ranked_gids.append(gid)
        
        for c in candidates:
            if c['gid'] not in ranked_gids:
                ranked_gids.append(c['gid'])
        
        logger.info(f"[LLM Rerank] Reranked successfully")
        return ranked_gids[:top_k]
        
    except Exception as e:
        logger.warning(f"[LLM Rerank] Failed: {e}, using vector order")
        return [c['gid'] for c in candidates[:top_k]]

def hybrid_retrieve(n4j, query: str, client=None, top_k: int = 3, 
                    vector_candidates: int = 20) -> List[str]:
    """
    Hybrid Retrieval: Vector search + LLM reranking
    
    Args:
        n4j: Neo4j connection
        query: User query
        client: Optional DedicatedKeyClient
        top_k: Number of final GIDs to return
        vector_candidates: Number of vector search candidates
    
    Returns:
        List of top_k GIDs
    """
    if client is None:
        client = create_dedicated_client(task_id="hybrid_retrieve")
    
    candidates = vector_search_summaries(n4j, query, top_n=vector_candidates)
    
    if not candidates:
        logger.error("[Hybrid Retrieve] No candidates found")
        return []
    
    if len(candidates) <= top_k:
        logger.info(f"[Hybrid Retrieve] Only {len(candidates)} candidates, skipping rerank")
        return [c['gid'] for c in candidates]
    
    ranked_gids = llm_rerank(candidates, query, client, top_k)
    
    logger.info(f"[Hybrid Retrieve] Final GIDs: {ranked_gids}")
    return ranked_gids

def get_ranked_context(n4j, gid: str, query: str, max_items: int = 50) -> List[str]:
    """
    Get context triples ranked by relevance to query
    
    Args:
        n4j: Neo4j connection
        gid: Graph ID
        query: User query
        max_items: Maximum context items to return
    
    Returns:
        List of context strings, sorted by relevance
    """
    logger.info(f"[Ranked Context] GID: {gid[:8]}..., max_items: {max_items}")
    
    MAX_TRIPLES = 1000
    
    ret_query = """
        MATCH (n)-[r]-(m)
        WHERE n.gid = $gid AND NOT n:Summary AND NOT m:Summary
          AND id(n) < id(m)
        RETURN n.id AS n_id, TYPE(r) AS rel_type, m.id AS m_id
        LIMIT $max_triples
    """
    
    results = n4j.query(ret_query, {'gid': gid, 'max_triples': MAX_TRIPLES})
    
    if not results:
        logger.warning(f"[Ranked Context] No triples found for GID: {gid[:8]}")
        return []
    
    logger.info(f"[Ranked Context] Retrieved {len(results)} triples (limit: {MAX_TRIPLES})")
    
    query_terms = set(query.lower().split())
    
    scored_triples = []
    for r in results:
        triple_str = f"{r['n_id']} {r['rel_type']} {r['m_id']}"
        triple_lower = triple_str.lower()
        
        matches = sum(1 for term in query_terms if term in triple_lower)
        relevance = matches / max(len(query_terms), 1)
        
        scored_triples.append((triple_str, relevance))
    
    scored_triples.sort(key=lambda x: x[1], reverse=True)
    
    logger.info(f"[Ranked Context] Scored {len(scored_triples)} triples")
    if scored_triples and scored_triples[0][1] > 0:
        logger.info(f"  Top relevance: {scored_triples[0][1]:.3f}")
    
    return [t[0] for t in scored_triples[:max_items]]

def get_ranked_link_context(n4j, gid: str, query: str, max_items: int = 50) -> List[str]:
    """
    Get link context (references) ranked by relevance to query
    
    Args:
        n4j: Neo4j connection
        gid: Graph ID
        query: User query
        max_items: Maximum context items to return
    
    Returns:
        List of reference strings, sorted by relevance
    """
    logger.info(f"[Ranked Link Context] GID: {gid[:8]}...")
    
    MAX_REFS = 500
    
    retrieve_query = """
        MATCH (n)
        WHERE n.gid = $gid AND NOT n:Summary
        MATCH (n)-[r:REFERENCE]->(m)
        WHERE NOT m:Summary
        MATCH (m)-[s]-(o)
        WHERE NOT o:Summary AND TYPE(s) <> 'REFERENCE'
        RETURN n.id AS n_id, m.id AS m_id, TYPE(s) AS rel_type, o.id AS o_id
        LIMIT $max_refs
    """
    
    results = n4j.query(retrieve_query, {'gid': gid, 'max_refs': MAX_REFS})
    
    if not results:
        logger.warning(f"[Ranked Link Context] No references found for GID: {gid[:8]}")
        return []
    
    logger.info(f"[Ranked Link Context] Retrieved {len(results)} references (limit: {MAX_REFS})")
    
    query_lower = query.lower()
    query_terms = set(query_lower.split())
    
    scored_refs = []
    for r in results:
        ref_str = f"Reference: {r['n_id']} has reference that {r['m_id']} {r['rel_type']} {r['o_id']}"
        ref_lower = ref_str.lower()
        
        matches = sum(1 for term in query_terms if term in ref_lower)
        relevance = matches / max(len(query_terms), 1)
        
        scored_refs.append((ref_str, relevance))
    
    scored_refs.sort(key=lambda x: x[1], reverse=True)
    
    seen = set()
    unique_refs = []
    for ref, score in scored_refs:
        if ref not in seen:
            seen.add(ref)
            unique_refs.append(ref)
            if len(unique_refs) >= max_items:
                break
    
    logger.info(f"[Ranked Link Context] Found {len(unique_refs)} unique references")
    return unique_refs

def aggregate_multi_subgraph_context(n4j, gids: List[str], query: str, 
                                      max_items: int = 100) -> Tuple[List[str], List[str]]:
    """
    Aggregate context from multiple subgraphs
    
    Args:
        n4j: Neo4j connection
        gids: List of Graph IDs
        query: User query
        max_items: Maximum total context items
    
    Returns:
        Tuple of (self_context, link_context)
    """
    logger.info(f"[Multi-Subgraph] Aggregating from {len(gids)} subgraphs...")
    
    all_self_context = []
    all_link_context = []
    
    items_per_subgraph = max_items // len(gids) if gids else max_items
    
    for gid in gids:
        self_ctx = get_ranked_context(n4j, gid, query, max_items=items_per_subgraph)
        link_ctx = get_ranked_link_context(n4j, gid, query, max_items=items_per_subgraph)
        
        all_self_context.extend(self_ctx)
        all_link_context.extend(link_ctx)
    
    all_self_context = list(dict.fromkeys(all_self_context))
    all_link_context = list(dict.fromkeys(all_link_context))
    
    logger.info(f"[Multi-Subgraph] Total: {len(all_self_context)} self, {len(all_link_context)} link")
    
    return all_self_context[:max_items], all_link_context[:max_items]

def get_improved_response(n4j, query: str, client=None, 
                          use_multi_subgraph: bool = False,
                          top_k_subgraphs: int = 1) -> Tuple[str, str]:
    """
    Generate response using improved retrieval pipeline
        
    Args:
        n4j: Neo4j connection
        query: User query
        client: Optional DedicatedKeyClient
        use_multi_subgraph: Whether to aggregate from multiple subgraphs (slower)
        top_k_subgraphs: Number of subgraphs to use (default 1 for speed)
    
    Returns:
        Tuple of (answer, primary_gid)
    """
    if client is None:
        client = create_dedicated_client(task_id="improved_response")
    
    logger.info(f"\n{'='*80}")
    logger.info("[Improved Response] Starting pipeline...")
    logger.info(f"{'='*80}")
    
    gids = hybrid_retrieve(n4j, query, client, top_k=top_k_subgraphs)
    
    if not gids:
        logger.error("[Improved Response] No GIDs found")
        return "", ""
    
    primary_gid = gids[0]
    
    if use_multi_subgraph and len(gids) > 1:
        self_context, link_context = aggregate_multi_subgraph_context(
            n4j, gids, query, max_items=100
        )
    else:
        self_context = get_ranked_context(n4j, primary_gid, query, max_items=50)
        link_context = get_ranked_link_context(n4j, primary_gid, query, max_items=50)
    
    sys_prompt_one = """
Please answer the question using insights supported by provided graph-based data relevant to medical information.
"""
    
    sys_prompt_two = """
Modify the response to the question using the provided references. Include precise citations relevant to your answer. You may use multiple citations simultaneously, denoting each with the reference index number. For example, cite the first and third documents as [1][3]. If the references do not pertain to the response, simply provide a concise answer to the original question.
"""
    MAX_CONTEXT_CHARS = 4000
    
    selfcont_str = "\n".join(self_context)
    linkcont_str = "\n".join(link_context)
    
    if len(selfcont_str) > MAX_CONTEXT_CHARS:
        selfcont_str = selfcont_str[:MAX_CONTEXT_CHARS] + "...(truncated)"
    
    if len(linkcont_str) > MAX_CONTEXT_CHARS:
        linkcont_str = linkcont_str[:MAX_CONTEXT_CHARS] + "...(truncated)"
    
    logger.info(f"[Improved Response] Context: {len(selfcont_str)} self, {len(linkcont_str)} link chars")
    
    user_one = f"the question is: {query}\n\nthe provided information is:\n{selfcont_str}"
    full_prompt_one = f"{sys_prompt_one}\n\n{user_one}"
    res = client.call_with_retry(full_prompt_one, model="models/gemini-2.5-flash-lite", max_retries=3)
    user_two = f"the question is: {query}\n\nthe last response of it is:\n{res}\n\nthe references are:\n{linkcont_str}"
    full_prompt_two = f"{sys_prompt_two}\n\n{user_two}"
    final_answer = client.call_with_retry(full_prompt_two, model="models/gemini-2.5-flash-lite", max_retries=3)
    
    logger.info(f"[Improved Response] Generated answer ({len(final_answer)} chars)")
    
    return final_answer, primary_gid

def improved_seq_ret(n4j, sumq, client=None):
    """
    Drop-in replacement for seq_ret with improved retrieval
    
    Args:
        n4j: Neo4j connection
        sumq: Query summary (list or string)
        client: Optional DedicatedKeyClient
    
    Returns:
        Best matching GID (single)
    """
    if client is None:
        client = create_dedicated_client(task_id="improved_seq_ret")
    query = sumq[0] if isinstance(sumq, list) else sumq
    gids = hybrid_retrieve(n4j, query, client, top_k=1)
    
    return gids[0] if gids else None

if __name__ == "__main__":
    import os
    from camel.storages import Neo4jGraph
    
    n4j = Neo4jGraph(
        url=os.getenv("NEO4J_URL"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD")
    )
    
    test_query = "What are the treatment options for heart failure with reduced ejection fraction?"
    
    client = create_dedicated_client(task_id="test")
    gids = hybrid_retrieve(n4j, test_query, client, top_k=3)
    print(f"\nHybrid Retrieve GIDs: {gids}")
    
    if gids:
        answer, gid = get_improved_response(n4j, test_query, client)
        print(f"\nAnswer:\n{answer}")
        print(f"\nPrimary GID: {gid}")
