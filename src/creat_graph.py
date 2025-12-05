import os
import re
from getpass import getpass
from camel.storages import Neo4jGraph
from camel.agents import KnowledgeGraphAgent
from camel.loaders import UnstructuredIO
from dataloader import load_high
import argparse
# from data_chunk import run_chunk
from chunking import chunk_document
from utils import *
from logger_ import get_logger

logger = get_logger("creat_graph", log_file="logs/creat_graph.log")


def is_valid_entity(entity_text: str) -> bool:
    """
    Filter out invalid entities (special characters, too short, etc.)
    
    Args:
        entity_text: Extracted entity text
    
    Returns:
        True if entity is valid, False otherwise
    """
    if not entity_text or len(entity_text.strip()) < 2:
        return False
    
    entity_text = entity_text.strip()
    
    # Check if entity is mostly special characters or numbers
    # Allow medical entities like "HF", "EF", "NT-proBNP"
    alpha_count = sum(c.isalpha() for c in entity_text)
    if alpha_count < 2:  # At least 2 alphabetic characters
        return False
    
    noise_patterns = [
        r'^\[+$',
        r'^\{+$',
        r'^\(+$',
        r'^[\d\s\.,;:]+$',
        r'^[^\w\s]+$',
    ]
    
    for pattern in noise_patterns:
        if re.match(pattern, entity_text):
            return False
    
    if len(entity_text) == 1 and not entity_text.isalnum():
        return False
    
    return True


def check_entities_in_bottom_layer(n4j, extracted_entities: dict, gid: str, min_overlap: int = 3) -> tuple:
    """
    Check how many extracted entities exist in the Bottom layer (UMLS/foundational knowledge)
    
    Args:
        n4j: Neo4j connection
        extracted_entities: Dict from NER {entity_class: [entity1, entity2, ...]}
        gid: Current graph ID
        min_overlap: Minimum overlapping entities to consider chunk relevant
    
    Returns:
        (relevant_count, matched_entities, total_entities)
    """
    # Flatten all entities from all classes
    all_entities = []
    for entity_class, entities in extracted_entities.items():
        for entity in entities:
            if is_valid_entity(entity):
                all_entities.append(entity.upper().strip())
    
    if not all_entities:
        return 0, [], 0
    
    total_entities = len(all_entities)
    
    # Query to check which entities exist in Bottom layer (UMLS medical knowledge)
    # Bottom layer = entities with source='UMLS' or medical labels
    # Use UPPER() for case-insensitive matching on n.name
    query = """
    MATCH (n)
    WHERE UPPER(n.name) IN $entity_ids 
      AND (n.source = 'UMLS'
           OR n:DISEASE OR n:MEDICATION OR n:SYMPTOM 
           OR n:PROCEDURE OR n:ANATOMY OR n:CONCEPT)
    RETURN DISTINCT n.name as matched_entity
    LIMIT 100
    """
    
    try:
        results = n4j.query(query, {'entity_ids': all_entities})
        matched_entities = [r['matched_entity'] for r in results]
        relevant_count = len(matched_entities)
        
        logger.info(f"[NER Filter] {relevant_count}/{total_entities} entities found in Bottom layer")
        if matched_entities:
            logger.debug(f"Sample matches: {matched_entities[:3]}")
        
        return relevant_count, matched_entities, total_entities
    
    except Exception as e:
        logger.warning(f"[NER Filter] Error checking Bottom layer: {e}")
        return 0, [], total_entities


def creat_metagraph(args, content, gid, n4j):
    """
    Legacy graph creation function with NER-based filtering
    Now includes smart filtering to reduce unnecessary LLM calls
    """
    from dedicated_key_manager import create_dedicated_client
    
    logger.info(f"[Graph Creation] Starting with NER filtering (GID: {gid[:8]}...)")

    uio = UnstructuredIO()
    kg_agent = KnowledgeGraphAgent()
    whole_chunk = content
    
    client = create_dedicated_client(task_id=f"legacy_gid_{gid[:8]}")
    
    logger.info("[NER] Initializing HeartExtractor model...")
    try:
        from ner.heart_extractor import HeartExtractor
        ner_extractor = HeartExtractor()
        logger.info("NER model loaded successfully")
    except Exception as e:
        logger.warning(f"NER model failed to load: {e}")
        logger.warning("Falling back to no-filter mode")
        ner_extractor = None

    # Chunking
    if args.grained_chunk == True:
        logger.info("[Chunking] Using semantic chunking...")
        content = chunk_document(
            content,
            threshold=0.85,
            max_chunk_sentences=15,
            max_chunk_tokens=512,
            log_stats=True
        )
        
        # OLD: Agentic chunking (LLM-based, expensive - kept for reference)
        # from data_chunk import run_chunk
        # content = run_chunk(content, client=client)
    else:
        logger.info("[Chunking] Using full content (no chunking)...")
        content = [content]
    
    logger.info(f"[Processing] Total chunks to process: {len(content)}")
    
    total_chunks = len(content)
    processed_chunks = 0
    skipped_chunks = 0
    
    for chunk_idx, cont in enumerate(content, 1):
        logger.info(f"[Chunk {chunk_idx}/{total_chunks}] Processing...")
        
        if ner_extractor is not None:
            logger.info(f"[NER] Extracting entities from chunk...")
            try:
                extracted_entities = ner_extractor.extract_entities(cont)
                
                total_extracted = sum(len(v) for v in extracted_entities.values())
                logger.info(f"Extracted {total_extracted} entities across {len(extracted_entities)} classes")
                
                top_classes = sorted(extracted_entities.items(), key=lambda x: len(x[1]), reverse=True)[:5]
                for entity_class, entities in top_classes:
                    logger.info(f"    - {entity_class}: {len(entities)} entities")
                
            except Exception as e:
                logger.warning(f"NER extraction failed: {e}")
                extracted_entities = {}
        else:
            extracted_entities = {}
        
        if extracted_entities and args.bottom_filter:
            relevant_count, matched_entities, total_entities = check_entities_in_bottom_layer(
                n4j, 
                extracted_entities, 
                gid,
                min_overlap=args.min_overlap if hasattr(args, 'min_overlap') else 3
            )
            
            min_threshold = getattr(args, 'min_overlap', 3)
            
            if relevant_count < min_threshold:
                logger.info(f"SKIPPING: Only {relevant_count} entities match Bottom layer (< {min_threshold})")
                logger.info(f"Matched entities: {matched_entities[:10]}...")
                skipped_chunks += 1
                continue
            else:
                logger.info(f"PROCESSING: {relevant_count} entities match Bottom layer (>= {min_threshold})")
                logger.info(f"Sample matches: {matched_entities[:5]}...")
        
        logger.info(f"[LLM] Calling KnowledgeGraphAgent...")
        
        # OLD CODE (kept for reference):
        # element_example = uio.create_element_from_text(text=cont)
        # ans_str = kg_agent.run(element_example, parse_graph_elements=False)
        # graph_elements = kg_agent.run(element_example, parse_graph_elements=True)
        # graph_elements = add_ge_emb(graph_elements)
        # graph_elements = add_gid(graph_elements, gid)
        # n4j.add_graph_elements(graph_elements=[graph_elements])
        
        try:
            element_example = uio.create_element_from_text(text=cont)
            
            # First pass: check if graph extraction makes sense
            ans_str = kg_agent.run(element_example, parse_graph_elements=False)
            
            # Second pass: parse graph elements
            graph_elements = kg_agent.run(element_example, parse_graph_elements=True)
            
            graph_elements = add_ge_emb(graph_elements)
            graph_elements = add_gid(graph_elements, gid)
            
            n4j.add_graph_elements(graph_elements=[graph_elements])
            
            processed_chunks += 1
            logger.info(f"Chunk processed successfully")
            
        except Exception as e:
            logger.error(f"Failed to process chunk: {e}")
            skipped_chunks += 1
            continue
    
    logger.info(f"[Summary] Chunk Processing Statistics:")
    logger.info(f"  Total chunks: {total_chunks}")
    logger.info(f"  Processed: {processed_chunks} ({processed_chunks*100//total_chunks if total_chunks > 0 else 0}%)")
    logger.info(f"  Skipped: {skipped_chunks} ({skipped_chunks*100//total_chunks if total_chunks > 0 else 0}%)")
    logger.info(f"  LLM calls saved: {skipped_chunks * 2}")  # Each chunk = 2 LLM calls
    
    if args.ingraphmerge and processed_chunks > 0:
        logger.info("[Post-processing] Merging similar nodes...")
        merge_similar_nodes(n4j, gid)
    
    logger.info("[Post-processing] Creating summary node...")
    add_sum(n4j, whole_chunk, gid, client=client)
    
    logger.info(f"\nGraph creation completed for GID: {gid[:8]}...\n")
    
    return n4j

