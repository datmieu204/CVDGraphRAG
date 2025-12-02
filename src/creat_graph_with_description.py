import time
"""
Integrated graph construction function
Uses nano_graphrag's entity_extraction prompt and parsing logic
But writes to Neo4j, supporting three-layer architecture
"""

import os

from dotenv import load_dotenv
load_dotenv()

import re
import asyncio
from typing import List, Dict
from collections import defaultdict

# Import nano_graphrag components
from nano_graphrag.prompt import PROMPTS
from nano_graphrag._utils import compute_mdhash_id

from dedicated_key_manager import create_dedicated_client

# Import existing components
from camel.loaders import UnstructuredIO
# from data_chunk import run_chunk
from chunking import chunk_document
from utils import get_embedding, str_uuid, add_sum
from logger_ import get_logger

logger = get_logger("creat_graph_with_description", log_file="logs/creat_graph_with_description.log")


def clean_str(input_str: str) -> str:
    """Clean string"""
    if not input_str:
        return ""
    # Remove quotes
    input_str = input_str.strip().strip('"').strip("'")
    return input_str


async def extract_entities_with_description(content: str, client, entity_types=None):
    """
    Extract entities and relationships using nano_graphrag's prompt (with descriptions)
    
    Args:
        content: Text content
        client: DedicatedKeyClient instance
        entity_types: Entity type list, if None uses default medical entity types
    
    Returns:
        (entities, relationships)
        entities: [{'entity_name': ..., 'entity_type': ..., 'description': ...}, ...]
        relationships: [{'src': ..., 'tgt': ..., 'description': ..., 'strength': ...}, ...]
    """
    if entity_types is None:
        entity_types = [
            "Disease", "Symptom", "Treatment", "Medication", "Test", 
            "Anatomy", "Procedure", "Condition", "Measurement", "Hormone",
            "Diagnostic_Criteria", "Clinical_Guideline", "Patient", "Doctor"
        ]
    
    prompt_template = PROMPTS["entity_extraction"]
    
    entity_types_str = ", ".join(entity_types)
    tuple_delimiter = "<|>"
    record_delimiter = "##"
    completion_delimiter = "<|COMPLETE|>"
    
    # Build complete prompt
    prompt = prompt_template.format(
        entity_types=entity_types_str,
        tuple_delimiter=tuple_delimiter,
        record_delimiter=record_delimiter,
        completion_delimiter=completion_delimiter,
        input_text=content[:3000]  # Limit input length
    )
    
    # Call Gemini with dedicated client
    logger.info(f"  [Entity Extraction] Extracting entities and relationships...")
    response = client.call_with_retry(prompt, model="models/gemini-2.5-flash-lite")
    
    entities = []
    relationships = []
    
    if not response:
        return entities, relationships
    
    records = response.split(record_delimiter)
    
    for record in records:
        record = record.strip()
        if not record or completion_delimiter in record:
            continue
        
        # Extract content within parentheses
        match = re.search(r'\((.*?)\)', record)
        if not match:
            continue
        
        record_content = match.group(1)
        attributes = record_content.split(tuple_delimiter)
        
        if len(attributes) < 2:
            continue
        
        record_type = clean_str(attributes[0])
        
        if record_type == "entity" and len(attributes) >= 4:
            # Entity record: ("entity"<|>entity_name<|>entity_type<|>entity_description)
            entity = {
                'entity_name': clean_str(attributes[1]).upper(),
                'entity_type': clean_str(attributes[2]).upper(),
                'description': clean_str(attributes[3])
            }
            if entity['entity_name']:
                entities.append(entity)
        
        elif record_type == "relationship" and len(attributes) >= 5:
            # Relationship record: ("relationship"<|>source<|>target<|>description<|>strength)
            relationship = {
                'src': clean_str(attributes[1]).upper(),
                'tgt': clean_str(attributes[2]).upper(),
                'description': clean_str(attributes[3]),
                'strength': clean_str(attributes[4])
            }
            if relationship['src'] and relationship['tgt']:
                relationships.append(relationship)

    logger.info(f"Extracted {len(entities)} entities, {len(relationships)} relationships")
    return entities, relationships


def create_neo4j_nodes_and_relationships(n4j, entities: List[Dict], relationships: List[Dict], gid: str):
    """
    Write extracted entities and relationships to Neo4j
    Uses batch processing to avoid memory overflow
    
    Args:
        n4j: Neo4j connection
        entities: Entity list
        relationships: Relationship list
        gid: Graph ID (for three-layer architecture)
    """
    import gc
    
    BATCH_SIZE = 32

    logger.info(f"  [Neo4j] Starting to write {len(entities)} entities...")

    # 1. Create entity nodes in batches
    total_batches = (len(entities) + BATCH_SIZE - 1) // BATCH_SIZE
    created_nodes = 0
    
    for batch_idx in range(0, len(entities), BATCH_SIZE):
        batch_entities = entities[batch_idx:batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1
        
        if total_batches > 1:
            logger.info(f"    [Batch {batch_num}/{total_batches}] Processing {len(batch_entities)} entities...")

        for entity in batch_entities:
            entity_name = entity['entity_name']
            entity_type = entity['entity_type']
            description = entity['description']
            
            # Generate embedding
            embedding_text = f"{entity_name}: {description}" if description else entity_name
            embedding = get_embedding(embedding_text)
            
            # Create node Cypher query
            create_node_query = """
            MERGE (n:`%s` {id: $id, gid: $gid})
            ON CREATE SET 
                n.description = $description,
                n.embedding = $embedding,
                n.source = 'nano_graphrag'
            ON MATCH SET
                n.description = CASE WHEN n.description IS NULL OR n.description = '' 
                                     THEN $description 
                                     ELSE n.description END,
                n.embedding = CASE WHEN n.embedding IS NULL 
                                   THEN $embedding 
                                   ELSE n.embedding END
            RETURN n
            """ % entity_type
            
            try:
                n4j.query(create_node_query, {
                    'id': entity_name,
                    'gid': gid,
                    'description': description,
                    'embedding': embedding
                })
                created_nodes += 1
            except Exception as e:
                logger.warning(f"  Failed to create node: {entity_name} - {e}")
        
        # Force garbage collection after each batch
        gc.collect()

    logger.info(f"Entity node creation completed: {created_nodes}/{len(entities)}")

    RELATION_BATCH_SIZE = 32
    logger.info(f"  [Neo4j] Starting to create {len(relationships)} relationships...")

    total_rel_batches = (len(relationships) + RELATION_BATCH_SIZE - 1) // RELATION_BATCH_SIZE
    created_relations = 0
    
    for batch_idx in range(0, len(relationships), RELATION_BATCH_SIZE):
        batch_relations = relationships[batch_idx:batch_idx + RELATION_BATCH_SIZE]
        batch_num = batch_idx // RELATION_BATCH_SIZE + 1
        
        if total_rel_batches > 1:
            logger.info(f"    [Batch {batch_num}/{total_rel_batches}] Processing {len(batch_relations)} relationships...")

        for rel in batch_relations:
            src = rel['src']
            tgt = rel['tgt']
            rel_type = "RELATED_TO"
            
            description = rel.get('description', '').lower()
            if 'treat' in description or 'cure' in description:
                rel_type = "TREATS"
            elif 'cause' in description or 'lead' in description:
                rel_type = "CAUSES"
            elif 'diagnose' in description or 'indicate' in description:
                rel_type = "INDICATES"
            elif 'symptom' in description or 'manifest' in description:
                rel_type = "HAS_SYMPTOM"
            
            create_rel_query = """
            MATCH (a {id: $src, gid: $gid})
            MATCH (b {id: $tgt, gid: $gid})
            MERGE (a)-[r:`%s`]->(b)
            ON CREATE SET r.description = $description, r.strength = $strength
            RETURN r
            """ % rel_type
            
            try:
                n4j.query(create_rel_query, {
                    'src': src,
                    'tgt': tgt,
                    'gid': gid,
                    'description': rel.get('description', ''),
                    'strength': rel.get('strength', '')
                })
                created_relations += 1
            except Exception as e:
                logger.warning(f"  Failed to create relationship: {src} -> {tgt} - {e}")
        
        gc.collect()

    logger.info(f"Relationship creation completed: {created_relations}/{len(relationships)}")


def creat_metagraph_with_description(args, content: str, gid: str, n4j):
    """
    Create knowledge graph using nano_graphrag's extraction logic (with description)
    But writes to Neo4j and supports three-layer architecture (gid)
    
    IMPROVEMENTS:
    - Semantic chunking (embedding-based)
    - NER-based filtering (skip chunks with low Bottom layer overlap)
    - Incremental Middle→Bottom linking
    
    Args:
        args: Arguments
        content: Text content
        gid: Graph ID (for three-layer architecture)
        n4j: Neo4j connection
    
    Returns:
        n4j: Updated Neo4j connection
    """
    logger.info(f"[Graph Construction] Starting knowledge graph construction (GID: {gid[:8]}...)")

    client = create_dedicated_client(task_id=f"gid_{gid[:8]}")
    logger.info(f"[API Key] Using dedicated key #{client.key_index + 1} for this file")

    uio = UnstructuredIO()
    
    ner_extractor = None
    if hasattr(args, 'bottom_filter') and args.bottom_filter:
        logger.info("[NER] Initializing HeartExtractor for filtering...")
        try:
            from ner.heart_extractor import HeartExtractor
            ner_extractor = HeartExtractor()
            logger.info("NER model loaded successfully")
        except Exception as e:
            logger.warning(f"NER model failed to load: {e}")
            logger.warning("  Falling back to no-filter mode")
    
    if args.grained_chunk:
        logger.info("[Chunking] Using semantic chunking (embedding-based)...")
        content_chunks = chunk_document(
            content,
            threshold=0.85,
            max_chunk_sentences=15,
            max_chunk_tokens=512,
            log_stats=True
        )
        
        # from data_chunk import run_chunk
        # content_chunks = run_chunk(content, client)
    else:
        logger.info("[Chunking] Using full content...")
        content_chunks = [content]
    
    total_chunks = len(content_chunks)
    processed_chunks = 0
    skipped_chunks = 0
    
    all_entities = []
    all_relationships = []
    
    for idx, chunk in enumerate(content_chunks, 1):
        logger.info(f"\n[Chunk {idx}/{total_chunks}] Processing...")
        
        if ner_extractor is not None:
            logger.info(f"  [NER Filter] Checking Bottom layer overlap...")
            try:
                from creat_graph import is_valid_entity, check_entities_in_bottom_layer
                
                extracted_ner = ner_extractor.extract_entities(chunk)
                
                min_threshold = getattr(args, 'min_overlap', 3)
                relevant_count, matched_entities, total_entities = check_entities_in_bottom_layer(
                    n4j, extracted_ner, gid, min_overlap=min_threshold
                )
                
                if relevant_count < min_threshold:
                    logger.info(f"SKIPPING: Only {relevant_count}/{total_entities} entities match Bottom layer (< {min_threshold})")
                    logger.info(f"LLM calls saved: 2 (entity extraction + relationship extraction)")
                    skipped_chunks += 1
                    continue
                else:
                    logger.info(f"PROCESSING: {relevant_count}/{total_entities} entities match Bottom layer")
            
            except Exception as e:
                logger.warning(f"NER filtering failed: {e}")
 
        entities, relationships = asyncio.run(
            extract_entities_with_description(chunk, client)
        )
        
        all_entities.extend(entities)
        all_relationships.extend(relationships)
        processed_chunks += 1
    
    logger.info(f"\n[Extraction Summary] Total extracted:")
    logger.info(f"  - Entities: {len(all_entities)}")
    logger.info(f"  - Relationships: {len(all_relationships)}")

    # Merge duplicate entities
    logger.info(f"\n[Merging] Merging duplicate entities...")
    entity_dict = {}
    for entity in all_entities:
        name = entity['entity_name']
        if name in entity_dict:
            # Merge descriptions
            existing_desc = entity_dict[name]['description']
            new_desc = entity['description']
            if new_desc and new_desc not in existing_desc:
                entity_dict[name]['description'] = f"{existing_desc}; {new_desc}"
        else:
            entity_dict[name] = entity
    
    merged_entities = list(entity_dict.values())
    logger.info(f"After merging: {len(merged_entities)} entities")
    logger.info(f"\n[Writing to Neo4j] Starting...")
    create_neo4j_nodes_and_relationships(n4j, merged_entities, all_relationships, gid)
    
    logger.info(f"\n[Incremental Linking] Creating Middle→Bottom references...")
    from smart_linking import link_middle_to_bottom_incremental
    links_created = link_middle_to_bottom_incremental(n4j, merged_entities, gid)
    logger.info(f"{links_created} IS_REFERENCE_OF links created")
    
    if args.ingraphmerge:
        logger.info(f"\n[In-Graph Merge] Merging similar nodes...")
        from utils import merge_similar_nodes
        merge_similar_nodes(n4j, gid)
    
    logger.info(f"\n[Summary] Creating summary node...")
    add_sum(n4j, content, gid, client=client)

    logger.info(f"[Graph Construction] Completed! (GID: {gid[:8]}...)")
    
    return n4j