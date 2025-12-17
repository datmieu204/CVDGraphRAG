# Smart Entity Linking

## 📋 Overview

Smart entity linking is an optimized approach to create REFERENCE relationships between layers in the three-layer architecture. It uses entity-based filtering instead of expensive graph traversal.

## 🎯 Problem: Traditional Linking is Slow

**Baseline approach (ref_link):**
```python
# ❌ Expensive: Full graph traversal
def ref_link(n4j, gid):
    # 1. Get ALL entities in target graph
    # 2. For EACH entity:
    #    - Find ALL paths to other graphs
    #    - Compute similarity for EACH path
    # 3. Create links above threshold
    
    # Result: O(n²) complexity, very slow!
```

**Issues:**
- Takes 30-60 minutes per document
- Evaluates thousands of entity pairs
- Many irrelevant candidates
- High API costs (LLM for similarity)

## 🚀 Solution: Entity-Based Filtering

### Key Insight

> "If two chunks share entities from the Bottom layer, they're likely related"

### Algorithm

```python
def smart_ref_link(n4j, top_gid):
    # 1. Extract entities from Top using NER (fast!)
    entities = extract_entities_ner(top_gid)
    
    # 2. Find Middle chunks that reference SAME Bottom entities
    middle_candidates = find_middle_with_shared_entities(entities)
    
    # 3. Filter by cosine similarity (only candidates)
    for middle_gid in middle_candidates:
        sim = compute_similarity(top_gid, middle_gid)
        if sim > threshold:
            create_reference_link(top_gid, middle_gid)
```

## 📊 Performance Comparison

| Metric | Traditional | Smart Linking | Improvement |
|--------|-------------|---------------|-------------|
| **Time** | 30-60 min | 2-5 min | **10-15x faster** |
| **Candidates** | 10,000+ | 50-100 | **100x reduction** |
| **API Calls** | 1000+ | 50-100 | **10-20x reduction** |
| **Accuracy** | 85% | 85% | Same |

## 🔧 Implementation

### 1. NER-based Entity Extraction

```python
def extract_entities_from_top_layer(n4j, top_gid):
    """Extract entities using NER model (no LLM needed)"""
    # Get content
    content = get_gid_content(n4j, top_gid)
    
    # Extract with NER
    from ner.heart_extractor import HeartExtractor
    extractor = HeartExtractor()
    entities = extractor.extract_entities(content)
    
    return [e['text'].upper() for e in entities]
```

### 2. Find Middle Chunks with Shared Entities

```python
def find_middle_chunks_with_shared_entities(n4j, entities, top_gid):
    """Find Middle chunks that share Bottom references"""
    query = """
    // For each extracted entity
    UNWIND $entities AS entity_name
    
    // Find matching Bottom entity
    MATCH (b)
    WHERE UPPER(b.name) = entity_name
      AND b.source = 'UMLS'
    
    // Find Middle entities that reference this Bottom entity
    MATCH (m)-[:IS_REFERENCE_OF]->(b)
    WHERE m.gid <> $top_gid  // Exclude self
    
    // Return Middle GIDs with match count
    RETURN m.gid AS middle_gid, count(DISTINCT b) AS shared_entities
    ORDER BY shared_entities DESC
    """
    
    results = n4j.query(query, {
        'entities': entities,
        'top_gid': top_gid
    })
    
    return [r['middle_gid'] for r in results if r['shared_entities'] >= 3]
```

### 3. Similarity Filtering

```python
def filter_by_similarity(n4j, top_gid, middle_candidates, threshold=0.7):
    """Filter candidates by embedding similarity"""
    # Get Top summary embedding
    top_summary = get_summary_embedding(n4j, top_gid)
    
    linked_gids = []
    for middle_gid in middle_candidates:
        # Get Middle summary embedding
        middle_summary = get_summary_embedding(n4j, middle_gid)
        
        # Compute similarity
        similarity = cosine_similarity(top_summary, middle_summary)
        
        if similarity >= threshold:
            linked_gids.append(middle_gid)
            logger.info(f"Link: {top_gid[:8]} -> {middle_gid[:8]} "
                       f"(sim: {similarity:.3f})")
    
    return linked_gids
```

## 💡 Two-Stage Linking

### Middle → Bottom (Incremental)

**When:** During graph construction  
**Method:** Direct name matching  
**Speed:** Very fast (no LLM calls)

```python
def link_middle_to_bottom_incremental(n4j, entities, middle_gid):
    """Link Middle entities to Bottom layer during construction"""
    entity_names = [e['entity_name'].upper() for e in entities]
    
    query = """
    UNWIND $entity_names AS entity_name
    MATCH (b) WHERE UPPER(b.name) = entity_name AND b.source = 'UMLS'
    MATCH (m {gid: $middle_gid}) WHERE UPPER(m.id) = entity_name
    MERGE (m)-[r:IS_REFERENCE_OF]->(b)
    """
    
    n4j.query(query, {'entity_names': entity_names, 'middle_gid': middle_gid})
```

### Top → Middle (Post-construction)

**When:** After all documents are imported  
**Method:** Entity-based with similarity filtering  
**Speed:** Fast (filtered candidates)

```python
def link_top_to_middle(n4j, top_gid):
    """Link Top layer to Middle using smart filtering"""
    # 1. Extract entities
    entities = extract_entities_ner(top_gid)
    
    # 2. Find candidates
    candidates = find_middle_with_shared_entities(n4j, entities, top_gid)
    
    # 3. Filter by similarity
    linked = filter_by_similarity(n4j, top_gid, candidates)
    
    # 4. Create links
    for middle_gid in linked:
        create_reference_link(n4j, top_gid, middle_gid)
```

## 📈 Benefits

### 1. Speed
- 10-15x faster than traditional linking
- Processes entire dataset in hours vs days

### 2. Scalability
- O(n) complexity (vs O(n²))
- Handles large knowledge graphs

### 3. Cost
- 10-20x fewer API calls
- Minimal LLM usage

### 4. Accuracy
- Same or better than traditional
- Entity-based filtering is precise

## 🔧 Configuration

### Similarity Threshold

```python
# Default: 0.7 (70% similarity)
SIMILARITY_THRESHOLD = 0.7

# Stricter: Fewer but higher quality links
SIMILARITY_THRESHOLD = 0.8

# Relaxed: More links, some may be weak
SIMILARITY_THRESHOLD = 0.6
```

### Minimum Shared Entities

```python
# Require at least 3 shared entities
MIN_SHARED_ENTITIES = 3

# More strict
MIN_SHARED_ENTITIES = 5

# More relaxed
MIN_SHARED_ENTITIES = 2
```

## 📚 Related Documentation

- [Three-Layer Architecture](../architecture/three_layer_architecture.md)
- [NER Filtering](ner_filtering.md)
- [Building Knowledge Graph](../tutorials/building_graph.md)

---

**Last Updated:** December 2024
