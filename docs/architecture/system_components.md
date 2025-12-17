# System Components Overview

## 📋 Architecture Overview

CVDGraphRAG consists of multiple interconnected modules that work together to provide medical knowledge graph RAG capabilities.

## 🏗️ Core Components

### 1. Data Import & Processing

#### Three-Layer Importer (`three_layer_import.py`)
**Purpose:** Orchestrates the import of all three layers

**Key Features:**
- Automatic layer ordering (Bottom → Middle → Top)
- Batch processing for large datasets
- Progress tracking and logging
- Database management (clear/verify)

**Usage:**
```bash
python three_layer_import.py --all --data-dir ../data
```

#### Multimodal Parser (`multimodal_parser/`)
**Purpose:** Extract text from various document formats

**Capabilities:**
- PDF parsing with layout awareness
- Office document processing (docx, pptx)
- Table and figure extraction
- Text structure preservation

**Supported Formats:**
- PDF, DOCX, PPTX, TXT
- HTML, XML, CSV

---

### 2. Graph Construction

#### Graph Constructor (`creat_graph_with_description.py`)
**Purpose:** Build knowledge graphs from text with entity descriptions

**Pipeline:**
1. **Semantic Chunking**: Break documents into coherent segments
2. **NER Filtering**: Skip chunks with low Bottom layer overlap
3. **Entity Extraction**: LLM-based extraction with descriptions
4. **Graph Creation**: Write nodes and relationships to Neo4j
5. **Incremental Linking**: Create Middle→Bottom references
6. **Summarization**: Generate and store document summary

**Improvements:**
- ✅ Semantic chunking (embedding-based)
- ✅ NER-based filtering (40-60% cost reduction)
- ✅ Batch processing (memory efficient)
- ✅ Incremental linking (fast)

#### Entity Linking (`smart_linking.py`)
**Purpose:** Create inter-layer REFERENCE relationships

**Methods:**
- **Incremental (Middle→Bottom)**: During construction
- **Entity-based (Top→Middle)**: Post-construction

**Performance:**
- 10-15x faster than traditional graph traversal
- 100x candidate reduction
- Same accuracy

---

### 3. Retrieval System

#### Hybrid U-Retrieval (`improved_retrieve.py`)
**Purpose:** Intelligent subgraph retrieval with reranking

**Components:**

1. **Vector Search**: Fast pre-filtering with BGE embeddings
2. **LLM Reranking**: Semantic relevance evaluation
3. **Context Extraction**: Query-aware triple ranking
4. **Multi-Subgraph**: Aggregate from multiple sources

**Performance:**
- 5-8x faster than baseline
- +15-20% accuracy
- Pre-computed embeddings for speed

#### Inference Engine (`inference_utils.py`, `run.py`)
**Purpose:** End-to-end question answering

**Pipeline:**
```
Query → Hybrid Retrieve → Context Extract → LLM Synthesis → Answer
```

**Modes:**
- Single-subgraph: Fast, focused
- Multi-subgraph: Comprehensive, slower

---

### 4. Chunking Systems

#### Semantic Chunker (`chunking/semantic_chunker.py`)
**Purpose:** Embedding-based document segmentation

**Algorithm:**
1. Split into sentences
2. Compute sentence embeddings
3. Find semantic breaks (low similarity)
4. Merge into coherent chunks

**Benefits:**
- No LLM costs
- Semantically coherent
- Configurable threshold

#### Agentic Chunker (`agentic_chunker.py`)
**Purpose:** LLM-guided intelligent chunking

**Algorithm:**
1. Extract propositions
2. Group similar propositions
3. Generate chunk summaries
4. Update dynamically

**Use Cases:**
- Complex narratives
- Topic-based grouping
- Adaptive granularity

---

### 5. Entity Recognition

#### Heart Extractor (`ner/heart_extractor.py`)
**Purpose:** Medical NER using fine-tuned BioBERT

**Capabilities:**
- Disease detection
- Medication recognition
- Symptom extraction
- Anatomy identification
- Procedure recognition

**Performance:**
- GPU-accelerated
- High precision for medical terms
- Works without LLM

**Integration:**
- Used in NER filtering
- Used in smart linking
- Used in entity extraction

---

### 6. API Management

#### Dedicated Key Manager (`dedicated_key_manager.py`)
**Purpose:** Multi-key management with automatic rotation

**Features:**
- Per-task key assignment
- Automatic load balancing
- Rate limiting (15 RPM per key)
- Auto-rotation on failure
- Thread-safe operations

**Benefits:**
- 3-5x throughput (parallel processing)
- 95% fewer rate limit errors
- Zero manual intervention

**Components:**
- `DedicatedKeyManager`: Singleton key pool
- `DedicatedKeyClient`: Per-task client

---

### 7. Embeddings

#### Embedding Manager (`utils.py`)
**Purpose:** Generate and manage embeddings

**Models:**
- **BGE-M3**: Fast, accurate, 1024-dim
- **BGE-small**: Lightweight alternative

**Pre-computation (`add_summary_embeddings.py`):**
- Batch processing
- Neo4j storage
- 10-20x faster retrieval

---

### 8. User Interfaces

#### Gradio Chatbot (`chatbot_gradio.py`)
**Purpose:** Web-based chat interface

**Features:**
- Real-time inference
- Single/multi-subgraph toggle
- Database status monitoring
- Example questions
- Public sharing (gradio.live)

**Deployment:**
```bash
python chatbot_gradio.py
# Access: http://localhost:7860
# Public: https://xxxxx.gradio.live
```

---

### 9. Utilities

#### Logger (`logger_.py`)
**Purpose:** Centralized logging system

**Features:**
- Per-module log files
- Structured logging
- Debug/info/warning/error levels
- Automatic log rotation

**Logs Location:**
```
logs/
├── chatbot_gradio.log
├── inference_utils.log
├── improved_retrieve.log
├── creat_graph_with_description.log
├── three_layer_importer.log
└── ...
```

#### Utils (`utils.py`)
**Purpose:** Common utility functions

**Functions:**
- `get_embedding()`: Generate embeddings
- `str_uuid()`: Generate unique IDs
- `add_sum()`: Create summary nodes
- `merge_similar_nodes()`: Deduplicate entities
- `cosine_similarity()`: Compute similarity
- `load_high()`: Load text files

---

## 📊 Component Dependencies

```
┌─────────────────────────────────────────────────────────┐
│                   User Interfaces                        │
│         chatbot_gradio.py, run.py (CLI)                 │
└────────────────────┬────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────┐
│                 Inference Layer                          │
│   inference_utils.py, improved_retrieve.py              │
└────────┬──────────────────────────┬─────────────────────┘
         │                          │
         ↓                          ↓
┌─────────────────┐      ┌─────────────────────────────┐
│  Graph Storage  │      │    API Management           │
│    Neo4j DB     │      │ dedicated_key_manager.py    │
└────────┬────────┘      └──────────┬──────────────────┘
         │                          │
         ↓                          ↓
┌─────────────────────────────────────────────────────────┐
│              Graph Construction Layer                    │
│  creat_graph_with_description.py, smart_linking.py     │
└────────┬───────────────────┬────────────────────────────┘
         │                   │
         ↓                   ↓
┌──────────────────┐  ┌────────────────────────┐
│   Chunking       │  │   Entity Recognition   │
│  semantic.py     │  │  heart_extractor.py    │
│  agentic.py      │  │  (NER)                 │
└──────────────────┘  └────────────────────────┘
         │                   │
         └─────────┬─────────┘
                   ↓
         ┌──────────────────┐
         │   Embeddings     │
         │   utils.py       │
         │   BGE-M3         │
         └──────────────────┘
```

## 🔄 Data Flow

### Graph Construction Flow

```
Raw Documents
    ↓
[Multimodal Parser]
    ↓
Plain Text
    ↓
[Semantic/Agentic Chunker]
    ↓
Text Chunks
    ↓
[NER Filter] ← HeartExtractor
    ↓ (filtered chunks)
[Entity Extraction] ← Dedicated Key Manager
    ↓
Entities & Relationships
    ↓
[Neo4j Writer]
    ↓
[Incremental Linking] → Bottom Layer
    ↓
[Summarization] ← Dedicated Key Manager
    ↓
Complete Subgraph
```

### Inference Flow

```
User Query
    ↓
[Embedding Generation] → BGE-M3
    ↓
[Vector Search] → Pre-computed Summary Embeddings
    ↓
Top-N Candidates
    ↓
[LLM Reranking] ← Dedicated Key Manager
    ↓
Top-K GIDs
    ↓
[Context Extraction]
    ├─ Self-context (triples)
    └─ Link-context (references)
    ↓
[Query-aware Ranking]
    ↓
Ranked Context
    ↓
[LLM Synthesis] ← Dedicated Key Manager
    ├─ Stage 1: Self-context → Draft answer
    └─ Stage 2: Link-context → Final answer with citations
    ↓
Final Answer
```

## 🎯 Component Selection Guide

### For Graph Construction

| Task | Component | When to Use |
|------|-----------|-------------|
| Import Bottom Layer | `three_layer_import.py --bottom` | Once per dataset |
| Import Middle Layer | `three_layer_import.py --middle` | For guidelines/papers |
| Import Top Layer | `three_layer_import.py --top` | For patient cases |
| Parse PDFs | `multimodal_parser/` | Non-text documents |

### For Retrieval

| Task | Component | When to Use |
|------|-----------|-------------|
| Fast single-source | `improved_retrieve.py` (single) | Simple queries |
| Comprehensive multi-source | `improved_retrieve.py` (multi) | Complex queries |
| Baseline retrieval | `retrieve.py` (deprecated) | Legacy support only |

### For Chunking

| Task | Component | When to Use |
|------|-----------|-------------|
| General documents | `semantic_chunker.py` | Most cases |
| Complex narratives | `agentic_chunker.py` | Adaptive needs |
| No chunking | Pass full text | Short documents |

### For Inference

| Task | Component | When to Use |
|------|-----------|-------------|
| CLI inference | `run.py -improved_inference` | Batch processing |
| Interactive chat | `chatbot_gradio.py` | User-facing |
| Programmatic | `inference_utils.infer()` | Custom integration |

## 📚 Related Documentation

- [Three-Layer Architecture](three_layer_architecture.md)
- [Data Flow & Processing](data_flow.md)
- [Getting Started](../tutorials/getting_started.md)
- [API Reference](../api/improved_retrieve.md)

---

**Last Updated:** December 2024
