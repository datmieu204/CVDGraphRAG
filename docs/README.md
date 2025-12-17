# CVDGraphRAG Documentation

Comprehensive documentation for the Medical Knowledge Graph RAG System

## 📚 Table of Contents

### 1. Architecture
- [Three-Layer Knowledge Graph Architecture](architecture/three_layer_architecture.md)
- [System Components Overview](architecture/system_components.md)
- [Data Flow & Processing Pipeline](architecture/data_flow.md)

### 2. Core Improvements
- [Hybrid U-Retrieval System](improvements/hybrid_retrieval.md)
- [Dedicated API Key Management](improvements/api_key_management.md)
- [Semantic Chunking](improvements/semantic_chunking.md)
- [Smart Entity Linking](improvements/smart_linking.md)
- [NER-based Filtering](improvements/ner_filtering.md)
- [Agentic Chunking](improvements/agentic_chunking.md)
- [Pre-computed Embeddings](improvements/precomputed_embeddings.md)
- [Gradio Chatbot Interface](improvements/chatbot_interface.md)

### 3. API Reference
- [Improved Retrieve Module](api/improved_retrieve.md)
- [Dedicated Key Manager](api/dedicated_key_manager.md)
- [Graph Construction](api/creat_graph_with_description.md)
- [Smart Linking](api/smart_linking.md)
- [Inference Utils](api/inference_utils.md)

### 4. Tutorials
- [Getting Started](tutorials/getting_started.md)
- [Building Knowledge Graph](tutorials/building_graph.md)
- [Running Inference](tutorials/running_inference.md)
- [Using the Chatbot](tutorials/using_chatbot.md)
- [Performance Optimization](tutorials/performance_optimization.md)

## 🎯 Quick Links

### For Developers
- Start with [System Components Overview](architecture/system_components.md)
- Review [API Reference](api/improved_retrieve.md)
- Check [Performance Optimization](tutorials/performance_optimization.md)

### For Users
- Begin with [Getting Started](tutorials/getting_started.md)
- Learn about [Using the Chatbot](tutorials/using_chatbot.md)
- Understand [Three-Layer Architecture](architecture/three_layer_architecture.md)

### For Researchers
- Read [Hybrid U-Retrieval System](improvements/hybrid_retrieval.md)
- Explore [Smart Entity Linking](improvements/smart_linking.md)
- Review [Data Flow & Processing](architecture/data_flow.md)

## 📊 Key Metrics

| Improvement | Impact | Details |
|-------------|--------|---------|
| **Hybrid Retrieval** | +15-20% accuracy | Vector search + LLM reranking |
| **NER Filtering** | -40-60% API costs | Skip irrelevant chunks early |
| **Dedicated Keys** | 3-5x throughput | Parallel processing without conflicts |
| **Semantic Chunking** | Zero LLM cost | Embedding-based segmentation |
| **Smart Linking** | 80% faster | Entity-based instead of full-graph search |

## 🚀 Recent Improvements (Latest Release)

### December 2024 Updates

1. **Hybrid U-Retrieval** (v2.0)
   - Combined vector search with LLM reranking
   - Multi-subgraph aggregation support
   - Query-aware context ranking
   - See: [Hybrid Retrieval](improvements/hybrid_retrieval.md)

2. **Dedicated API Key Manager** (v1.5)
   - Per-task key assignment
   - Automatic rotation on failure
   - Rate limiting (15 RPM per key)
   - See: [API Key Management](improvements/api_key_management.md)

3. **Gradio Chatbot Interface** (v1.0)
   - Interactive web interface
   - Real-time inference
   - Multi-subgraph mode toggle
   - Public sharing via gradio.live
   - See: [Chatbot Interface](improvements/chatbot_interface.md)

4. **Pre-computed Embeddings** (v1.2)
   - BGE-M3 embeddings cached in Neo4j
   - 10-20x faster vector search
   - Batch processing support
   - See: [Pre-computed Embeddings](improvements/precomputed_embeddings.md)

## 🔧 System Requirements

### Minimum
- Python 3.10+
- Neo4j 5.0+
- 8GB RAM
- 1 Gemini API key

### Recommended
- Python 3.10+
- Neo4j 5.0+
- 16GB+ RAM
- NVIDIA GPU (for NER)
- 3+ Gemini API keys (for parallel processing)

## 📝 Version History

### v2.1.0 (Current)
- Gradio chatbot interface
- Improved error handling
- Enhanced logging

### v2.0.0
- Hybrid U-Retrieval system
- Multi-subgraph aggregation
- Query-aware ranking

### v1.5.0
- Dedicated API key management
- Automatic key rotation
- Smart linking optimization

### v1.0.0
- Initial three-layer architecture
- Basic retrieval system
- UMLS integration

## 🤝 Contributing

We welcome contributions! See areas for improvement:
- [ ] Add more medical entity types
- [ ] Improve relationship extraction
- [ ] Optimize embedding models
- [ ] Enhance chatbot UI/UX
- [ ] Add evaluation metrics

## 📧 Contact

For questions or issues:
- GitHub Issues: [CVDGraphRAG Issues](https://github.com/datmieu204/CVDGraphRAG/issues)
- Email: datmieu204@gmail.com

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

---

**Last Updated:** December 2024  
**Documentation Version:** 2.1.0
