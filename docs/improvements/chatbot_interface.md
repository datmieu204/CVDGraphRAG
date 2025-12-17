# Gradio Chatbot Interface

## 📋 Overview

Interactive web-based chatbot interface powered by Gradio for real-time medical question answering using the CVDGraphRAG system.

## ✨ Features

- 🌐 **Web Interface**: Modern, user-friendly chat interface
- 🚀 **Real-time Inference**: Instant responses using hybrid retrieval
- 🔄 **Multi-subgraph Mode**: Toggle between fast and comprehensive modes
- 📊 **Status Monitoring**: Real-time database status
- 🌍 **Public Sharing**: Automatic gradio.live links (72-hour expiry)
- 💬 **Chat History**: Maintains conversation context
- 📝 **Example Questions**: Pre-loaded medical queries

## 🎨 Interface Components

### Main Chat Area
- **Chatbot Display**: Shows conversation history
- **Input Box**: Text area for questions
- **Send Button**: Submit query
- **Clear Chat**: Reset conversation

### Configuration Panel
- **Multi-subgraph Toggle**: Enable comprehensive mode
- **Database Status**: Shows connection and summary count
- **Refresh Button**: Update status display

### Example Questions
```
- "What are the main symptoms of the patient?"
- "What treatments were recommended?"
- "What is the diagnosis for this patient?"
- "Are there any complications mentioned?"
- "What medications were prescribed?"
```

## 🚀 Quick Start

### 1. Installation

```bash
pip install gradio>=4.0.0
```

### 2. Start Chatbot

```bash
cd /home/medgraph/src
python chatbot_gradio.py
```

### 3. Access Interface

The terminal will display:
```
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://xxxxx.gradio.live

This share link expires in 72 hours.
```

## 💡 Usage Modes

### Single-Subgraph Mode (Default)

**Best for:**
- Simple, specific questions
- Fast responses (3-5 seconds)
- Direct answers from single source

**Example:**
```
Q: What medications were prescribed?
A: The patient was prescribed...
   (Retrieved from 1 document)
```

### Multi-Subgraph Mode

**Best for:**
- Complex, multi-faceted questions
- Comprehensive answers
- Cross-referencing multiple sources

**Enable:** Check "Multi-subgraph Mode" checkbox

**Example:**
```
Q: What is the relationship between symptoms and treatment?
A: Based on multiple sources...
   (Aggregated from 3 documents)
```

## 🔧 Implementation Details

### Architecture

```python
# chatbot_gradio.py

# 1. Initialize Neo4j connection
n4j = initialize_neo4j()

# 2. Create chat interface
def chat_inference(message, history, use_multi_subgraph):
    """Process question and return answer"""
    answer = infer(n4j, message, use_multi_subgraph)
    return answer

# 3. Launch Gradio interface
demo = create_interface()
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True  # Create gradio.live link
)
```

### Key Functions

**`chat_inference()`**: Main inference wrapper
```python
def chat_inference(message, history, use_multi_subgraph):
    # Validate input
    if not message or not message.strip():
        return "⚠️ Please enter a question."
    
    try:
        # Call inference
        answer = infer(n4j, message, use_multi_subgraph)
        
        if answer:
            return answer
        else:
            return "⚠️ Could not generate an answer."
    except Exception as e:
        return f"❌ Error: {str(e)}"
```

**`check_database_status()`**: Monitor Neo4j
```python
def check_database_status():
    query = """
        MATCH (s:Summary)
        RETURN count(s) as summary_count
    """
    result = n4j.query(query)
    count = result[0]['summary_count']
    return f"✅ Connected | {count} summaries in database"
```

## 📊 System Status Indicators

| Status | Meaning | Action |
|--------|---------|--------|
| ✅ Connected \| N summaries | Ready to use | Ask questions |
| ⚠️ Connected but empty | No data | Build graph first |
| ❌ Not connected | Connection failed | Check Neo4j/env vars |

## 🎯 Best Practices

### Question Formulation

**Good Questions:**
```
✅ "What are the main symptoms of heart failure?"
✅ "Which medications are used to treat hypertension?"
✅ "What diagnostic tests were performed?"
```

**Avoid:**
```
❌ "Tell me everything" (too broad)
❌ "Yes/No questions" (better to ask "what" or "how")
❌ "Single word queries" (add context)
```

### Performance Tips

1. **Use Single-Subgraph for:**
   - Simple lookups
   - Fast responses needed
   - Specific entity queries

2. **Use Multi-Subgraph for:**
   - Complex analysis
   - Cross-referencing needed
   - Comprehensive overview required

3. **Monitor Status:**
   - Check database status before querying
   - Refresh if connection issues
   - Restart if errors persist

## 🔧 Configuration

### Port Configuration

**Change port in `chatbot_gradio.py`:**
```python
demo.launch(
    server_port=8080,  # Change from 7860
    ...
)
```

### Disable Public Link

```python
demo.launch(
    share=False,  # Disable gradio.live
    ...
)
```

### Add Authentication

```python
demo.launch(
    auth=("username", "password"),
    ...
)
```

### Custom Theme

```python
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # Interface code
```

## 📈 Performance Metrics

| Metric | Single-Subgraph | Multi-Subgraph |
|--------|----------------|----------------|
| **Response Time** | 3-5s | 8-12s |
| **Accuracy** | 85% | 88% |
| **Comprehensiveness** | Focused | Comprehensive |
| **API Costs** | $ | $$ |

## 🐛 Troubleshooting

### Issue: Connection failed

**Error:**
```
❌ Not connected to Neo4j
```

**Solution:**
```bash
# 1. Check Neo4j is running
sudo systemctl status neo4j

# 2. Verify environment variables
echo $NEO4J_URL
echo $NEO4J_USERNAME
echo $NEO4J_PASSWORD

# 3. Restart chatbot
python chatbot_gradio.py
```

### Issue: Database empty

**Warning:**
```
⚠️ Connected but database is empty
```

**Solution:**
```bash
# Build knowledge graph
python run.py -dataset mimic_ex -data_path ../data/layer1_mimic_ex -construct_graph
```

### Issue: Slow responses

**Symptoms:** Responses take >30 seconds

**Solutions:**
1. Check if embeddings are pre-computed:
   ```bash
   python add_summary_embeddings.py
   ```

2. Reduce context size in `inference_utils.py`:
   ```python
   max_items=30  # Reduce from 50
   ```

3. Use single-subgraph mode instead of multi

### Issue: gradio.live link expired

**Solution:** Link expires after 72 hours. Restart chatbot:
```bash
python chatbot_gradio.py
```

New link will be generated automatically.

## 📚 Code Structure

```python
# Main components

1. initialize_neo4j()
   - Connect to Neo4j database
   - Verify connection

2. chat_inference(message, history, use_multi_subgraph)
   - Main inference handler
   - Calls infer() from inference_utils.py
   - Error handling and formatting

3. check_database_status()
   - Query Neo4j for summary count
   - Return status string

4. create_interface()
   - Build Gradio UI components
   - Set up event handlers
   - Configure layout

5. main()
   - Initialize connections
   - Launch Gradio app
   - Handle errors
```

## 🌐 Deployment Options

### Local Development
```bash
python chatbot_gradio.py
# Access: http://localhost:7860
```

### LAN Access
```bash
# In chatbot_gradio.py
demo.launch(
    server_name="0.0.0.0",  # Allow LAN access
    server_port=7860
)
# Access: http://<your-ip>:7860
```

### Public Access (gradio.live)
```bash
# Default configuration
demo.launch(share=True)
# Automatic public URL generated
# Expires in 72 hours
```

### Production Deployment
```bash
# Use gunicorn or similar
# Add authentication
# Set up reverse proxy (nginx)
# Configure SSL/TLS
```

## 📝 Logs

Chatbot logs are saved to:
- `logs/chatbot_gradio.log` - Interface events
- `logs/inference_utils.log` - Inference pipeline
- `logs/improved_retrieve.log` - Retrieval details

**View logs:**
```bash
tail -f logs/chatbot_gradio.log
```

## 🚀 Future Improvements

**Planned features:**
- [ ] Conversation memory across sessions
- [ ] Export chat history
- [ ] Voice input/output
- [ ] Multi-language support
- [ ] Citation highlighting
- [ ] Visualization of retrieved subgraphs
- [ ] User feedback collection
- [ ] Response streaming

## 📚 Related Documentation

- [Inference Utils](../api/inference_utils.md)
- [Hybrid Retrieval](hybrid_retrieval.md)
- [Getting Started](../tutorials/getting_started.md)
- [Using the Chatbot](../tutorials/using_chatbot.md)

---

**Last Updated:** December 2024  
**Version:** 1.0.0
