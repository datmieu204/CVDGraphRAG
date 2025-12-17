# Getting Started with CVDGraphRAG

## 📋 Prerequisites

### System Requirements

**Minimum:**
- Ubuntu 20.04+ or similar Linux distribution
- Python 3.10+
- Neo4j 5.0+
- 8GB RAM
- 20GB free disk space

**Recommended:**
- 16GB+ RAM
- NVIDIA GPU (for NER)
- SSD storage
- 50GB+ free disk space

### Required Accounts

1. **Neo4j Database**
   - Download: https://neo4j.com/download/
   - OR use Neo4j Cloud (AuraDB)

2. **Google Gemini API Keys**
   - Get keys: https://ai.google.dev/
   - Recommended: 3-5 keys for parallel processing

## 🚀 Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/datmieu204/CVDGraphRAG.git
cd CVDGraphRAG
```

### Step 2: Create Virtual Environment

```bash
# Create environment
python -m venv medgraphenv

# Activate
source medgraphenv/bin/activate  # Linux/Mac
# OR
medgraphenv\Scripts\activate  # Windows
```

### Step 3: Install Dependencies

```bash
# Install requirements
pip install -r requirements.txt

# Install Gradio (for chatbot)
pip install gradio>=4.0.0
```

### Step 4: Set Up Neo4j

#### Option A: Local Installation

```bash
# Install Neo4j
sudo apt-get update
sudo apt-get install neo4j

# Start Neo4j
sudo systemctl start neo4j
sudo systemctl enable neo4j

# Verify
sudo systemctl status neo4j
```

#### Option B: Docker

```bash
docker run \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/your_password \
    -v $HOME/neo4j/data:/data \
    neo4j:5.0
```

### Step 5: Configure Environment Variables

Create `.env` file in project root:

```bash
cd /home/medgraph
nano .env
```

Add the following:

```bash
# Neo4j Configuration
NEO4J_URL=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Gemini API Keys (add as many as you have)
GEMINI_API_KEY_1=your_first_gemini_key
GEMINI_API_KEY_2=your_second_gemini_key
GEMINI_API_KEY_3=your_third_gemini_key
GEMINI_API_KEY_4=your_fourth_gemini_key
GEMINI_API_KEY_5=your_fifth_gemini_key

# Optional: Custom settings
# EMBEDDING_MODEL=BAAI/bge-m3
# MAX_CHUNK_SIZE=512
```

**Important:** Replace placeholder values with your actual credentials!

### Step 6: Verify Installation

```bash
cd src

# Test Neo4j connection
python -c "
from camel.storages import Neo4jGraph
import os
from dotenv import load_dotenv

load_dotenv('../.env')

n4j = Neo4jGraph(
    url=os.getenv('NEO4J_URL'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD')
)

result = n4j.query('RETURN 1 as test')
print('✅ Neo4j connection successful!')
print(f'Test result: {result}')
"
```

Expected output:
```
✅ Neo4j connection successful!
Test result: [{'test': 1}]
```

## 📦 Download Sample Data

### Option 1: Use Provided Sample Data

```bash
cd /home/medgraph/data

# Sample data is already included in repository
ls layer1_mimic_ex/  # Should show sample reports
```

### Option 2: Download Additional Data

**MIMIC-IV (requires PhysioNet credential):**
1. Get access: https://physionet.org/content/mimiciv/
2. Download discharge summaries
3. Place in `data/layer1_mimic_ex/`

**PubMed Central Articles:**
1. Download from: https://www.ncbi.nlm.nih.gov/pmc/
2. Extract full-text articles
3. Place in `data/layer2_pmc/`

**UMLS (requires UMLS license):**
1. Get license: https://www.nlm.nih.gov/research/umls/
2. Download UMLS files
3. Convert to CSV format
4. Place in `data/layer3_umls/`

## 🏗️ Build Your First Knowledge Graph

### Step 1: Import Bottom Layer (Optional)

**Note:** This is optional and takes significant time. Skip if you want to start quickly.

```bash
cd /home/medgraph/src

# Import UMLS (if you have UMLS data)
python three_layer_import.py \
    --bottom \
    --data-path ../data/layer3_umls \
    --clear  # Clear database first
```

This will take 1-2 hours for full UMLS.

### Step 2: Import Sample Documents

```bash
# Import sample reports (much faster!)
python run.py \
    -construct_graph \
    -dataset mimic_ex \
    -data_path ../data/layer1_mimic_ex \
    -grained_chunk
```

**Options:**
- `-grained_chunk`: Use semantic chunking (recommended)
- `-bottom_filter`: Use NER filtering (requires GPU)
- `-ingraphmerge`: Merge similar entities

**Expected output:**
```
[Graph Construction] Starting knowledge graph construction...
[Chunking] Using semantic chunking...
[Chunk 1/5] Processing...
[Entity Extraction] Extracting entities...
Extracted 45 entities, 89 relationships
[Neo4j] Writing to database...
[Summary] Creating summary node...
[Graph Construction] Completed!
```

### Step 3: Pre-compute Embeddings

```bash
# This makes retrieval 10-20x faster!
python add_summary_embeddings.py --batch-size 50
```

Expected output:
```
Found 20 summaries without embeddings
Adding embeddings: 100%|██████████| 20/20
Successfully added embeddings: 20
```

### Step 4: Verify Graph

```bash
# Check database
python -c "
from camel.storages import Neo4jGraph
import os
from dotenv import load_dotenv

load_dotenv('../.env')

n4j = Neo4jGraph(
    url=os.getenv('NEO4J_URL'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD')
)

# Count nodes by type
query = '''
MATCH (n)
RETURN labels(n)[0] as label, count(n) as count
ORDER BY count DESC
LIMIT 10
'''

results = n4j.query(query)
print('\n📊 Database Contents:')
for r in results:
    print(f'  {r[\"label\"]}: {r[\"count\"]}')
"
```

## 🎯 Run Your First Query

### Method 1: CLI Inference

Create a question file:

```bash
cd /home/medgraph/src
echo "What are the main symptoms of the patient?" > prompt.txt

# Run inference
python run.py -improved_inference
```

### Method 2: Interactive Chatbot

```bash
# Start chatbot
python chatbot_gradio.py
```

Then open browser to: http://localhost:7860

### Method 3: Python Script

```python
# test_query.py
import os
from camel.storages import Neo4jGraph
from inference_utils import infer
from dotenv import load_dotenv

load_dotenv('../.env')

# Connect to Neo4j
n4j = Neo4jGraph(
    url=os.getenv('NEO4J_URL'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD')
)

# Ask question
question = "What are treatment options for heart failure?"
answer = infer(n4j, question, use_multi_subgraph=False)

print(f"\nQuestion: {question}")
print(f"\nAnswer:\n{answer}")
```

Run:
```bash
python test_query.py
```

## ✅ Verification Checklist

- [ ] Neo4j is running and accessible
- [ ] Environment variables are set correctly
- [ ] At least 1 Gemini API key is configured
- [ ] Sample data imported successfully
- [ ] Embeddings pre-computed
- [ ] First query returns valid answer
- [ ] Chatbot launches successfully

## 🐛 Common Issues

### Issue: Neo4j connection failed

**Error:**
```
ServiceUnavailable: Could not connect to bolt://localhost:7687
```

**Solution:**
```bash
# Check if Neo4j is running
sudo systemctl status neo4j

# If not running, start it
sudo systemctl start neo4j

# Check firewall
sudo ufw allow 7687/tcp
```

### Issue: API key invalid

**Error:**
```
PERMISSION_DENIED: API key not valid
```

**Solution:**
1. Verify key at: https://ai.google.dev/
2. Check `.env` file for typos
3. Ensure no spaces around `=` in `.env`
4. Try regenerating the key

### Issue: Out of memory

**Error:**
```
OutOfMemoryError: Java heap space
```

**Solution:**
Increase Neo4j memory in `neo4j.conf`:
```
server.memory.heap.initial_size=2G
server.memory.heap.max_size=4G
```

### Issue: Import too slow

**Solutions:**
1. Use fewer documents for testing
2. Skip Bottom layer initially
3. Disable `-ingraphmerge` flag
4. Add more API keys for parallelization

## 📚 Next Steps

Now that you have CVDGraphRAG running, explore:

1. **[Building Knowledge Graph](building_graph.md)** - Learn advanced import options
2. **[Running Inference](running_inference.md)** - Explore retrieval modes
3. **[Using the Chatbot](using_chatbot.md)** - Master the chat interface
4. **[Performance Optimization](performance_optimization.md)** - Speed up your system

## 🆘 Getting Help

- Check [Troubleshooting Guide](troubleshooting.md)
- Review [API Documentation](../api/improved_retrieve.md)
- See [Architecture Overview](../architecture/system_components.md)
- Open GitHub Issue: https://github.com/datmieu204/CVDGraphRAG/issues

---

**Last Updated:** December 2024  
**Estimated Setup Time:** 30-60 minutes
