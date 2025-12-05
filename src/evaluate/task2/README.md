# Task 2: Medical QnA Evaluation System

Hệ thống đánh giá chất lượng câu trả lời y khoa sử dụng Knowledge Graph + LLM với dataset test thực tế.

## 📊 Các chỉ số đánh giá (Metrics)

### 1. **Pertinence (Pert.)** - Độ phù hợp (15%)
- Đo lường mức độ liên quan giữa câu hỏi và câu trả lời
- Scale: 0-1 (1 = hoàn toàn phù hợp)
- Method: LLM-as-a-judge evaluation

### 2. **Correctness (Cor.)** - Độ chính xác (25%) ⭐
- Đánh giá tính chính xác của nội dung về mặt y học
- Scale: 0-1 (1 = hoàn toàn chính xác)
- **Trọng số cao nhất** vì tính mạng y khoa
- Method: LLM evaluation với KG context làm ground truth

### 3. **Citation Precision (CP)** - Độ chính xác trích dẫn (10%)
- Tỷ lệ các entity được trích dẫn chính xác từ KG
- Scale: 0-1
- Formula: (Entities mentioned correctly) / (Total entities mentioned)
- Method: Rule-based entity matching

### 4. **Citation Recall (CR)** - Độ đầy đủ trích dẫn (10%)
- Khả năng tham chiếu đầy đủ các entities liên quan trong KG
- Scale: 0-1
- Formula: (Entities mentioned) / (Total relevant entities in KG)
- Method: Rule-based entity coverage

### 5. **Understandability (Und.)** - Tính dễ hiểu (15%)
- Khả năng diễn giải rõ ràng cho người không chuyên
- Scale: 0-1 (1 = rất dễ hiểu)
- Method: LLM evaluation for clarity

### 6. **Answer Consistency** - Tính nhất quán (10%)
- Đo tính logic và đồng nhất trong câu trả lời
- Scale: 0-1 (1 = hoàn toàn nhất quán, không mâu thuẫn)
- Method: LLM evaluation for logical consistency

### 7. **Faithfulness** - Tính trung thực (15%)
- Đảm bảo câu trả lời có căn cứ trong KG, không "hallucinate"
- Scale: 0-1 (1 = hoàn toàn dựa trên KG)
- Method: LLM evaluation with KG grounding check

### Overall Score - Điểm tổng hợp
Weighted average:
```
Overall = 0.15×Pert + 0.25×Cor + 0.10×CP + 0.10×CR + 0.15×Und + 0.10×Cons + 0.15×Faith
```

## 📁 Dataset

Dataset test được load từ:
- **Questions**: `/home/medgraph/qna/questions_en.txt` (42 câu hỏi)
- **Answers**: `/home/medgraph/qna/answers_en.txt` (42 câu trả lời ground truth)

Mỗi file có 1 câu hỏi/trả lời trên mỗi dòng, tương ứng 1-1.

## 🚀 Cách sử dụng

### 1. Test nhanh với câu hỏi đầu tiên

```bash
cd /home/medgraph/src/evaluate/task2
python quick_eval.py
```

### 2. Đánh giá toàn bộ dataset (42 câu hỏi)

```bash
cd /home/medgraph/src/evaluate/task2
python run_batch_eval.py
```

### 3. Đánh giá một phần dataset

```bash
# Đánh giá 5 câu hỏi đầu tiên
python run_batch_eval.py --limit 5

# Đánh giá từ câu thứ 10 đến 20
python run_batch_eval.py --start 10 --limit 10

# Đánh giá với custom output
python run_batch_eval.py --output results/my_evaluation.json
```

### 4. Đánh giá với custom dataset

```bash
python run_batch_eval.py \
  --questions /path/to/questions.txt \
  --answers /path/to/answers.txt \
  --output results/custom_eval.json
```

### 5. Test dataset loader

```bash
cd /home/medgraph/src/evaluate/task2
python dataset_loader.py
```

## 📂 Cấu trúc files

```
evaluate/task2/
├── qna_evaluator.py      # Main evaluator class với 7 metrics
├── dataset_loader.py     # Load questions + answers từ txt files
├── quick_eval.py         # Script test nhanh 1 câu hỏi
├── run_batch_eval.py     # Script đánh giá batch nhiều câu hỏi
└── README.md            # Documentation này
```

## 📊 Output Format

Kết quả được lưu dạng JSON:

```json
{
  "dataset_info": {
    "questions_file": "/home/medgraph/qna/questions_en.txt",
    "answers_file": "/home/medgraph/qna/answers_en.txt",
    "total_questions": 42,
    "evaluated_count": 42,
    "start_index": 0
  },
  "results": [
    {
      "question": "EMPULSE Trial: In acute heart failure...",
      "answer": "Based on the knowledge graph...",
      "gid": "abc123...",
      "ground_truth": "EMPULSE Trial: Acute Kidney Injury...",
      "metrics": {
        "pertinence": 0.95,
        "correctness": 0.88,
        "citation_precision": 0.75,
        "citation_recall": 0.82,
        "understandability": 0.90,
        "answer_consistency": 0.93,
        "faithfulness": 0.85,
        "overall_score": 0.872
      },
      "kg_context_summary": {
        "entity_count": 42,
        "sample_entities": ["Heart Failure", "Empagliflozin", "Acute Kidney Injury"]
      }
    }
  ],
  "aggregate": {
    "avg_pertinence": 0.89,
    "std_pertinence": 0.05,
    "avg_correctness": 0.85,
    "std_correctness": 0.07,
    "avg_overall_score": 0.867,
    "std_overall_score": 0.04
  }
}
```

## 🔄 Pipeline

```
┌─────────────┐
│  Question   │ (from questions_en.txt)
└──────┬──────┘
       │
       ├─ 1. Summarize question (process_chunks)
       │
       ├─ 2. Retrieve relevant KG subgraph (seq_ret) → GID
       │
       ├─ 3. Extract KG context (entities + relationships)
       │
       ├─ 4. Generate answer with LLM + KG context
       │
       └─ 5. Evaluate with 7 metrics
              │
              ├─ Pertinence (LLM judge)
              ├─ Correctness (LLM + KG verification)
              ├─ Citation Precision (entity matching)
              ├─ Citation Recall (entity coverage)
              ├─ Understandability (LLM judge)
              ├─ Consistency (LLM judge)
              └─ Faithfulness (KG grounding)
              
       Compare with Ground Truth (answers_en.txt)
```

## ⚙️ Dependencies

- **Knowledge Graph**: Neo4j với entity embeddings
- **LLM**: Gemini-2.0-flash (68 API keys với rotation)
- **Embeddings**: bge-m3 (1024-dim)
- **Evaluation**: LLM-as-a-judge + rule-based metrics
- **Dataset Loader**: dataloader.py (load_high function)

## 📝 Logs

- Main log: `logs/evaluate/task2_qna.log`
- Batch log: `logs/evaluate/batch_eval.log`

## 🎯 Example Results

```bash
$ python quick_eval.py

Loading MedGraph QnA dataset...
✅ Loaded 42 question-answer pairs

================================================================================
Quick Medical QA Evaluation Test
================================================================================

📝 Question:
EMPULSE Trial: In acute heart failure with concomitant acute kidney injury (AKI), does empagliflozin worsen renal function and electrolyte balance, leading to Major Adverse Cardiovascular Events (MACE)?

🎯 Ground Truth:
EMPULSE Trial: Acute Kidney Injury (AKI) is not a contraindication, but empagliflozin should not be initiated hastily in hemodynamically unstable patients...

💬 Generated Answer:
Based on the EMPULSE trial data in the knowledge graph, empagliflozin does not significantly worsen renal function in acute heart failure patients with AKI...

📊 EVALUATION RESULTS
================================================================================
  Pertinence..................................... 0.950
  Correctness.................................... 0.880
  Citation Precision............................. 0.750
  Citation Recall................................ 0.820
  Understandability.............................. 0.900
  Answer Consistency............................. 0.930
  Faithfulness................................... 0.850
--------------------------------------------------------------------------------
  Overall Score.................................. 0.872
================================================================================
```

## 🔧 Configuration

Set environment variables in `.env`:
```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
GEMINI_API_KEY_1=your_key_1
GEMINI_API_KEY_2=your_key_2
...
```

## 📈 Performance

- Average evaluation time: ~30-60s per question
- LLM calls per question: ~7-10 (depending on metrics)
- Rate limiting: Auto-managed với 68 API keys
- Memory usage: ~2-4GB (embedding model + KG queries)

## 🚨 Troubleshooting

**Error: No relevant knowledge graph found**
- Check if Neo4j has data imported
- Verify KG has Middle layer with medical entities

**Error: All API keys exhausted**
- Wait 24 hours for quota reset
- Add more Gemini API keys to .env

**Error: Dataset mismatch**
- Ensure questions_en.txt and answers_en.txt have same number of lines
- Check file encoding (UTF-8)
