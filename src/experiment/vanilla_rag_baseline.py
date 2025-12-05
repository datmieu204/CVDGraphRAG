"""
Vanilla RAG Baseline - Simple Vector Retrieval
Uses embedding similarity to retrieve chunks without graph structure
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv('/home/medgraph/.env')

import argparse
import json
import numpy as np
from typing import List, Dict, Tuple
from dataloader import load_high
from utils import get_embedding
from dedicated_key_manager import create_dedicated_client
from logger_ import get_logger
from camel.storages import Neo4jGraph

logger = get_logger("vanilla_rag", log_file="logs/experiment/vanilla_rag.log")


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Compute cosine similarity between two vectors
    """
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot / (norm1 * norm2)


def retrieve_vanilla(n4j, question: str, top_k: int = 5) -> List[Dict]:
    """
    Vanilla RAG retrieval using pure vector similarity
    
    Args:
        n4j: Neo4j connection
        question: User question
        top_k: Number of chunks to retrieve
    
    Returns:
        List of retrieved chunks with scores
    """
    logger.info(f"Retrieving top-{top_k} chunks for question...")
    
    q_emb = get_embedding(question)
    
    query = """
    MATCH (n)
    WHERE n.embedding IS NOT NULL
      AND n.description IS NOT NULL
      AND NOT n:Summary
    RETURN n.id as id, n.description as description, n.embedding as embedding
    LIMIT 1000
    """
    
    nodes = n4j.query(query)
    
    if not nodes:
        logger.warning("No nodes with embeddings found")
        return []
    
    logger.info(f"Found {len(nodes)} candidate chunks")
    
    scored_nodes = []
    for node in nodes:
        if not node['embedding']:
            continue
        
        sim = cosine_similarity(q_emb, node['embedding'])
        scored_nodes.append({
            'id': node['id'],
            'description': node['description'],
            'score': float(sim)
        })
    
    scored_nodes.sort(key=lambda x: x['score'], reverse=True)
    
    top_chunks = scored_nodes[:top_k]
    
    logger.info(f"Retrieved {len(top_chunks)} chunks")
    for i, chunk in enumerate(top_chunks, 1):
        logger.info(f"  [{i}] Score: {chunk['score']:.3f} - {chunk['id'][:50]}...")
    
    return top_chunks


def answer_with_vanilla_rag(
    n4j,
    question: str,
    client=None,
    top_k: int = 5,
    model: str = "models/gemini-2.5-flash-lite"
) -> Tuple[str, List[Dict]]:
    """
    Answer question using vanilla RAG
    
    Args:
        n4j: Neo4j connection
        question: Medical question
        client: DedicatedKeyClient
        top_k: Number of chunks to retrieve
        model: Gemini model to use
    
    Returns:
        (answer, retrieved_chunks)
    """
    if client is None:
        client = create_dedicated_client(task_id="vanilla_rag_baseline")
    
    chunks = retrieve_vanilla(n4j, question, top_k)
    
    if not chunks:
        logger.warning("No chunks retrieved, using direct LLM")
        return "", []
    
    context = "\n\n".join([
        f"[{i}] {chunk['description']}"
        for i, chunk in enumerate(chunks, 1)
    ])
    
    system_prompt = """You are a medical expert assistant. Answer the question using the provided context. Include citations [1], [2], etc. when using information from the context."""
    
    full_prompt = f"""{system_prompt}

Context:
{context}

Question: {question}

Answer:"""
    
    try:
        response = client.call_with_retry(
            full_prompt,
            model=model,
            max_retries=3
        )
        return response.strip(), chunks
    
    except Exception as e:
        logger.error(f"Failed to generate answer: {e}")
        return "", chunks


def evaluate_vanilla_rag(
    questions: List[str],
    ground_truths: List[str],
    neo4j_url: str,
    neo4j_username: str,
    neo4j_password: str,
    output_file: str = None,
    top_k: int = 5,
    model: str = "models/gemini-2.5-flash-lite"
) -> List[Dict]:
    """
    Evaluate Vanilla RAG baseline
    
    Args:
        questions: List of questions
        ground_truths: List of ground truth answers
        neo4j_url, neo4j_username, neo4j_password: Neo4j connection
        output_file: Path to save results
        top_k: Number of chunks to retrieve
        model: Gemini model to use
    
    Returns:
        List of result dictionaries
    """
    logger.info("="*80)
    logger.info("Vanilla RAG Baseline Evaluation")
    logger.info(f"Model: {model}")
    logger.info(f"Top-K: {top_k}")
    logger.info(f"Questions: {len(questions)}")
    logger.info("="*80)
    
    # Connect to Neo4j
    n4j = Neo4jGraph(
        url=neo4j_url,
        username=neo4j_username,
        password=neo4j_password
    )
    logger.info("Connected to Neo4j")
    
    client = create_dedicated_client(task_id="vanilla_rag_evaluation")
    results = []
    
    for i, (question, ground_truth) in enumerate(zip(questions, ground_truths), 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Question {i}/{len(questions)}")
        logger.info(f"{'='*80}")
        logger.info(f"\nQuestion: {question}")
        
        # Get answer
        answer, chunks = answer_with_vanilla_rag(n4j, question, client, top_k, model)
        
        logger.info(f"\nGenerated Answer:\n{answer}")
        logger.info(f"\nGround Truth:\n{ground_truth}")
        
        results.append({
            'question': question,
            'answer': answer,
            'ground_truth': ground_truth,
            'retrieved_chunks': [
                {'id': c['id'], 'score': c['score']}
                for c in chunks
            ],
            'model': model,
            'top_k': top_k,
            'method': 'Vanilla RAG'
        })
        
        if output_file and i % 10 == 0:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({'results': results}, f, indent=2, ensure_ascii=False)
            logger.info(f"Checkpoint saved: {i}/{len(questions)}")
    
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({'results': results}, f, indent=2, ensure_ascii=False)
        logger.info(f"\nFinal results saved to: {output_file}")
    
    logger.info("\nEvaluation completed!")
    return results


def main():
    parser = argparse.ArgumentParser(description='Vanilla RAG Baseline Evaluation')
    parser.add_argument('--questions', type=str,
                       default='/home/medgraph/qna/quang/questions.txt')
    parser.add_argument('--answers', type=str,
                       default='/home/medgraph/qna/quang/answers_with_explain.txt')
    parser.add_argument('--output', type=str,
                       default='results/vanilla_rag_results.json')
    parser.add_argument('--model', type=str,
                       default='models/gemini-2.5-flash-lite')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of chunks to retrieve')
    parser.add_argument('--limit', type=int,
                       help='Limit number of questions')
    
    parser.add_argument('--neo4j-url', type=str,
                       default=os.getenv('NEO4J_URI', 'bolt://localhost:7687'))
    parser.add_argument('--neo4j-username', type=str,
                       default=os.getenv('NEO4J_USERNAME', 'neo4j'))
    parser.add_argument('--neo4j-password', type=str,
                       default=os.getenv('NEO4J_PASSWORD'))
    
    args = parser.parse_args()
    
    if not args.neo4j_password:
        logger.error("NEO4J_PASSWORD required")
        return 1
    
    questions_text = load_high(args.questions)
    answers_text = load_high(args.answers)
    
    questions = [q.strip() for q in questions_text.strip().split('\n') if q.strip()]
    answers = [a.strip() for a in answers_text.strip().split('\n') if a.strip()]
    
    if len(questions) != len(answers):
        logger.error(f"Mismatch: {len(questions)} questions, {len(answers)} answers")
        return 1
    
    logger.info(f"Loaded {len(questions)} question-answer pairs")
    
    if args.limit:
        questions = questions[:args.limit]
        answers = answers[:args.limit]
        logger.info(f"Will evaluate {len(questions)} questions")
    
    # Evaluate
    evaluate_vanilla_rag(
        questions, answers,
        args.neo4j_url, args.neo4j_username, args.neo4j_password,
        args.output, args.top_k, args.model
    )
    
    return 0


if __name__ == '__main__':
    exit(main())
