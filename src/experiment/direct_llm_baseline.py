import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv('/home/medgraph/.env')

import argparse
import json
from typing import List, Tuple, Dict
from dataloader import load_high
from dedicated_key_manager import create_dedicated_client
from logger_ import get_logger

logger = get_logger("direct_llm", log_file="logs/experiment/direct_llm.log")


def answer_with_direct_llm(question: str, client=None, model: str = "models/gemini-2.5-flash-lite") -> str:
    """
    Answer medical question using LLM directly without any retrieval
    
    Args:
        question: Medical question
        client: DedicatedKeyClient instance
        model: Gemini model to use
    
    Returns:
        Generated answer
    """
    if client is None:
        client = create_dedicated_client(task_id="direct_llm_baseline")
    
    system_prompt = """You are a medical expert assistant. Answer the following medical question accurately and concisely based on your medical knowledge. Provide clear explanations when needed."""
    
    full_prompt = f"{system_prompt}\n\nQuestion: {question}\n\nAnswer:"
    
    try:
        response = client.call_with_retry(
            full_prompt,
            model=model,
            max_retries=3
        )
        return response.strip()
    
    except Exception as e:
        logger.error(f"Failed to get response: {e}")
        return ""


def evaluate_direct_llm(
    questions: List[str],
    ground_truths: List[str],
    output_file: str = None,
    model: str = "models/gemini-2.5-flash-lite"
) -> List[Dict]:
    """
    Evaluate Direct LLM baseline on medical QA dataset
    
    Args:
        questions: List of questions
        ground_truths: List of ground truth answers
        output_file: Path to save results
        model: Gemini model to use
    
    Returns:
        List of result dictionaries
    """
    logger.info(f"Model: {model}")
    logger.info(f"Questions: {len(questions)}")
    
    client = create_dedicated_client(task_id="direct_llm_evaluation")
    results = []
    
    for i, (question, ground_truth) in enumerate(zip(questions, ground_truths), 1):
        logger.info(f"Question {i}/{len(questions)}")
        logger.info(f"\nQuestion: {question}")
        
        # Get answer
        answer = answer_with_direct_llm(question, client, model)
        
        logger.info(f"\nGenerated Answer:\n{answer}")
        logger.info(f"\nGround Truth:\n{ground_truth}")
        
        results.append({
            'question': question,
            'answer': answer,
            'ground_truth': ground_truth,
            'model': model,
            'method': 'Direct LLM (No RAG)'
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
    parser = argparse.ArgumentParser(description='Direct LLM Baseline Evaluation')
    parser.add_argument('--questions', type=str,
                       default='/home/medgraph/qna/quang/questions.txt',
                       help='Path to questions file')
    parser.add_argument('--answers', type=str,
                       default='/home/medgraph/qna/quang/answers_with_explain.txt',
                       help='Path to ground truth answers')
    parser.add_argument('--output', type=str,
                       default='results/direct_llm_results.json',
                       help='Output file path')
    parser.add_argument('--model', type=str,
                       default='models/gemini-2.5-flash-lite',
                       help='Gemini model to use')
    parser.add_argument('--limit', type=int,
                       help='Limit number of questions')
    
    args = parser.parse_args()
    
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
    
    evaluate_direct_llm(questions, answers, args.output, args.model)
    
    return 0

if __name__ == '__main__':
    exit(main())
