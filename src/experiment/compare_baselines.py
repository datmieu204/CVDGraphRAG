import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv('/home/medgraph/.env')

import argparse
import json
import numpy as np
from typing import List, Dict
from dataloader import load_high
from logger_ import get_logger

from direct_llm_baseline import evaluate_direct_llm
from vanilla_rag_baseline import evaluate_vanilla_rag

sys.path.insert(0, str(Path(__file__).parent.parent / 'evaluate' / 'task2'))
from semantic_evaluator import SemanticEvaluator

logger = get_logger("comparison", log_file="logs/experiment/comparison.log")


def compute_metrics_from_results(results: List[Dict], evaluator) -> Dict:
    """
    Compute evaluation metrics from raw results
    
    Args:
        results: List of result dicts with 'answer' and 'ground_truth'
        evaluator: SemanticEvaluator instance for metric computation
    
    Returns:
        Aggregate metrics dict
    """
    logger.info(f"\nComputing metrics for {len(results)} results...")
    
    metrics_list = []
    
    for i, result in enumerate(results, 1):
        question = result['question']
        answer = result['answer']
        ground_truth = result['ground_truth']
        
        if not answer:
            logger.warning(f"Skipping question {i}: empty answer")
            continue
        
        answer_sim = evaluator.compute_similarity(answer, ground_truth)
        
        qa_rel = evaluator.compute_similarity(question, answer)
        
        rouge = evaluator.compute_rouge_scores(answer, ground_truth)
        
        bleu = evaluator.compute_bleu_score(answer, ground_truth)
        
        metrics_list.append({
            'answer_similarity': answer_sim,
            'question_answer_relevance': qa_rel,
            'rouge_1': rouge['rouge_1'],
            'rouge_2': rouge['rouge_2'],
            'rouge_l': rouge['rouge_l'],
            'bleu': bleu,
            'overall_score': (
                0.4 * answer_sim +
                0.3 * (rouge['rouge_1'] + rouge['rouge_2'] + rouge['rouge_l']) / 3.0 +
                0.2 * bleu +
                0.1 * qa_rel
            )
        })
    
    if not metrics_list:
        logger.error("No valid metrics computed")
        return {}
    
    aggregate = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list]
        aggregate[f'avg_{key}'] = float(np.mean(values))
        aggregate[f'std_{key}'] = float(np.std(values))
        aggregate[f'min_{key}'] = float(np.min(values))
        aggregate[f'max_{key}'] = float(np.max(values))
    
    logger.info("Metrics computed")
    return aggregate


def compare_all_methods(
    questions: List[str],
    ground_truths: List[str],
    neo4j_url: str,
    neo4j_username: str,
    neo4j_password: str,
    output_dir: str = "results/comparison",
    model: str = "models/gemini-2.5-flash-lite"
):
    """
    Run and compare all baseline methods
    
    Args:
        questions: List of questions
        ground_truths: List of ground truth answers
        neo4j_url, neo4j_username, neo4j_password: Neo4j connection
        output_dir: Directory to save results
        model: Gemini model to use
    """
    os.makedirs(output_dir, exist_ok=True)
    

    logger.info(f"Questions: {len(questions)}")
    logger.info(f"Model: {model}")
    logger.info(f"Output: {output_dir}")
    
    evaluator = SemanticEvaluator(neo4j_url, neo4j_username, neo4j_password)
    
    all_results = {}

    
    direct_results = evaluate_direct_llm(
        questions, ground_truths,
        output_file=f"{output_dir}/direct_llm_raw.json",
        model=model
    )
    
    direct_metrics = compute_metrics_from_results(direct_results, evaluator)
    all_results['Direct LLM'] = {
        'metrics': direct_metrics,
        'method': 'No RAG - Pure LLM'
    }
    
    vanilla_results = evaluate_vanilla_rag(
        questions, ground_truths,
        neo4j_url, neo4j_username, neo4j_password,
        output_file=f"{output_dir}/vanilla_rag_raw.json",
        top_k=5,
        model=model
    )
    
    vanilla_metrics = compute_metrics_from_results(vanilla_results, evaluator)
    all_results['Vanilla RAG'] = {
        'metrics': vanilla_metrics,
        'method': 'Vector Similarity Retrieval'
    }
    
    cvd_results = evaluator.evaluate_dataset(
        list(zip(questions, ground_truths)),
        output_file=f"{output_dir}/cvdgraphrag_raw.json"
    )
    
    cvd_valid = [r for r in cvd_results if r.get('metrics')]
    if cvd_valid:
        cvd_metrics = {}
        for key in cvd_valid[0]['metrics'].keys():
            values = [r['metrics'][key] for r in cvd_valid]
            cvd_metrics[f'avg_{key}'] = float(np.mean(values))
            cvd_metrics[f'std_{key}'] = float(np.std(values))
    else:
        cvd_metrics = {}
    
    all_results['CVDGraphRAG'] = {
        'metrics': cvd_metrics,
        'method': 'Three-Layer Graph RAG with Smart Linking'
    }
    
    comparison_table = []
    
    for method, data in all_results.items():
        metrics = data['metrics']
        if not metrics:
            continue
        
        row = {
            'Method': method,
            'Overall Score': f"{metrics.get('avg_overall_score', 0):.3f}",
            'Answer Similarity': f"{metrics.get('avg_answer_similarity', 0):.3f}",
            'Q-A Relevance': f"{metrics.get('avg_question_answer_relevance', 0):.3f}",
            'ROUGE-1': f"{metrics.get('avg_rouge_1', 0):.3f}",
            'ROUGE-L': f"{metrics.get('avg_rouge_l', 0):.3f}",
            'BLEU': f"{metrics.get('avg_bleu', 0):.3f}"
        }
        comparison_table.append(row)
    
    logger.info("\n" + "-"*120)
    header = f"{'Method':<25} {'Overall':<12} {'Ans Sim':<12} {'Q-A Rel':<12} {'ROUGE-1':<12} {'ROUGE-L':<12} {'BLEU':<12}"
    logger.info(header)
    logger.info("-"*120)
    
    for row in comparison_table:
        line = f"{row['Method']:<25} {row['Overall Score']:<12} {row['Answer Similarity']:<12} {row['Q-A Relevance']:<12} {row['ROUGE-1']:<12} {row['ROUGE-L']:<12} {row['BLEU']:<12}"
        logger.info(line)
    
    logger.info("-"*120)
    
    comparison_file = f"{output_dir}/comparison_summary.json"
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump({
            'results': all_results,
            'table': comparison_table
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nComparison results saved to: {comparison_file}")
    logger.info("\nAll experiments completed!")


def main():
    parser = argparse.ArgumentParser(description='Compare All Baseline Methods')
    parser.add_argument('--questions', type=str,
                       default='/home/medgraph/qna/quang/questions.txt')
    parser.add_argument('--answers', type=str,
                       default='/home/medgraph/qna/quang/answers_with_explain.txt')
    parser.add_argument('--output-dir', type=str,
                       default='results/comparison')
    parser.add_argument('--model', type=str,
                       default='models/gemini-2.5-flash-lite')
    parser.add_argument('--limit', type=int,
                       help='Limit number of questions for testing')
    
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
        logger.error(f" Mismatch: {len(questions)} questions, {len(answers)} answers")
        return 1
    
    logger.info(f"Loaded {len(questions)} question-answer pairs")
    
    if args.limit:
        questions = questions[:args.limit]
        answers = answers[:args.limit]
        logger.info(f"Will evaluate {len(questions)} questions")
    
    compare_all_methods(
        questions, answers,
        args.neo4j_url, args.neo4j_username, args.neo4j_password,
        args.output_dir, args.model
    )
    
    return 0


if __name__ == '__main__':
    exit(main())
