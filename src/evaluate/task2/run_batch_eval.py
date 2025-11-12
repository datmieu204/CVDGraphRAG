#!/usr/bin/env python3
"""
Batch evaluation runner for Task 2 Medical QnA
Evaluates all questions from the test dataset
"""

import sys
import os
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load .env from project root BEFORE importing any modules
from dotenv import load_dotenv
load_dotenv('/home/medgraph/.env')

import argparse
from evaluate.task2.qna_evaluator import MedicalQAEvaluator
from evaluate.task2.dataset_loader import load_medgraph_qna_dataset, QnADataset
from logger_ import get_logger

logger = get_logger("batch_eval", log_file="logs/evaluate/batch_eval.log")


def main():
    parser = argparse.ArgumentParser(description='Batch Medical QnA Evaluation')
    
    # Dataset options
    parser.add_argument('--questions', type=str, 
                       default='/home/medgraph/qna/questions_en.txt',
                       help='Questions file path')
    parser.add_argument('--answers', type=str,
                       default='/home/medgraph/qna/answers_en.txt',
                       help='Ground truth answers file path')
    parser.add_argument('--limit', type=int, 
                       help='Limit number of questions to evaluate')
    parser.add_argument('--start', type=int, default=0,
                       help='Start index (for partial evaluation)')
    
    # Output options
    parser.add_argument('--output', type=str, 
                       default='results/task2_batch_evaluation.json',
                       help='Output JSON file')
    
    # Neo4j config
    parser.add_argument('--neo4j-url', type=str,
                       default=os.getenv('NEO4J_URI', 'bolt://localhost:7687'))
    parser.add_argument('--neo4j-username', type=str,
                       default=os.getenv('NEO4J_USERNAME', 'neo4j'))
    parser.add_argument('--neo4j-password', type=str,
                       default=os.getenv('NEO4J_PASSWORD'))
    
    args = parser.parse_args()
    
    if not args.neo4j_password:
        logger.error("❌ NEO4J_PASSWORD required")
        return 1
    
    # Load dataset
    logger.info("="*80)
    logger.info("Loading MedGraph QnA Test Dataset")
    logger.info("="*80)
    
    try:
        dataset = QnADataset(args.questions, args.answers)
        logger.info(f"✅ Loaded {len(dataset)} question-answer pairs")
        logger.info(f"   Questions: {args.questions}")
        logger.info(f"   Answers:   {args.answers}")
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        return 1
    
    # Apply start/limit for partial evaluation
    all_pairs = list(dataset)
    eval_pairs = all_pairs[args.start:]
    if args.limit:
        eval_pairs = eval_pairs[:args.limit]
    
    logger.info(f"\n📊 Will evaluate {len(eval_pairs)} questions")
    if args.start > 0 or args.limit:
        logger.info(f"   Range: [{args.start}:{args.start + len(eval_pairs)}] of {len(dataset)} total")
    
    # Initialize evaluator
    logger.info("\n" + "="*80)
    logger.info("Initializing Medical QA Evaluator")
    logger.info("="*80)
    
    evaluator = MedicalQAEvaluator(
        args.neo4j_url,
        args.neo4j_username,
        args.neo4j_password
    )
    
    # Run evaluation with ground truth
    logger.info("\n" + "="*80)
    logger.info("Starting Batch Evaluation")
    logger.info("="*80)
    
    results = []
    for i, (question, ground_truth) in enumerate(eval_pairs, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Question {i}/{len(eval_pairs)} (Index {args.start + i - 1})")
        logger.info(f"{'='*80}")
        
        result = evaluator.evaluate_qa_pair(question, ground_truth=ground_truth)
        results.append(result)
    
    # Compute aggregate statistics
    valid_results = [r for r in results if r.get('metrics')]
    if valid_results:
        import numpy as np
        avg_metrics = {}
        for key in valid_results[0]['metrics'].keys():
            values = [r['metrics'][key] for r in valid_results]
            avg_metrics[f'avg_{key}'] = float(np.mean(values))
            avg_metrics[f'std_{key}'] = float(np.std(values))
        
        logger.info("\n" + "="*80)
        logger.info("📊 AGGREGATE RESULTS")
        logger.info("="*80)
        for key in sorted([k for k in avg_metrics.keys() if k.startswith('avg_')]):
            std_key = key.replace('avg_', 'std_')
            logger.info(f"  {key[4:]:.<40} {avg_metrics[key]:.3f} ± {avg_metrics[std_key]:.3f}")
    
    # Save results
    import json
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({
            'dataset_info': {
                'questions_file': args.questions,
                'answers_file': args.answers,
                'total_questions': len(dataset),
                'evaluated_count': len(eval_pairs),
                'start_index': args.start
            },
            'results': results,
            'aggregate': avg_metrics if valid_results else {}
        }, f, indent=2, ensure_ascii=False)
    
    logger.info("\n" + "="*80)
    logger.info("✅ Batch evaluation completed!")
    logger.info(f"📁 Results saved to: {args.output}")
    logger.info("="*80)
    
    return 0


if __name__ == '__main__':
    exit(main())
