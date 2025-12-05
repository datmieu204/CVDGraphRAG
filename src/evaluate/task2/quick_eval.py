#!/usr/bin/env python3
"""
Quick evaluation script for medical QA system
Tests with first question from the real dataset
"""

import sys
import os
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load .env from project root BEFORE importing any modules
from dotenv import load_dotenv
load_dotenv('/home/medgraph/.env')

from evaluate.task2.qna_evaluator import MedicalQAEvaluator
from evaluate.task2.dataset_loader import load_medgraph_qna_dataset

def main():
    # Load dataset
    print("Loading MedGraph QnA dataset...")
    dataset = load_medgraph_qna_dataset()
    print(f"✅ Loaded {len(dataset)} question-answer pairs\n")
    
    # Get first question
    question, ground_truth = dataset[0]
    
    # Initialize evaluator
    evaluator = MedicalQAEvaluator(
        neo4j_url=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        neo4j_username=os.getenv('NEO4J_USERNAME', 'neo4j'),
        neo4j_password=os.getenv('NEO4J_PASSWORD')
    )
    
    print("\n" + "="*80)
    print("Quick Medical QA Evaluation Test")
    print("="*80)
    print(f"\n📝 Question:\n{question}\n")
    if ground_truth:
        print(f"🎯 Ground Truth:\n{ground_truth[:200]}...")
    print()
    
    # Evaluate
    result = evaluator.evaluate_qa_pair(question, ground_truth=ground_truth)
    
    if result.get('metrics'):
        print("\n" + "="*80)
        print("📊 EVALUATION RESULTS")
        print("="*80)
        print(f"\n💬 Answer:\n{result['answer']}\n")
        print("\n📈 Metrics:")
        print("-" * 80)
        for metric, value in result['metrics'].items():
            if metric != 'overall_score':
                print(f"  {metric.replace('_', ' ').title():.<40} {value:.3f}")
        print("-" * 80)
        print(f"  {'Overall Score':.<40} {result['metrics']['overall_score']:.3f}")
        print("="*80)
    else:
        print(f"\n❌ Error: {result.get('error', 'Unknown error')}")

if __name__ == '__main__':
    main()
