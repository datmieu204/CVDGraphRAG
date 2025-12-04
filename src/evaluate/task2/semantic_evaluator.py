#!/usr/bin/env python3
"""
Semantic-Only Evaluator
Chỉ dùng BGE embeddings để tính similarity - không regex, không hardcoded rules
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv('/home/medgraph/.env')

import argparse
import json
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu

from camel.storages import Neo4jGraph
from dataloader import load_high
from summerize import process_chunks
from retrieve import seq_ret
from utils import get_response, get_bge_m3_embedding
from dedicated_key_manager import create_dedicated_client
from logger_ import get_logger

logger = get_logger("semantic_eval", log_file="logs/evaluate/semantic_eval.log")


@dataclass
class SemanticMetrics:
    """Enhanced semantic metrics with BGE + ROUGE + BLEU"""
    answer_similarity: float  # 0-1: BGE similarity between predicted & ground truth
    question_answer_relevance: float  # 0-1: BGE similarity between question & answer
    rouge_1: float  # 0-1: ROUGE-1 F1 score
    rouge_2: float  # 0-1: ROUGE-2 F1 score
    rouge_l: float  # 0-1: ROUGE-L F1 score
    bleu: float  # 0-1: BLEU score
    
    def overall_score(self) -> float:
        """Weighted average: BGE (40%) + ROUGE (30%) + BLEU (20%) + Q-A Relevance (10%)"""
        avg_rouge = (self.rouge_1 + self.rouge_2 + self.rouge_l) / 3.0
        return (
            0.4 * self.answer_similarity + 
            0.3 * avg_rouge + 
            0.2 * self.bleu + 
            0.1 * self.question_answer_relevance
        )
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['overall_score'] = self.overall_score()
        return result


class SemanticEvaluator:
    """Evaluator using BGE embeddings + ROUGE + BLEU metrics"""
    
    def __init__(self, neo4j_url: str, neo4j_username: str, neo4j_password: str):
        logger.info("\n" + "="*80)
        logger.info("Semantic Evaluator (BGE + ROUGE + BLEU)")
        logger.info("="*80)
        
        self.n4j = Neo4jGraph(
            url=neo4j_url,
            username=neo4j_username,
            password=neo4j_password
        )
        logger.info("✅ Connected to Neo4j")
        
        self.client = create_dedicated_client(task_id="semantic_evaluator")
        logger.info("✅ Client initialized")
    
    def get_answer_from_system(self, question: str) -> Tuple[str, str]:
        """
        Get answer from QA system
        Returns: (answer, gid)
        """
        logger.info(f"\n📝 Question: {question}")
        
        # Summarize
        logger.info("  [1/3] Summarizing...")
        summaries = process_chunks(question, client=self.client)
        summary = " ".join(summaries) if summaries else question
        
        # Retrieve GID
        logger.info("  [2/3] Retrieving GID...")
        gid = seq_ret(self.n4j, summary, client=self.client)
        
        if gid is None:
            logger.error("  ❌ No GID found")
            return "", ""
        
        logger.info(f"  ✅ GID: {gid}")
        
        # Generate answer
        logger.info("  [3/3] Generating answer...")
        answer = get_response(self.n4j, gid, question, client=self.client)
        
        return answer, gid
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity using BGE embeddings"""
        try:
            emb1 = get_bge_m3_embedding(text1)
            emb2 = get_bge_m3_embedding(text2)
            
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return float(np.clip(similarity, 0.0, 1.0))
        except Exception as e:
            logger.error(f"  ❌ Embedding error: {e}")
            return 0.0
    
    def compute_rouge_scores(self, prediction: str, reference: str) -> Dict[str, float]:
        """Compute ROUGE scores"""
        try:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(reference, prediction)
            return {
                'rouge_1': scores['rouge1'].fmeasure,
                'rouge_2': scores['rouge2'].fmeasure,
                'rouge_l': scores['rougeL'].fmeasure
            }
        except Exception as e:
            logger.error(f"  ❌ ROUGE error: {e}")
            return {'rouge_1': 0.0, 'rouge_2': 0.0, 'rouge_l': 0.0}
    
    def compute_bleu_score(self, prediction: str, reference: str) -> float:
        """Compute BLEU score"""
        try:
            bleu = corpus_bleu([prediction], [[reference]])
            return bleu.score / 100.0  # Normalize to 0-1
        except Exception as e:
            logger.error(f"  ❌ BLEU error: {e}")
            return 0.0
    
    def evaluate_qa_pair(self, question: str, ground_truth: str) -> Dict:
        """Evaluate single QA pair"""
        logger.info("\n" + "="*80)
        logger.info("Evaluation Pipeline")
        logger.info("="*80)
        
        # Get answer
        answer, gid = self.get_answer_from_system(question)
        
        if not answer:
            logger.error("❌ Failed to get answer")
            return {
                'question': question,
                'answer': '',
                'error': 'Failed to generate answer',
                'metrics': None
            }
        
        logger.info(f"\n💬 Generated Answer:\n{answer}\n")
        logger.info(f"🎯 Ground Truth:\n{ground_truth}\n")
        
        # Compute metrics
        logger.info("="*80)
        logger.info("Computing Evaluation Metrics")
        logger.info("="*80)
        
        logger.info("\n[1/4] Answer Similarity (BGE: Predicted vs Ground Truth)...")
        answer_sim = self.compute_similarity(answer, ground_truth)
        logger.info(f"  ✅ Similarity: {answer_sim:.3f}")
        
        logger.info("\n[2/4] Question-Answer Relevance (BGE)...")
        qa_relevance = self.compute_similarity(question, answer)
        logger.info(f"  ✅ Relevance: {qa_relevance:.3f}")
        
        logger.info("\n[3/4] ROUGE Scores...")
        rouge_scores = self.compute_rouge_scores(answer, ground_truth)
        logger.info(f"  ✅ ROUGE-1: {rouge_scores['rouge_1']:.3f}")
        logger.info(f"  ✅ ROUGE-2: {rouge_scores['rouge_2']:.3f}")
        logger.info(f"  ✅ ROUGE-L: {rouge_scores['rouge_l']:.3f}")
        
        logger.info("\n[4/4] BLEU Score...")
        bleu_score = self.compute_bleu_score(answer, ground_truth)
        logger.info(f"  ✅ BLEU: {bleu_score:.3f}")
        
        # Create metrics
        metrics = SemanticMetrics(
            answer_similarity=answer_sim,
            question_answer_relevance=qa_relevance,
            rouge_1=rouge_scores['rouge_1'],
            rouge_2=rouge_scores['rouge_2'],
            rouge_l=rouge_scores['rouge_l'],
            bleu=bleu_score
        )
        
        logger.info(f"\n🎯 Overall Score: {metrics.overall_score():.3f}")
        
        return {
            'question': question,
            'answer': answer,
            'ground_truth': ground_truth,
            'gid': gid,
            'metrics': metrics.to_dict()
        }
    
    def evaluate_dataset(self, pairs: List[Tuple[str, str]], output_file: str = None) -> List[Dict]:
        """Evaluate multiple QA pairs"""
        logger.info(f"\n📊 Evaluating {len(pairs)} questions...")
        
        results = []
        for i, (question, ground_truth) in enumerate(pairs, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"Question {i}/{len(pairs)}")
            logger.info(f"{'='*80}")
            
            result = self.evaluate_qa_pair(question, ground_truth)
            results.append(result)
        
        # Aggregate
        valid_results = [r for r in results if r.get('metrics')]
        
        if valid_results:
            avg_metrics = {}
            for key in valid_results[0]['metrics'].keys():
                values = [r['metrics'][key] for r in valid_results]
                avg_metrics[f'avg_{key}'] = float(np.mean(values))
                avg_metrics[f'std_{key}'] = float(np.std(values))
                avg_metrics[f'min_{key}'] = float(np.min(values))
                avg_metrics[f'max_{key}'] = float(np.max(values))
            
            logger.info("\n" + "="*80)
            logger.info("📈 AGGREGATE RESULTS")
            logger.info("="*80)
            for key in sorted([k for k in avg_metrics.keys() if k.startswith('avg_')]):
                metric_name = key[4:]
                logger.info(
                    f"  {metric_name:.<35} "
                    f"{avg_metrics[key]:.3f} ± {avg_metrics[f'std_{metric_name}']:.3f}"
                )
        
        # Save
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'results': results,
                    'aggregate': avg_metrics if valid_results else {}
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"\n💾 Saved to: {output_file}")
        
        logger.info("\n✅ Evaluation completed!")
        return results


def main():
    parser = argparse.ArgumentParser(description='Semantic QA Evaluation')
    parser.add_argument('--questions', type=str, 
                       default='/home/medgraph/qna/quang/questions.txt')
    parser.add_argument('--answers', type=str,
                       default='/home/medgraph/qna/quang/answers_with_explain.txt')
    parser.add_argument('--limit', type=int, help='Limit number of questions')
    parser.add_argument('--output', type=str, 
                       default='results/semantic_eval.json')
    
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
    logger.info("Loading Dataset")
    logger.info("="*80)
    
    questions_text = load_high(args.questions)
    answers_text = load_high(args.answers)
    
    questions = [q.strip() for q in questions_text.strip().split('\n') if q.strip()]
    answers = [a.strip() for a in answers_text.strip().split('\n') if a.strip()]
    
    if len(questions) != len(answers):
        logger.error(f"❌ Mismatch: {len(questions)} questions, {len(answers)} answers")
        return 1
    
    pairs = list(zip(questions, answers))
    logger.info(f"✅ Loaded {len(pairs)} pairs")
    
    if args.limit:
        pairs = pairs[:args.limit]
        logger.info(f"📊 Will evaluate {len(pairs)} questions")
    
    # Evaluate
    evaluator = SemanticEvaluator(
        args.neo4j_url,
        args.neo4j_username,
        args.neo4j_password
    )
    
    evaluator.evaluate_dataset(pairs, output_file=args.output)
    
    return 0


if __name__ == '__main__':
    exit(main())