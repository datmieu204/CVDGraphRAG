#!/usr/bin/env python3
"""
Improved Semantic Evaluator
===========================
Uses improved U-Retrieval with:
1. Hybrid Retrieval (Vector + LLM Reranking)
2. Query-aware context ranking
3. Multi-subgraph aggregation

Author: CVDGraphRAG Team
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
from datetime import datetime
from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu

from camel.storages import Neo4jGraph
from dataloader import load_high
from improved_retrieve import hybrid_retrieve, get_improved_response, improved_seq_ret
from utils import get_bge_m3_embedding
from dedicated_key_manager import create_dedicated_client
from logger_ import get_logger

logger = get_logger("improved_eval", log_file="logs/evaluate/improved_eval.log")


@dataclass
class SemanticMetrics:
    """Enhanced semantic metrics"""
    answer_similarity: float
    question_answer_relevance: float
    rouge_1: float
    rouge_2: float
    rouge_l: float
    bleu: float
    
    def overall_score(self) -> float:
        """Weighted average"""
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


class ImprovedSemanticEvaluator:
    """Evaluator using improved U-Retrieval"""
    
    def __init__(self, neo4j_url: str, neo4j_username: str, neo4j_password: str):
        logger.info("\n" + "="*80)
        logger.info("Improved Semantic Evaluator (Hybrid Retrieval)")
        logger.info("="*80)
        
        self.n4j = Neo4jGraph(
            url=neo4j_url,
            username=neo4j_username,
            password=neo4j_password
        )
        logger.info("Connected to Neo4j")
        
        self.client = create_dedicated_client(task_id="improved_evaluator")
        logger.info("Client initialized")
        
        # Rouge scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
        )
    
    def get_answer_improved(self, question: str, use_multi_subgraph: bool = False) -> Tuple[str, str]:
        """
        Get answer using improved retrieval
        
        OPTIMIZED: Default to single subgraph for 10x speedup
        
        Returns: (answer, gid)
        """
        logger.info(f"\nQuestion: {question[:100]}...")
        
        answer, gid = get_improved_response(
            self.n4j, question, self.client,
            use_multi_subgraph=use_multi_subgraph,
            top_k_subgraphs=1  # Single subgraph for speed
        )
        
        return answer, gid
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity using BGE embeddings"""
        try:
            emb1 = get_bge_m3_embedding(text1)
            emb2 = get_bge_m3_embedding(text2)
            
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return float(np.clip(similarity, 0.0, 1.0))
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return 0.0
    
    def compute_rouge(self, prediction: str, reference: str) -> Dict[str, float]:
        """Compute ROUGE scores"""
        try:
            scores = self.rouge_scorer.score(reference, prediction)
            return {
                'rouge_1': scores['rouge1'].fmeasure,
                'rouge_2': scores['rouge2'].fmeasure,
                'rouge_l': scores['rougeL'].fmeasure
            }
        except Exception as e:
            logger.error(f"ROUGE error: {e}")
            return {'rouge_1': 0.0, 'rouge_2': 0.0, 'rouge_l': 0.0}
    
    def compute_bleu(self, prediction: str, reference: str) -> float:
        """Compute BLEU score"""
        try:
            bleu = corpus_bleu([prediction], [[reference]])
            return bleu.score / 100.0
        except Exception as e:
            logger.error(f"BLEU error: {e}")
            return 0.0
    
    def evaluate_qa_pair(self, question: str, ground_truth: str, 
                         use_multi_subgraph: bool = True) -> Dict:
        """Evaluate single QA pair"""
        logger.info("\n" + "="*80)
        logger.info("Evaluation Pipeline (Improved)")
        logger.info("="*80)
        
        # Get answer
        answer, gid = self.get_answer_improved(question, use_multi_subgraph)
        
        if not answer:
            logger.error("Failed to get answer")
            return {
                'question': question,
                'answer': '',
                'error': 'Failed to generate answer',
                'metrics': None
            }
        
        logger.info(f"\nGenerated Answer:\n{answer[:500]}...")
        logger.info(f"\nGround Truth:\n{ground_truth[:500]}...")
        
        # Compute metrics
        logger.info("\n[1/4] Answer Similarity...")
        answer_sim = self.compute_similarity(answer, ground_truth)
        logger.info(f"  Similarity: {answer_sim:.3f}")
        
        logger.info("\n[2/4] Question-Answer Relevance...")
        qa_relevance = self.compute_similarity(question, answer)
        logger.info(f"  Relevance: {qa_relevance:.3f}")
        
        logger.info("\n[3/4] ROUGE Scores...")
        rouge_scores = self.compute_rouge(answer, ground_truth)
        logger.info(f"  ROUGE-1: {rouge_scores['rouge_1']:.3f}")
        logger.info(f"  ROUGE-2: {rouge_scores['rouge_2']:.3f}")
        logger.info(f"  ROUGE-L: {rouge_scores['rouge_l']:.3f}")
        
        logger.info("\n[4/4] BLEU Score...")
        bleu_score = self.compute_bleu(answer, ground_truth)
        logger.info(f"  BLEU: {bleu_score:.3f}")
        
        metrics = SemanticMetrics(
            answer_similarity=answer_sim,
            question_answer_relevance=qa_relevance,
            rouge_1=rouge_scores['rouge_1'],
            rouge_2=rouge_scores['rouge_2'],
            rouge_l=rouge_scores['rouge_l'],
            bleu=bleu_score
        )
        
        logger.info(f"\nOverall Score: {metrics.overall_score():.3f}")
        
        return {
            'question': question,
            'answer': answer,
            'ground_truth': ground_truth,
            'gid': gid,
            'metrics': metrics.to_dict()
        }
    
    def evaluate_batch(self, qa_pairs: List[Tuple[str, str]], 
                       output_file: str,
                       start_index: int = 0,
                       use_multi_subgraph: bool = True,
                       save_interval: int = 10) -> Dict:
        """
        Evaluate batch of QA pairs
        
        Args:
            qa_pairs: List of (question, ground_truth) tuples
            output_file: JSON file to save results
            start_index: Resume from this index
            use_multi_subgraph: Whether to use multi-subgraph aggregation
            save_interval: Save checkpoint every N questions
        
        Returns:
            Aggregate metrics
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Batch Evaluation: {len(qa_pairs)} questions")
        logger.info(f"Starting from index: {start_index}")
        logger.info(f"Output: {output_file}")
        logger.info(f"{'='*80}")
        
        # Load existing results
        results = []
        if os.path.exists(output_file) and start_index > 0:
            try:
                with open(output_file, 'r') as f:
                    data = json.load(f)
                results = data.get('results', [])[:start_index]
                logger.info(f"Loaded {len(results)} existing results")
            except:
                pass
        
        # Process questions
        for i, (question, ground_truth) in enumerate(qa_pairs):
            if i < start_index:
                continue
            
            logger.info(f"\n{'#'*80}")
            logger.info(f"Question {i+1}/{len(qa_pairs)}")
            logger.info(f"{'#'*80}")
            
            try:
                result = self.evaluate_qa_pair(question, ground_truth, use_multi_subgraph)
                results.append(result)
            except Exception as e:
                logger.error(f"Error on question {i+1}: {e}")
                results.append({
                    'question': question,
                    'answer': '',
                    'ground_truth': ground_truth,
                    'error': str(e),
                    'metrics': None
                })
            
            # Save checkpoint
            if (i + 1) % save_interval == 0 or i == len(qa_pairs) - 1:
                self._save_results(output_file, results)
                logger.info(f"[Checkpoint] Saved {len(results)} results")
        
        # Calculate aggregate
        aggregate = self._calculate_aggregate(results)
        
        # Final save
        final_data = {
            'config': {
                'use_multi_subgraph': use_multi_subgraph,
                'timestamp': datetime.now().isoformat(),
                'total_questions': len(qa_pairs),
                'method': 'Improved U-Retrieval (Hybrid + Multi-subgraph)'
            },
            'results': results,
            'aggregate': aggregate
        }
        
        with open(output_file, 'w') as f:
            json.dump(final_data, f, indent=2)
        
        logger.info(f"\n{'='*80}")
        logger.info("EVALUATION COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Total: {len(results)} questions")
        
        if aggregate and 'avg_overall_score' in aggregate:
            logger.info(f"Overall Score: {aggregate['avg_overall_score']:.3f} ± {aggregate['std_overall_score']:.3f}")
        else:
            logger.warning("No valid results to calculate aggregate metrics")
        
        return aggregate
    
    def _save_results(self, output_file: str, results: List[Dict]):
        """Save intermediate results"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'results': results
        }
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _calculate_aggregate(self, results: List[Dict]) -> Dict:
        """Calculate aggregate metrics"""
        metrics_list = [r['metrics'] for r in results if r.get('metrics')]
        
        if not metrics_list:
            return {}
        
        agg = {}
        for key in ['answer_similarity', 'question_answer_relevance', 
                    'rouge_1', 'rouge_2', 'rouge_l', 'bleu', 'overall_score']:
            values = [m[key] for m in metrics_list if key in m]
            if values:
                agg[f'avg_{key}'] = float(np.mean(values))
                agg[f'std_{key}'] = float(np.std(values))
                agg[f'min_{key}'] = float(np.min(values))
                agg[f'max_{key}'] = float(np.max(values))
        
        agg['total_questions'] = len(results)
        agg['valid_questions'] = len(metrics_list)
        agg['failed_questions'] = len(results) - len(metrics_list)
        agg['success_rate'] = len(metrics_list) / len(results) if results else 0.0
        
        return agg


def main():
    parser = argparse.ArgumentParser(description='Improved Semantic Evaluator')
    parser.add_argument('--questions', type=str, required=True, help='Questions file')
    parser.add_argument('--answers', type=str, required=True, help='Ground truth answers file')
    parser.add_argument('--output', type=str, default='improved_eval_results.json', help='Output file')
    parser.add_argument('--start', type=int, default=0, help='Start index (for resume)')
    parser.add_argument('--limit', type=int, default=None, help='Max questions to evaluate')
    parser.add_argument('--single-subgraph', action='store_true', help='Use single subgraph only')
    
    args = parser.parse_args()
    
    # Load dataset
    from dataset_loader import QnADataset
    dataset = QnADataset(args.questions, args.answers)
    qa_pairs = list(dataset)
    
    if args.limit:
        qa_pairs = qa_pairs[:args.limit]
    
    logger.info(f"Loaded {len(qa_pairs)} QA pairs")
    
    # Initialize evaluator
    evaluator = ImprovedSemanticEvaluator(
        neo4j_url=os.getenv("NEO4J_URL"),
        neo4j_username=os.getenv("NEO4J_USERNAME"),
        neo4j_password=os.getenv("NEO4J_PASSWORD")
    )
    
    # Run evaluation
    aggregate = evaluator.evaluate_batch(
        qa_pairs,
        output_file=args.output,
        start_index=args.start,
        use_multi_subgraph=not args.single_subgraph
    )
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY (Improved U-Retrieval)")
    print("="*60)
    print(f"Total Questions: {aggregate.get('total_questions', 0)}")
    print(f"Valid Questions: {aggregate.get('valid_questions', 0)}")
    print(f"Failed Questions: {aggregate.get('failed_questions', 0)}")
    print(f"Success Rate: {aggregate.get('success_rate', 0)*100:.1f}%")
    
    if aggregate.get('avg_overall_score') is not None:
        print(f"Overall Score: {aggregate.get('avg_overall_score', 0):.3f} ± {aggregate.get('std_overall_score', 0):.3f}")
        print(f"Answer Similarity: {aggregate.get('avg_answer_similarity', 0):.3f}")
        print(f"Q-A Relevance: {aggregate.get('avg_question_answer_relevance', 0):.3f}")
        print(f"ROUGE-L: {aggregate.get('avg_rouge_l', 0):.3f}")
        print(f"BLEU: {aggregate.get('avg_bleu', 0):.3f}")
    else:
        print("No valid results to display metrics")


if __name__ == "__main__":
    main()
