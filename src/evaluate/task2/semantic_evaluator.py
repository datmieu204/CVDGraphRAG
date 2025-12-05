#!/usr/bin/env python3

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
import re

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
        logger.info("Connected to Neo4j")
        
        self.client = create_dedicated_client(task_id="semantic_evaluator")
        logger.info("Client initialized")
    
    def parse_log_checkpoint(self, log_file: str) -> Tuple[List[Dict], int]:
        """Parse results from log file"""
        if not os.path.exists(log_file):
            logger.info(f"No log file found: {log_file}")
            return [], 0
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                log_content = f.read()
            
            results = []
            
            question_pattern = r'Question: (.+?)\n'
            answer_pattern = r'Generated Answer:\n(.+?)(?=\n2025-|\nGround Truth:)'
            ground_truth_pattern = r'Ground Truth:\n(.+?)(?=\n2025-|\n=)'
            gid_pattern = r'GID: ([a-f0-9-]+)'
            
            # Metrics patterns
            similarity_pattern = r'Similarity: ([0-9.]+)'
            relevance_pattern = r'Relevance: ([0-9.]+)'
            rouge1_pattern = r'ROUGE-1: ([0-9.]+)'
            rouge2_pattern = r'ROUGE-2: ([0-9.]+)'
            rougel_pattern = r'ROUGE-L: ([0-9.]+)'
            bleu_pattern = r'BLEU: ([0-9.]+)'
            overall_pattern = r'Overall Score: ([0-9.]+)'
            
            question_blocks = re.split(r'Question \d+/200', log_content)[1:]  # Skip header
            
            for block in question_blocks:
                try:
                    # Extract data
                    question_match = re.search(question_pattern, block)
                    answer_match = re.search(answer_pattern, block, re.DOTALL)
                    gt_match = re.search(ground_truth_pattern, block, re.DOTALL)
                    gid_match = re.search(gid_pattern, block)
                    
                    if not question_match:
                        continue
                    
                    question = question_match.group(1).strip()
                    answer = answer_match.group(1).strip() if answer_match else ""
                    ground_truth = gt_match.group(1).strip() if gt_match else ""
                    gid = gid_match.group(1) if gid_match else None
                    
                    # Extract metrics
                    metrics = None
                    sim_match = re.search(similarity_pattern, block)
                    
                    if sim_match:  # Has metrics
                        rel_match = re.search(relevance_pattern, block)
                        r1_match = re.search(rouge1_pattern, block)
                        r2_match = re.search(rouge2_pattern, block)
                        rl_match = re.search(rougel_pattern, block)
                        bleu_match = re.search(bleu_pattern, block)
                        overall_match = re.search(overall_pattern, block)
                        
                        metrics = {
                            'answer_similarity': float(sim_match.group(1)) if sim_match else 0.0,
                            'question_answer_relevance': float(rel_match.group(1)) if rel_match else 0.0,
                            'rouge_1': float(r1_match.group(1)) if r1_match else 0.0,
                            'rouge_2': float(r2_match.group(1)) if r2_match else 0.0,
                            'rouge_l': float(rl_match.group(1)) if rl_match else 0.0,
                            'bleu': float(bleu_match.group(1)) if bleu_match else 0.0,
                            'overall_score': float(overall_match.group(1)) if overall_match else 0.0
                        }
                    
                    results.append({
                        'question': question,
                        'answer': answer,
                        'ground_truth': ground_truth,
                        'gid': gid,
                        'metrics': metrics
                    })
                
                except Exception as e:
                    logger.warning(f"Failed to parse block: {e}")
                    continue
            
            last_index = len(results)
            logger.info(f"Parsed log checkpoint: {last_index} questions completed")
            return results, last_index
        
        except Exception as e:
            logger.error(f"Failed to parse log: {e}")
            return [], 0
    
    def load_checkpoint(self, checkpoint_file: str) -> Tuple[List[Dict], int]:
        """Load previous results from JSON or parse from log"""
        # Try JSON first
        if checkpoint_file.endswith('.json') and os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                results = data.get('results', [])
                last_index = len(results)
                
                logger.info(f"Loaded JSON checkpoint: {last_index} questions completed")
                return results, last_index
            except Exception as e:
                logger.warning(f"Failed to load JSON: {e}")
        
        # Try log file
        if checkpoint_file.endswith('.log') or not checkpoint_file.endswith('.json'):
            log_file = checkpoint_file if checkpoint_file.endswith('.log') else checkpoint_file.replace('.json', '.log')
            return self.parse_log_checkpoint(log_file)
        
        logger.info(f"No checkpoint found: {checkpoint_file}")
        return [], 0
    
    def get_answer_from_system(self, question: str) -> Tuple[str, str]:
        """
        Get answer from QA system
        Returns: (answer, gid)
        """
        logger.info(f"\nQuestion: {question}")
        
        # Summarize
        logger.info("  [1/3] Summarizing...")
        summaries = process_chunks(question, client=self.client)
        summary = " ".join(summaries) if summaries else question
        
        # Retrieve GID
        logger.info("  [2/3] Retrieving GID...")
        gid = seq_ret(self.n4j, summary, client=self.client)
        
        if gid is None:
            logger.error("  No GID found")
            return "", ""
        
        logger.info(f"  GID: {gid}")
        
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
            logger.error(f"  Embedding error: {e}")
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
            logger.error(f"  ROUGE error: {e}")
            return {'rouge_1': 0.0, 'rouge_2': 0.0, 'rouge_l': 0.0}
    
    def compute_bleu_score(self, prediction: str, reference: str) -> float:
        """Compute BLEU score"""
        try:
            bleu = corpus_bleu([prediction], [[reference]])
            return bleu.score / 100.0  # Normalize to 0-1
        except Exception as e:
            logger.error(f"  BLEU error: {e}")
            return 0.0
    
    def evaluate_qa_pair(self, question: str, ground_truth: str) -> Dict:
        """Evaluate single QA pair"""
        logger.info("\n" + "="*80)
        logger.info("Evaluation Pipeline")
        logger.info("="*80)
        
        # Get answer
        answer, gid = self.get_answer_from_system(question)
        
        if not answer:
            logger.error("Failed to get answer")
            return {
                'question': question,
                'answer': '',
                'error': 'Failed to generate answer',
                'metrics': None
            }
        
        logger.info(f"\nGenerated Answer:\n{answer}\n")
        logger.info(f"Ground Truth:\n{ground_truth}\n")
        
        logger.info("\n[1/4] Answer Similarity (BGE: Predicted vs Ground Truth)...")
        answer_sim = self.compute_similarity(answer, ground_truth)
        logger.info(f"  Similarity: {answer_sim:.3f}")
        
        logger.info("\n[2/4] Question-Answer Relevance (BGE)...")
        qa_relevance = self.compute_similarity(question, answer)
        logger.info(f"  Relevance: {qa_relevance:.3f}")
        
        logger.info("\n[3/4] ROUGE Scores...")
        rouge_scores = self.compute_rouge_scores(answer, ground_truth)
        logger.info(f"  ROUGE-1: {rouge_scores['rouge_1']:.3f}")
        logger.info(f"  ROUGE-2: {rouge_scores['rouge_2']:.3f}")
        logger.info(f"  ROUGE-L: {rouge_scores['rouge_l']:.3f}")
        
        logger.info("\n[4/4] BLEU Score...")
        bleu_score = self.compute_bleu_score(answer, ground_truth)
        logger.info(f"  BLEU: {bleu_score:.3f}")
        
        # Create metrics
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
    
    def evaluate_dataset(self, pairs: List[Tuple[str, str]], output_file: str = None, 
                        start_from: int = 0, checkpoint_results: List[Dict] = None) -> List[Dict]:
        """Evaluate multiple QA pairs with checkpoint support"""
        total_questions = len(pairs)
        
        results = checkpoint_results if checkpoint_results is not None else []
        
        start_idx = max(start_from, len(results))
        
        if start_idx > 0:
            logger.info(f"\nResuming from question {start_idx + 1}/{total_questions}")
        else:
            logger.info(f"\nEvaluating {total_questions} questions...")
        
        for i in range(start_idx, total_questions):
            question, ground_truth = pairs[i]
            
            logger.info(f"\n{'='*80}")
            logger.info(f"Question {i + 1}/{total_questions}")
            logger.info(f"{'='*80}")
            
            try:
                result = self.evaluate_qa_pair(question, ground_truth)
                results.append(result)
                
                if output_file and (i + 1) % 5 == 0:  # Save every 5 questions
                    self._save_checkpoint(results, output_file)
                    logger.info(f"Checkpoint saved: {i + 1}/{total_questions}")
            
            except Exception as e:
                logger.error(f"Error on question {i + 1}: {e}")
                # Save partial result
                results.append({
                    'question': question,
                    'ground_truth': ground_truth,
                    'error': str(e),
                    'metrics': None
                })
                
                # Save checkpoint on error
                if output_file:
                    self._save_checkpoint(results, output_file)
                    logger.info(f"Emergency checkpoint saved at question {i + 1}")
                
                # Continue with next question
                continue
        
        # Aggregate
        valid_results = [r for r in results if r.get('metrics')]
        
        avg_metrics = {}
        if valid_results:
            for key in valid_results[0]['metrics'].keys():
                values = [r['metrics'][key] for r in valid_results]
                avg_metrics[f'avg_{key}'] = float(np.mean(values))
                avg_metrics[f'std_{key}'] = float(np.std(values))
                avg_metrics[f'min_{key}'] = float(np.min(values))
                avg_metrics[f'max_{key}'] = float(np.max(values))
            
            logger.info(f"AGGREGATE RESULTS ({len(valid_results)}/{len(results)} valid)")
            logger.info("="*80)
            for key in sorted([k for k in avg_metrics.keys() if k.startswith('avg_')]):
                metric_name = key[4:]
                logger.info(
                    f"  {metric_name:.<35} "
                    f"{avg_metrics[key]:.3f} ± {avg_metrics[f'std_{metric_name}']:.3f}"
                )
        
        if output_file:
            self._save_checkpoint(results, output_file, avg_metrics)
            logger.info(f"\nFinal results saved to: {output_file}")
        
        logger.info("\nEvaluation completed!")
        return results
    
    def _save_checkpoint(self, results: List[Dict], output_file: str, 
                        aggregate: Dict = None) -> None:
        """Save results to checkpoint file"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        data = {'results': results}
        if aggregate:
            data['aggregate'] = aggregate
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description='Semantic QA Evaluation')
    parser.add_argument('--questions', type=str, 
                       default='/home/medgraph/qna/quang/questions.txt')
    parser.add_argument('--answers', type=str,
                       default='/home/medgraph/qna/quang/answers_with_explain.txt')
    parser.add_argument('--limit', type=int, help='Limit number of questions')
    parser.add_argument('--output', type=str, 
                       default='results/semantic_eval.json')
    parser.add_argument('--checkpoint', type=str,
                       help='Resume from checkpoint file (JSON or log)')
    parser.add_argument('--log-file', type=str,
                       default='logs/evaluate/semantic.log',
                       help='Log file to parse for checkpoint recovery')
    parser.add_argument('--start-from', type=int, default=0,
                       help='Start from question index (0-based)')
    
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
    
    pairs = list(zip(questions, answers))
    logger.info(f"Loaded {len(pairs)} pairs")
    
    if args.limit:
        pairs = pairs[:args.limit]
        logger.info(f"Will evaluate {len(pairs)} questions")
    
    # Evaluate
    evaluator = SemanticEvaluator(
        args.neo4j_url,
        args.neo4j_username,
        args.neo4j_password
    )
    
    checkpoint_results = []
    start_from = args.start_from
    
    if args.checkpoint:
        checkpoint_results, last_index = evaluator.load_checkpoint(args.checkpoint)
        start_from = max(start_from, last_index)
        logger.info(f"Will resume from question {start_from + 1}")
    elif os.path.exists(args.log_file):
        # Auto-detect from log file
        logger.info(f"Auto-detecting checkpoint from: {args.log_file}")
        checkpoint_results, last_index = evaluator.parse_log_checkpoint(args.log_file)
        if last_index > 0:
            start_from = max(start_from, last_index)
            logger.info(f"Will resume from question {start_from + 1}")
    
    evaluator.evaluate_dataset(
        pairs, 
        output_file=args.output,
        start_from=start_from,
        checkpoint_results=checkpoint_results
    )
    
    return 0


if __name__ == '__main__':
    exit(main())