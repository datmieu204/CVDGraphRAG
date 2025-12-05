"""
Task 2: Medical QnA Evaluation with Knowledge Graph + LLM
Comprehensive metrics for medical question answering quality assessment
"""

import os
import json
import argparse
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from dotenv import load_dotenv

from camel.storages import Neo4jGraph
from dataloader import load_high
from summerize import process_chunks
from retrieve import seq_ret
from utils import get_response, ret_context, link_context
from dedicated_key_manager import create_dedicated_client
from logger_ import get_logger

load_dotenv()
logger = get_logger("task2_qna", log_file="logs/evaluate/task2_qna.log")


@dataclass
class MedicalQAMetrics:
    """Medical QA evaluation metrics"""
    pertinence: float  # 0-1: relevance between question and answer
    correctness: float  # 0-1: medical accuracy
    citation_precision: float  # 0-1: accuracy of cited sources
    citation_recall: float  # 0-1: completeness of citations
    understandability: float  # 0-1: clarity for non-experts
    answer_consistency: float  # 0-1: logical consistency
    faithfulness: float  # 0-1: grounding in KG/documents (anti-hallucination)
    
    def overall_score(self) -> float:
        """Weighted average of all metrics"""
        weights = {
            'pertinence': 0.15,
            'correctness': 0.25,  # Most important for medical
            'citation_precision': 0.10,
            'citation_recall': 0.10,
            'understandability': 0.15,
            'answer_consistency': 0.10,
            'faithfulness': 0.15
        }
        return (
            self.pertinence * weights['pertinence'] +
            self.correctness * weights['correctness'] +
            self.citation_precision * weights['citation_precision'] +
            self.citation_recall * weights['citation_recall'] +
            self.understandability * weights['understandability'] +
            self.answer_consistency * weights['answer_consistency'] +
            self.faithfulness * weights['faithfulness']
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary with overall score"""
        result = asdict(self)
        result['overall_score'] = self.overall_score()
        return result


class MedicalQAEvaluator:
    """Evaluator for medical question answering systems"""
    
    def __init__(self, neo4j_url: str, neo4j_username: str, neo4j_password: str):
        """Initialize evaluator with Neo4j connection"""
        logger.info("\n" + "="*80)
        logger.info("Medical QA Evaluation System")
        logger.info("="*80)
        
        # Connect to Neo4j
        self.n4j = Neo4jGraph(
            url=neo4j_url,
            username=neo4j_username,
            password=neo4j_password
        )
        logger.info("✅ Connected to Neo4j")
        
        # Create dedicated client for evaluation
        self.client = create_dedicated_client(task_id="qna_evaluator")
        logger.info("✅ Dedicated client initialized for QnA evaluation")
    
    def get_kg_context(self, gid: str) -> Dict:
        """Retrieve knowledge graph context for a given GID"""
        # Get entities and relationships from KG
        # Note: Entities use 'id' property, not 'name'
        query = f"""
        MATCH (n {{gid: '{gid}'}})
        WHERE n.id IS NOT NULL
        WITH n
        OPTIONAL MATCH (n)-[r]-(m)
        WHERE m.gid = '{gid}' AND m.id IS NOT NULL
        RETURN n.id as entity, 
               labels(n)[0] as type, 
               n.description as desc,
               collect(DISTINCT {{
                   rel: type(r), 
                   target: m.id, 
                   target_type: labels(m)[0]
               }}) as relationships
        LIMIT 50
        """
        
        try:
            results = self.n4j.query(query)
            
            # Debug log
            if not results:
                logger.warning(f"  ⚠️ Query returned 0 entities for GID: {gid}")
                # Try alternative query to check if nodes exist
                check_query = f"MATCH (n {{gid: '{gid}'}}) RETURN count(n) as count"
                count_result = self.n4j.query(check_query)
                if count_result:
                    logger.info(f"  📊 Total nodes with GID: {count_result[0].get('count', 0)}")
            
            return {
                'entities': results,
                'entity_count': len(results),
                'gid': gid
            }
        except Exception as e:
            logger.error(f"  ❌ Error querying KG: {e}")
            return {
                'entities': [],
                'entity_count': 0,
                'gid': gid
            }
    
    def get_answer_from_system(self, question: str) -> Tuple[str, Optional[str], Dict]:
        """
        Get answer from the medical QA system using standard flow
        Flow: question -> summarize -> retrieve GID -> get_response
        Returns: (answer, gid, kg_context)
        """
        logger.info(f"\n📝 Question: {question}")
        
        # Step 1: Summarize question
        logger.info("  [1/3] Summarizing question...")
        summaries = process_chunks(question, client=self.client)
        summary = " ".join(summaries) if summaries else question
        
        # Step 2: Retrieve relevant subgraph GID
        logger.info("  [2/3] Retrieving relevant knowledge graph...")
        gid = seq_ret(self.n4j, summary)
        
        if gid is None:
            logger.error("  ❌ No relevant knowledge graph found")
            logger.info("💡 Database might be empty. Build graph first with:")
            logger.info("   python three_layer_import.py --middle ../data/layer2_pmc")
            return "", None, {'entities': [], 'entity_count': 0, 'context_text': ''}
        
        logger.info(f"  ✅ Found GID: {gid}")
        
        # Step 3: Get KG context (entities + relations)
        logger.info("  [3/4] Retrieving KG context...")
        kg_context = self.get_kg_context_for_evaluation(gid)
        logger.info(f"  ✅ Retrieved {kg_context['entity_count']} entities")
        
        # Step 4: Generate answer using standard get_response
        logger.info("  [4/4] Generating answer with get_response...")
        answer = get_response(self.n4j, gid, question)
        
        return answer, gid, kg_context
    
    def get_kg_context_for_evaluation(self, gid: str) -> Dict:
        """
        Get KG context for evaluation metrics
        Returns entities, relationships, and text context
        LIMIT: 50 entities maximum
        """
        # Get self context (entities and relations within the subgraph)
        # Format: list of strings like "EntityA RELATION EntityB"
        self_context_result = ret_context(self.n4j, gid)
        
        # Get link context (cross-layer relationships)
        # Format: list of strings like "Reference X: EntityA has reference that EntityB REL EntityC"
        link_context_result = link_context(self.n4j, gid)
        
        # Parse entities from context strings
        entities = []
        entity_names = set()
        max_entities = 50  # LIMIT: Maximum 50 entities
        
        # Extract from self_context (format: "Entity1 RELATION Entity2")
        if self_context_result and len(entities) < max_entities:
            for item in self_context_result:
                if len(entities) >= max_entities:
                    break
                    
                if isinstance(item, str):
                    # Split by spaces to get entities
                    parts = item.split()
                    if len(parts) >= 3:
                        # First entity
                        entity1 = parts[0]
                        if entity1 and entity1 not in entity_names and len(entities) < max_entities:
                            entities.append({
                                'entity': entity1,
                                'type': 'UNKNOWN',
                                'desc': ''
                            })
                            entity_names.add(entity1)
                        
                        # Last entity
                        entity2 = parts[-1]
                        if entity2 and entity2 not in entity_names and len(entities) < max_entities:
                            entities.append({
                                'entity': entity2,
                                'type': 'UNKNOWN',
                                'desc': ''
                            })
                            entity_names.add(entity2)
        
        # Extract from link_context (only if we still have room)
        if link_context_result and len(entities) < max_entities:
            for item in link_context_result:
                if len(entities) >= max_entities:
                    break
                    
                if isinstance(item, str):
                    # Remove common words and split
                    cleaned = item.replace('Reference', '').replace('has the reference that', ' ')
                    words = cleaned.split()
                    
                    for word in words:
                        if len(entities) >= max_entities:
                            break
                            
                        # Keep words that look like entities (capitalized or with underscores)
                        if word and len(word) > 2 and (word[0].isupper() or '_' in word):
                            if word not in entity_names and word not in ['', 'X', 'A', 'B', 'C']:
                                entities.append({
                                    'entity': word,
                                    'type': 'REFERENCE',
                                    'desc': ''
                                })
                                entity_names.add(word)
        
        # Build context text for evaluation
        context_text = f"Self context: {len(self_context_result) if self_context_result else 0} triples\n"
        context_text += f"Link context: {len(link_context_result) if link_context_result else 0} references\n"
        context_text += f"Entities: {len(entities)}\n"
        
        return {
            'entities': entities,
            'entity_count': len(entities),
            'gid': gid,
            'self_context': self_context_result,
            'link_context': link_context_result,
            'context_text': context_text
        }
    
    def evaluate_pertinence(self, question: str, answer: str) -> float:
        """Evaluate relevance between question and answer (0-1)"""
        prompt = f"""Evaluate how relevant the answer is to the question on a scale of 0-1.

Question: {question}

Answer: {answer}

Scoring guide:
- 1.0: Answer directly addresses all aspects of the question
- 0.7-0.9: Answer addresses most aspects but may miss some details
- 0.4-0.6: Answer is partially relevant but misses key points
- 0.1-0.3: Answer is tangentially related
- 0.0: Answer is completely irrelevant

Provide ONLY a number between 0 and 1."""
        
        try:
            response = self.client.call_with_retry(prompt, model="models/gemini-2.5-flash-lite")
            return float(response.strip())
        except:
            return 0.5
    
    def evaluate_correctness(self, question: str, answer: str, kg_context: Dict) -> float:
        """Evaluate medical accuracy (0-1) based on KG grounding"""
        # Format entities for evaluation
        entities_sample = kg_context.get('entities', [])[:15]
        context_summary = kg_context.get('context_text', 'Context not available')
        
        entities_text = "\n".join([
            f"- {e['entity']} ({e['type']}): {e.get('desc', 'N/A')}"
            for e in entities_sample
        ])
        
        prompt = f"""As a medical expert, evaluate the medical accuracy of this answer on a scale of 0-1.

Question: {question}

Answer: {answer}

Knowledge Graph Context (ground truth):
{entities_text}

Context Summary:
{context_summary}

Scoring guide:
- 1.0: Medically accurate, supported by KG evidence
- 0.7-0.9: Mostly accurate with minor issues
- 0.4-0.6: Partially accurate but has significant errors
- 0.1-0.3: Major medical inaccuracies
- 0.0: Completely incorrect or dangerous information

Provide ONLY a number between 0 and 1."""
        
        try:
            response = self.client.call_with_retry(prompt, model="models/gemini-2.5-flash-lite")
            score = float(response.strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.5
    
    def evaluate_citations(self, answer: str, kg_context: Dict) -> Tuple[float, float]:
        """
        Evaluate citation quality based on entity mentions
        Returns: (precision, recall)
        """
        entities = kg_context.get('entities', [])
        
        if not entities:
            return 0.0, 0.0
        
        # Extract entity names from KG (normalize to lowercase)
        kg_entity_names = set()
        for e in entities:
            entity_name = e.get('entity', '').lower().strip()
            if entity_name:
                kg_entity_names.add(entity_name)
                # Also add individual words for partial matching
                words = entity_name.split()
                if len(words) > 1:
                    for word in words:
                        if len(word) > 3:  # Skip short words
                            kg_entity_names.add(word)
        
        if not kg_entity_names:
            return 0.0, 0.0
        
        # Check which KG entities are mentioned in answer
        answer_lower = answer.lower()
        mentioned_entities = set()
        
        for entity in kg_entity_names:
            if entity in answer_lower:
                mentioned_entities.add(entity)
        
        # Calculate metrics
        # Precision: of entities we could cite, how many did we mention?
        precision = len(mentioned_entities) / len(kg_entity_names) if kg_entity_names else 0.0
        
        # Recall: same as precision in this case (no external citations)
        recall = precision
        
        return precision, recall
    
    def evaluate_understandability(self, answer: str) -> float:
        """Evaluate clarity for non-experts (0-1)"""
        prompt = f"""Evaluate how understandable this medical answer is for a non-expert on a scale of 0-1.

Answer: {answer}

Scoring guide:
- 1.0: Clear, well-explained with simple language and examples
- 0.7-0.9: Mostly clear but uses some unexplained medical terms
- 0.4-0.6: Moderately understandable but quite technical
- 0.1-0.3: Very technical, hard for non-experts
- 0.0: Incomprehensible jargon

Provide ONLY a number between 0 and 1."""
        
        try:
            response = self.client.call_with_retry(prompt, model="models/gemini-2.5-flash-lite")
            score = float(response.strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.5
    
    def evaluate_consistency(self, question: str, answer: str) -> float:
        """Evaluate logical consistency within the answer (0-1)"""
        prompt = f"""Evaluate the internal logical consistency of this answer on a scale of 0-1.

Question: {question}

Answer: {answer}

Scoring guide:
- 1.0: Perfectly consistent, no contradictions
- 0.7-0.9: Mostly consistent with minor inconsistencies
- 0.4-0.6: Some logical inconsistencies or contradictions
- 0.1-0.3: Major contradictions
- 0.0: Completely contradictory

Provide ONLY a number between 0 and 1."""
        
        try:
            response = self.client.call_with_retry(prompt, model="models/gemini-2.5-flash-lite")
            score = float(response.strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.5
    
    def evaluate_faithfulness(self, answer: str, kg_context: Dict) -> float:
        """Evaluate grounding in KG (anti-hallucination) (0-1)"""
        entities_sample = kg_context.get('entities', [])[:15]
        context_summary = kg_context.get('context_text', 'Context not available')
        
        entities_text = "\n".join([
            f"- {e['entity']} ({e['type']}): {e.get('desc', 'N/A')}"
            for e in entities_sample
        ])
        
        prompt = f"""Evaluate how well this answer is grounded in the provided knowledge graph context (0-1).
Check for hallucinations or unsupported claims.

Answer: {answer}

Knowledge Graph Context:
{entities_text}

Context Summary:
{context_summary}

Scoring guide:
- 1.0: All claims supported by KG, no hallucinations
- 0.7-0.9: Mostly grounded with minimal unsupported claims
- 0.4-0.6: Some claims not in KG but reasonable
- 0.1-0.3: Many unsupported or contradictory claims
- 0.0: Complete hallucination, not based on KG

Provide ONLY a number between 0 and 1."""
        
        try:
            response = self.client.call_with_retry(prompt, model="models/gemini-2.5-flash-lite")
            score = float(response.strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.5
    
    def evaluate_answer(self, question: str, answer: str, gid: str, kg_context: Dict) -> MedicalQAMetrics:
        """
        Comprehensive evaluation of a medical QA answer
        """
        logger.info("\n" + "="*80)
        logger.info("Evaluating Answer Quality")
        logger.info("="*80)
        
        # Evaluate each metric
        logger.info("\n[1/7] Evaluating Pertinence...")
        pertinence = self.evaluate_pertinence(question, answer)
        logger.info(f"  ✅ Pertinence: {pertinence:.3f}")
        
        logger.info("\n[2/7] Evaluating Correctness...")
        correctness = self.evaluate_correctness(question, answer, kg_context)
        logger.info(f"  ✅ Correctness: {correctness:.3f}")
        
        logger.info("\n[3/7] Evaluating Citations...")
        citation_precision, citation_recall = self.evaluate_citations(answer, kg_context)
        logger.info(f"  ✅ Citation Precision: {citation_precision:.3f}")
        logger.info(f"  ✅ Citation Recall: {citation_recall:.3f}")
        
        logger.info("\n[4/7] Evaluating Understandability...")
        understandability = self.evaluate_understandability(answer)
        logger.info(f"  ✅ Understandability: {understandability:.3f}")
        
        logger.info("\n[5/7] Evaluating Consistency...")
        consistency = self.evaluate_consistency(question, answer)
        logger.info(f"  ✅ Answer Consistency: {consistency:.3f}")
        
        logger.info("\n[6/7] Evaluating Faithfulness...")
        faithfulness = self.evaluate_faithfulness(answer, kg_context)
        logger.info(f"  ✅ Faithfulness: {faithfulness:.3f}")
        
        # Create metrics object
        metrics = MedicalQAMetrics(
            pertinence=pertinence,
            correctness=correctness,
            citation_precision=citation_precision,
            citation_recall=citation_recall,
            understandability=understandability,
            answer_consistency=consistency,
            faithfulness=faithfulness
        )
        
        logger.info("\n[7/7] Computing Overall Score...")
        logger.info(f"  ✅ Overall Score: {metrics.overall_score():.3f}")
        
        return metrics
    
    def evaluate_qa_pair(self, question: str, ground_truth: Optional[str] = None) -> Dict:
        """
        Full pipeline: get answer and evaluate
        """
        logger.info("\n" + "="*80)
        logger.info("Medical QA Evaluation Pipeline")
        logger.info("="*80)
        
        # Get answer from system
        answer, gid, kg_context = self.get_answer_from_system(question)
        
        if not answer or gid is None:
            logger.error("❌ Failed to get answer from system")
            return {
                'question': question,
                'answer': '',
                'error': 'Failed to retrieve answer',
                'metrics': None
            }
        
        logger.info(f"\n💬 Generated Answer:\n{answer}\n")
        
        # Evaluate answer
        metrics = self.evaluate_answer(question, answer, gid, kg_context)
        
        # Prepare result
        result = {
            'question': question,
            'answer': answer,
            'gid': gid,
            'ground_truth': ground_truth,
            'metrics': metrics.to_dict(),
            'kg_context_summary': {
                'entity_count': kg_context.get('entity_count', 0),
                'sample_entities': [e.get('entity', 'N/A') for e in kg_context.get('entities', [])[:5]],
                'context_summary': kg_context.get('context_text', 'N/A')
            }
        }
        
        return result
    
    def evaluate_dataset(self, questions: List[str], output_file: str = None) -> List[Dict]:
        """Evaluate multiple questions and save results"""
        logger.info(f"\n📊 Evaluating {len(questions)} questions...")
        
        results = []
        for i, question in enumerate(questions, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"Question {i}/{len(questions)}")
            logger.info(f"{'='*80}")
            
            result = self.evaluate_qa_pair(question)
            results.append(result)
        
        # Compute aggregate statistics
        valid_results = [r for r in results if r.get('metrics')]
        if valid_results:
            avg_metrics = {}
            for key in valid_results[0]['metrics'].keys():
                avg_metrics[f'avg_{key}'] = np.mean([r['metrics'][key] for r in valid_results])
            
            logger.info("\n" + "="*80)
            logger.info("Aggregate Results")
            logger.info("="*80)
            for key, value in avg_metrics.items():
                logger.info(f"  {key}: {value:.3f}")
        
        # Save results
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'results': results,
                    'aggregate': avg_metrics if valid_results else {}
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"\n💾 Results saved to: {output_file}")
        
        return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Medical QA Evaluation')
    parser.add_argument('--question', type=str, help='Single question to evaluate')
    parser.add_argument('--questions-file', type=str, help='File with multiple questions (one per line)')
    parser.add_argument('--output', type=str, default='results/task2_evaluation.json',
                       help='Output file for results')
    
    # Neo4j config
    parser.add_argument('--neo4j-url', type=str, 
                       default=os.getenv('NEO4J_URI', 'bolt://localhost:7687'))
    parser.add_argument('--neo4j-username', type=str, 
                       default=os.getenv('NEO4J_USERNAME', 'neo4j'))
    parser.add_argument('--neo4j-password', type=str, 
                       default=os.getenv('NEO4J_PASSWORD'))
    
    args = parser.parse_args()
    
    if not args.neo4j_password:
        logger.error("❌ Neo4j password required")
        return
    
    # Initialize evaluator
    evaluator = MedicalQAEvaluator(
        args.neo4j_url,
        args.neo4j_username,
        args.neo4j_password
    )
    
    # Prepare questions
    questions = []
    if args.question:
        questions = [args.question]
    elif args.questions_file:
        with open(args.questions_file, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
    else:
        # Default test questions
        questions = [
            "What is heart failure and what are its main symptoms?",
            "What medications are commonly used to treat heart failure?",
            "What are the risk factors for cardiovascular disease?"
        ]
    
    # Run evaluation
    evaluator.evaluate_dataset(questions, output_file=args.output)
    
    logger.info("\n✅ Evaluation completed!")


if __name__ == '__main__':
    main()

