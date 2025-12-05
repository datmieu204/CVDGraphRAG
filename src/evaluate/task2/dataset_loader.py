"""
Dataset loader for Task 2 QnA evaluation
Loads questions and ground truth answers from text files
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import List, Tuple, Dict
import os
from dataloader import load_high


class QnADataset:
    """Dataset loader for question-answer pairs"""
    
    def __init__(self, questions_file: str, answers_file: str = None):
        """
        Initialize dataset
        
        Args:
            questions_file: Path to questions file (one per line)
            answers_file: Optional path to ground truth answers (one per line)
        """
        self.questions_file = questions_file
        self.answers_file = answers_file
        
        # Load using dataloader.load_high
        questions_text = load_high(questions_file)
        self.questions = [q.strip() for q in questions_text.strip().split('\n') if q.strip()]
        
        if answers_file:
            answers_text = load_high(answers_file)
            self.answers = [a.strip() for a in answers_text.strip().split('\n') if a.strip()]
        else:
            self.answers = None
        
        # Validate matching counts
        if self.answers and len(self.questions) != len(self.answers):
            raise ValueError(
                f"Mismatch: {len(self.questions)} questions but {len(self.answers)} answers"
            )
    
    def __len__(self) -> int:
        """Return number of questions"""
        return len(self.questions)
    
    def __getitem__(self, idx: int) -> Tuple[str, str]:
        """
        Get question-answer pair by index
        
        Returns:
            (question, ground_truth_answer or None)
        """
        question = self.questions[idx]
        answer = self.answers[idx] if self.answers else None
        return question, answer
    
    def __iter__(self):
        """Iterate over question-answer pairs"""
        for i in range(len(self)):
            yield self[i]
    
    def get_questions(self) -> List[str]:
        """Get all questions"""
        return self.questions
    
    def get_answers(self) -> List[str]:
        """Get all ground truth answers (if available)"""
        return self.answers if self.answers else []
    
    def get_pairs(self) -> List[Dict[str, str]]:
        """Get all question-answer pairs as dictionaries"""
        pairs = []
        for q, a in self:
            pairs.append({
                'question': q,
                'ground_truth': a
            })
        return pairs
    
    def sample(self, n: int = 5, seed: int = None) -> List[Tuple[str, str]]:
        """
        Sample random question-answer pairs
        
        Args:
            n: Number of samples
            seed: Random seed for reproducibility
            
        Returns:
            List of (question, answer) tuples
        """
        import random
        if seed:
            random.seed(seed)
        
        indices = random.sample(range(len(self)), min(n, len(self)))
        return [self[i] for i in indices]


def load_medgraph_qna_dataset() -> QnADataset:
    """
    Load default MedGraph QnA dataset
    
    Returns:
        QnADataset with questions and ground truth answers
    """
    # Default paths relative to project root
    questions_file = "/home/medgraph/qna/questions_en.txt"
    answers_file = "/home/medgraph/qna/answers_en.txt"
    
    return QnADataset(questions_file, answers_file)


if __name__ == '__main__':
    # Test dataset loading
    print("Loading MedGraph QnA dataset...")
    dataset = load_medgraph_qna_dataset()
    
    print(f"✅ Loaded {len(dataset)} question-answer pairs")
    print(f"\nSample (first 3):")
    for i in range(min(3, len(dataset))):
        q, a = dataset[i]
        print(f"\n[Q{i+1}] {q}")
        print(f"[A{i+1}] {a[:150]}..." if len(a) > 150 else f"[A{i+1}] {a}")
