#!/usr/bin/env python3

import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import sys

def load_results(json_path: str) -> List[Dict]:
    """Load results from JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('results', [])

def compute_aggregates(results: List[Dict]) -> Dict:
    """Compute aggregate statistics across all results"""
    
    # Filter valid results (those with metrics)
    valid_results = [r for r in results if r.get('metrics') is not None]
    
    if not valid_results:
        print("No valid results found!")
        return {}
    
    print(f"Found {len(valid_results)} valid results out of {len(results)} total")
    
    # Collect all metric values
    metric_names = list(valid_results[0]['metrics'].keys())
    aggregates = {}
    
    for metric_name in metric_names:
        values = [r['metrics'][metric_name] for r in valid_results if metric_name in r['metrics']]
        
        if values:
            aggregates[f'avg_{metric_name}'] = float(np.mean(values))
            aggregates[f'std_{metric_name}'] = float(np.std(values))
            aggregates[f'min_{metric_name}'] = float(np.min(values))
            aggregates[f'max_{metric_name}'] = float(np.max(values))
            aggregates[f'median_{metric_name}'] = float(np.median(values))
    
    # Add summary statistics
    aggregates['total_questions'] = len(results)
    aggregates['valid_questions'] = len(valid_results)
    aggregates['failed_questions'] = len(results) - len(valid_results)
    aggregates['success_rate'] = len(valid_results) / len(results) if results else 0.0
    
    return aggregates

def print_summary_table(aggregates: Dict):
    """Print formatted summary table"""
    
    print("\n" + "="*80)
    print("AGGREGATE EVALUATION RESULTS (200 Questions)")
    print("="*80)
    
    print(f"\nDataset Statistics:")
    print(f"  Total Questions:    {aggregates['total_questions']}")
    print(f"  Valid Results:      {aggregates['valid_questions']}")
    print(f"  Failed Results:     {aggregates['failed_questions']}")
    print(f"  Success Rate:       {aggregates['success_rate']*100:.2f}%")
    
    # Group metrics by category
    semantic_metrics = ['answer_similarity', 'question_answer_relevance']
    rouge_metrics = ['rouge_1', 'rouge_2', 'rouge_l']
    other_metrics = ['bleu', 'overall_score']
    
    print(f"\n📌 Semantic Similarity Metrics (BGE-based):")
    print(f"{'Metric':<30} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Median':<10}")
    print("-" * 80)
    
    for metric in semantic_metrics:
        if f'avg_{metric}' in aggregates:
            print(f"{metric:<30} "
                  f"{aggregates[f'avg_{metric}']:<10.4f} "
                  f"{aggregates[f'std_{metric}']:<10.4f} "
                  f"{aggregates[f'min_{metric}']:<10.4f} "
                  f"{aggregates[f'max_{metric}']:<10.4f} "
                  f"{aggregates[f'median_{metric}']:<10.4f}")
    
    print(f"\nROUGE Metrics:")
    print(f"{'Metric':<30} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Median':<10}")
    print("-" * 80)
    
    for metric in rouge_metrics:
        if f'avg_{metric}' in aggregates:
            print(f"{metric:<30} "
                  f"{aggregates[f'avg_{metric}']:<10.4f} "
                  f"{aggregates[f'std_{metric}']:<10.4f} "
                  f"{aggregates[f'min_{metric}']:<10.4f} "
                  f"{aggregates[f'max_{metric}']:<10.4f} "
                  f"{aggregates[f'median_{metric}']:<10.4f}")
    
    print(f"\nOverall Metrics:")
    print(f"{'Metric':<30} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Median':<10}")
    print("-" * 80)
    
    for metric in other_metrics:
        if f'avg_{metric}' in aggregates:
            print(f"{metric:<30} "
                  f"{aggregates[f'avg_{metric}']:<10.4f} "
                  f"{aggregates[f'std_{metric}']:<10.4f} "
                  f"{aggregates[f'min_{metric}']:<10.4f} "
                  f"{aggregates[f'max_{metric}']:<10.4f} "
                  f"{aggregates[f'median_{metric}']:<10.4f}")
    
    print("\n" + "="*80)
    
    # Weighted Overall Score breakdown
    if 'avg_overall_score' in aggregates:
        print(f"\nOverall Score Breakdown:")
        print(f"  Final Score: {aggregates['avg_overall_score']:.4f}")
        print(f"  Calculation: 0.4 × Answer_Sim + 0.3 × Avg_ROUGE + 0.2 × BLEU + 0.1 × QA_Rel")
        
        if all(f'avg_{m}' in aggregates for m in ['answer_similarity', 'rouge_1', 'rouge_2', 'rouge_l', 'bleu', 'question_answer_relevance']):
            avg_rouge = (aggregates['avg_rouge_1'] + aggregates['avg_rouge_2'] + aggregates['avg_rouge_l']) / 3.0
            calculated = (
                0.4 * aggregates['avg_answer_similarity'] +
                0.3 * avg_rouge +
                0.2 * aggregates['avg_bleu'] +
                0.1 * aggregates['avg_question_answer_relevance']
            )
            print(f"  Verified:    {calculated:.4f}")
            print(f"    - Answer Similarity contribution:  {0.4 * aggregates['avg_answer_similarity']:.4f}")
            print(f"    - Avg ROUGE contribution:          {0.3 * avg_rouge:.4f}")
            print(f"    - BLEU contribution:               {0.2 * aggregates['avg_bleu']:.4f}")
            print(f"    - QA Relevance contribution:       {0.1 * aggregates['avg_question_answer_relevance']:.4f}")
    
    print("="*80)

def save_aggregates(aggregates: Dict, output_path: str):
    """Save aggregates to JSON file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(aggregates, f, indent=2, ensure_ascii=False)
    print(f"\nAggregates saved to: {output_path}")

def generate_latex_table(aggregates: Dict) -> str:
    """Generate LaTeX table for paper"""
    
    latex = []
    latex.append("% CVDGraphRAG Evaluation Results - 200 Medical Questions")
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{CVDGraphRAG Performance on Medical QA Dataset (200 questions)}")
    latex.append("\\label{tab:cvdgraphrag_results}")
    latex.append("\\begin{tabular}{lcccc}")
    latex.append("\\hline")
    latex.append("\\textbf{Metric} & \\textbf{Mean} & \\textbf{Std} & \\textbf{Min} & \\textbf{Max} \\\\")
    latex.append("\\hline")
    
    # Semantic metrics
    latex.append("\\multicolumn{5}{l}{\\textit{Semantic Similarity (BGE-based)}} \\\\")
    for metric in ['answer_similarity', 'question_answer_relevance']:
        if f'avg_{metric}' in aggregates:
            name = metric.replace('_', ' ').title()
            latex.append(f"{name} & "
                        f"{aggregates[f'avg_{metric}']:.3f} & "
                        f"{aggregates[f'std_{metric}']:.3f} & "
                        f"{aggregates[f'min_{metric}']:.3f} & "
                        f"{aggregates[f'max_{metric}']:.3f} \\\\")
    
    latex.append("\\hline")
    
    # ROUGE metrics
    latex.append("\\multicolumn{5}{l}{\\textit{ROUGE Scores}} \\\\")
    for metric in ['rouge_1', 'rouge_2', 'rouge_l']:
        if f'avg_{metric}' in aggregates:
            name = metric.upper().replace('_', '-')
            latex.append(f"{name} & "
                        f"{aggregates[f'avg_{metric}']:.3f} & "
                        f"{aggregates[f'std_{metric}']:.3f} & "
                        f"{aggregates[f'min_{metric}']:.3f} & "
                        f"{aggregates[f'max_{metric}']:.3f} \\\\")
    
    latex.append("\\hline")
    
    # Overall
    latex.append("\\multicolumn{5}{l}{\\textit{Overall Metrics}} \\\\")
    for metric in ['bleu', 'overall_score']:
        if f'avg_{metric}' in aggregates:
            name = 'BLEU' if metric == 'bleu' else 'Overall Score'
            latex.append(f"\\textbf{{{name}}} & "
                        f"\\textbf{{{aggregates[f'avg_{metric}']:.3f}}} & "
                        f"{aggregates[f'std_{metric}']:.3f} & "
                        f"{aggregates[f'min_{metric}']:.3f} & "
                        f"{aggregates[f'max_{metric}']:.3f} \\\\")
    
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)

def main():
    # Paths
    results_path = Path(__file__).parent.parent / "evaluate/task2/results/semantic_eval.json"
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    aggregates_path = output_dir / "aggregates.json"
    latex_path = output_dir / "results_table.tex"
    

    print(f"\nLoading results from: {results_path}")
    
    results = load_results(str(results_path))
    
    if not results:
        print("No results found in the JSON file!")
        return 1
    
    aggregates = compute_aggregates(results)
    
    if not aggregates:
        return 1
    
    print_summary_table(aggregates)
    
    save_aggregates(aggregates, str(aggregates_path))
    
    latex_table = generate_latex_table(aggregates)
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(latex_table)
    print(f"LaTeX table saved to: {latex_path}")
    
    print(f"\nAggregate computation completed successfully!")
    print(f"   - Aggregates JSON: {aggregates_path}")
    print(f"   - LaTeX table:     {latex_path}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
