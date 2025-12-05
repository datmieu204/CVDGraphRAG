#!/bin/bash

# Full evaluation script for paper results
# Runs all baselines on complete 200-question dataset

cd /home/medgraph/src/experiment

echo "======================================"
echo "Full Baseline Comparison (200 Q)"
echo "For Paper Results"
echo "======================================"
echo ""
echo "⚠️  This will take approximately 2-3 hours"
echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
sleep 5

echo ""
echo "Starting evaluation..."
echo ""

python compare_baselines.py \
  --questions /home/medgraph/qna/quang/questions.txt \
  --answers /home/medgraph/qna/quang/answers_with_explain.txt \
  --output-dir results/paper_final \
  --model models/gemini-2.5-flash-lite \
  --limit 200

echo ""
echo "✅ Full evaluation completed!"
echo "Results saved to: results/paper_final/"
echo ""
echo "Summary:"
cat results/paper_final/comparison_summary.json | jq '.table'
