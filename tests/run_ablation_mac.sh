#!/bin/bash
echo "============================================"
echo "STEERING VS PROMPTING ABLATION STUDY"
echo "============================================"
echo ""

cd "$(dirname "$0")"

# Check if vectors exist
if [ ! -f "vectors/melatonin.pt" ]; then
    echo "ERROR: vectors/melatonin.pt not found"
    echo "Please ensure steering vectors are in the vectors/ directory"
    exit 1
fi

echo "Running ablation study..."
echo "Compound: MELATONIN"
echo "Tests: T5_introspection"
echo "Intensities: 5.0, 8.0, 12.0"
echo "Iterations: 20 per condition"
echo ""

python3 tests/run_ablation.py -c melatonin -t T5_introspection -n 20 --multi-intensity

echo ""
echo "============================================"
echo "COMPLETE - Results in ablation_results/"
echo "============================================"
