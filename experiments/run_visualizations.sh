#!/bin/bash
# Quick start script for AStock Arena visualizations

echo "🎯 AStock Arena - Visualization Quick Start"
echo "==========================================="
echo ""

# Check which model version to use
if [ "$MODEL_VERSION" = "pro" ]; then
    echo "📌 Using: PRO models (Claude Opus, DeepSeek Reasoner, GPT-5.2, Qwen3-Max, Gemini 3 Pro)"
else
    echo "📌 Using: LITE models (Claude Haiku, DeepSeek Chat, GPT-5.1, Qwen3-235b, Gemini 2.5 Flash)"
fi

echo ""
echo "📊 Generating visualizations..."
echo ""

# Run visualization
python3 experiments/visualize.py

# Show generated files
echo ""
echo "📁 Generated files in experiments/visualizations/:"
ls -lh experiments/visualizations/*.png | awk '{print "   " $9 " (" $5 ")"}'

echo ""
echo "✅ Done! Open the images to view the charts."
echo ""
echo "💡 Tip: To switch versions, run:"
echo "   MODEL_VERSION=pro ./run_visualizations.sh"
echo "   MODEL_VERSION=lite ./run_visualizations.sh"
