# YAML Configuration Files Quick Reference

This directory contains several YAML configuration files for different testing scenarios. Choose the right one based on your goal:

## Quick Selection Guide

- **New model, want to verify it works**: `quick_solvable.yaml`
- **Local model testing**: `local_friendly.yaml`
- **Compare reasoning skills**: `reasoning_test.yaml`
- **Find math skill ceiling**: `math_gradient.yaml`
- **Broad capability overview**: `capability_sampler.yaml`
- **Test math olympiad skills**: `aime24_only.yaml`
- **Complete evaluation**: `sample1.yaml`

## Quick Testing & Validation

**`quick_solvable.yaml`** (10 problems, ~2-5 minutes)
- **Purpose**: Verify benchmark system works with good models
- **Content**: 5 easy GSM8K + 5 ARC Challenge problems
- **Expected accuracy**: 80-90% with good cloud models
- **Best for**: System validation, quick sanity checks

**`tiny_test.yaml`** (2 problems, ~30 seconds)
- **Purpose**: Minimal test for debugging or quick checks
- **Content**: 1 GSM8K + 1 Puzzte problem
- **Best for**: Testing timeout settings, basic functionality

## Model Evaluation by Use Case

**`local_friendly.yaml`** (23 problems, ~10-20 minutes with long timeout)
- **Purpose**: Test local models that need more time but can be capable
- **Content**: Mix of easier problems across domains with some challenging ones
- **Timeout**: Suggested 180-300 seconds
- **Expected accuracy**: 50-70% for decent local models
- **Best for**: Evaluating local LLMs, comparing local vs cloud performance

**`quick_test.yaml`** (120 problems, ~8-12 minutes)
- **Purpose**: Fast evaluation with reliable difficulty indicators
- **Content**: Knights & Knaves, Puzzte, AIME 2024
- **Best for**: Quick but comprehensive evaluation

## Capability-Focused Testing

**`reasoning_test.yaml`** (15 problems, ~5-10 minutes)
- **Purpose**: Focus specifically on logical reasoning abilities
- **Content**: 8 ARC Challenge + 7 Knights & Knaves logic puzzles
- **Best for**: Comparing reasoning capabilities between models
- **Note**: Knights & Knaves shows "not evaluated" until we implement evaluation

**`math_gradient.yaml`** (15 problems, ~8-15 minutes)
- **Purpose**: Test mathematical reasoning across difficulty levels
- **Content**: 8 GSM8K (easy→medium) + 3 Numina Math + 4 AIME 2024
- **Shows**: Where model's math capabilities break down
- **Best for**: Understanding math skill ceiling
- **Note**: Numina Math shows "not evaluated" until we implement evaluation

**`capability_sampler.yaml`** (20 problems, ~10-15 minutes)
- **Purpose**: Broad model comparison across all capability areas
- **Content**: Samples from 6 different domains (math, science, logic, etc.)
- **Creates**: Complete capability profile for model comparison
- **Best for**: Understanding model strengths/weaknesses across domains
- **Note**: Some datasets show "not evaluated" warnings

## Specialized Testing

**`aime24_only.yaml`** (configurable, 1-30 problems)
- **Purpose**: Test upper limits of mathematical reasoning
- **Content**: AIME 2024 competition problems (all very hard)
- **Expected accuracy**: 0-30% for most models
- **Best for**: Testing mathematical olympiad-level capabilities

**`sample1.yaml`** (640+ problems, ~15-20 minutes)
- **Purpose**: Comprehensive benchmark across all datasets
- **Content**: Full sampling from all available datasets
- **Best for**: Complete model evaluation (when you have time)

## Evaluation Status

✅ **Full Evaluation**: AIME, GSM8K, ARC Challenge, Winogrande, PIQA, HellaSwag, LogiQA, MMLU, MATH
⚠️ **Not Yet Evaluated**: Puzzte, Numina Math

Files with non-evaluated datasets will show completion rates but not accuracy scores for those problems.