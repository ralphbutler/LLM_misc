# 🚀 Benchmark System Documentation

## 🏗️ Philosophy & Architecture

This benchmark system uses a **two-stage evaluation pipeline** designed for reliability, flexibility, and deep analysis:

1. **Stage 1 (`benchmark_runner.py`)**: Runs LLMs on test problems using traditional regex-based answer extraction
2. **Stage 2 (`answer_extractor.py`)**: Re-processes the same responses using advanced LLM-based semantic extraction

This architecture provides:
- **Robust comparison**: Regex vs semantic extraction on identical LLM responses
- **Future-proofing**: Semantic extraction adapts to new LLM response formats automatically
- **Zero risk**: Preserves your working benchmark system while adding advanced capabilities
- **Rich analysis**: Compare extraction methods and identify failure root causes

## 🔧 Core Programs

### 1. 📊 benchmark_runner.py - Primary Benchmark Engine

**Purpose**: Runs LLMs on test datasets and evaluates performance using regex extraction.

**Basic Usage**:
```bash
# Run a specific test configuration
python benchmark_runner.py reasoning1.yaml

# Run with custom timeout
python benchmark_runner.py reasoning1.yaml --timeout 120

# Use different model (if configured)
python benchmark_runner.py balanced1.yaml --model gpt-4
```

**Key Features**:
- Loads test problems from YAML configurations
- Supports multiple datasets (GSM8K, Winogrande, PIQA, etc.)
- Handles timeouts, retries, and error recovery
- Saves detailed results to `JSON_RESULTS/benchmark_results_*.json`
- Uses regex patterns for answer extraction

**Output Files**:
- `JSON_RESULTS/benchmark_results_[model]_[timestamp].json` - Detailed results
- `tempout_[timestamp]` - Debug/logging output

### 2. 🧠 answer_extractor.py - LLM-Based Answer Extraction

**Purpose**: Re-processes benchmark results using semantic LLM extraction to improve accuracy.

**Basic Usage**:
```bash
# Auto-find and process latest benchmark results
python answer_extractor.py

# Process specific results file
python answer_extractor.py JSON_RESULTS/benchmark_results_claude-sonnet-4_20250914_154955.json

# Change extraction model (edit EXTRACTION_MODEL variable)
# Default: openai/phi-4-reasoning-plus-mlx via LM Studio
```

**Key Features**:
- **Automatic file discovery**: Finds newest benchmark results if no file specified
- **Semantic extraction**: Uses local LLM to understand answer formats and context
- **Head-to-head comparison**: Shows regex vs LLM extraction accuracy side-by-side
- **Rich metadata**: Confidence scores, extraction source, reasoning excerpts
- **Dataset-aware prompts**: Specialized extraction logic for math, multiple choice, etc.

**Sample Output**:
```
EXTRACTION COMPARISON SUMMARY
============================================================
Original benchmark accuracy: 68.2%
Extraction success rate: 100.0%
Average extraction confidence: 0.95

Dataset Comparison:
Dataset              Original   LLM        Improved   Degraded
------------------------------------------------------------
gsm8k                80.0%     80.0%     0         0
logiqa               50.0%     100.0%    1         0
------------------------------------------------------------
OVERALL              68.2%     72.7%

Net improvement: +4.5%
```

**Output Files**:
- `[original_name]_llm_extracted.json` - Enhanced results with LLM extraction data

## 🔍 Analysis & Viewing Programs

### 3. 🕵️ view_results.py - Results Analysis Tool

**Purpose**: Analyze, filter, and understand benchmark results with automatic failure analysis.

**Basic Usage**:
```bash
# Interactive file selection and summary
python view_results.py

# View specific results file
python view_results.py JSON_RESULTS/benchmark_results_*.json

# Show only failed problems with automatic analysis
python view_results.py --failures [results_file.json]

# Show only successful problems
python view_results.py --successes [results_file.json]

# Filter by specific dataset
python view_results.py --dataset gsm8k [results_file.json]

# Compare two result files
python view_results.py --compare file1.json file2.json
```

**🔬 Failure Analysis Features**:
- **Automatic root cause detection**: Distinguishes LLM reasoning errors vs extraction failures
- **Smart pattern matching**: Finds what the LLM actually calculated/said
- **Clear diagnosis**: Shows whether regex or LLM was at fault

**Sample Analysis Output**:
```
Problem 1: [gsm8k_medium_001] ○ Incorrect
EVALUATION:
Expected: 2280
Extracted: 180.0
ANALYSIS:
✓ LLM calculated/said: 2180
✗ Expected answer: 2280
✗ But extracted: 180.0
→ ROOT CAUSE: Both LLM error AND extraction failure
```

### 4. 📚 view_dataset.py - Dataset Inspector

**Purpose**: Inspect and preview test datasets before running benchmarks.

**Basic Usage**:
```bash
# View dataset structure and samples
python view_dataset.py

# Inspect specific YAML configuration
python view_dataset.py reasoning1.yaml

# Show problems from specific dataset
python view_dataset.py --dataset gsm8k --limit 5
```

## ⚙️ Configuration Files

### 📝 YAML Test Configurations

**🧠 reasoning1.yaml** - 22 problems focused on pure reasoning:
```bash
python benchmark_runner.py reasoning1.yaml
```

**⚖️ balanced1.yaml** - 30 problems mixing knowledge + reasoning:
```bash
python benchmark_runner.py balanced1.yaml
```

**📊 sample1.yaml** - 825+ problems comprehensive benchmark:
```bash
python benchmark_runner.py sample1.yaml
```

## 🔄 Typical Workflow

### 1. 🏃‍♂️ Run Initial Benchmark
```bash
# Run your test suite
python benchmark_runner.py reasoning1.yaml

# Check results summary
python view_results.py
```

### 2. 🧠 Analyze with LLM Extraction
```bash
# Re-process with semantic extraction
python answer_extractor.py

# The output shows improvement/degradation comparison
```

### 3. 🔍 Investigate Failures
```bash
# Deep dive into what went wrong
python view_results.py --failures

# Focus on specific dataset issues
python view_results.py --failures --dataset gsm8k
```

### 4. 🆚 Compare Approaches
```bash
# Compare original vs LLM-enhanced results
python view_results.py --compare \
  JSON_RESULTS/benchmark_results_original.json \
  JSON_RESULTS/benchmark_results_original_llm_extracted.json
```

## 🤖 Model Configuration

### 🎯 Extraction Model Setup

The extraction model is configured in `answer_extractor.py`:

```python
# Configuration - easily changeable
EXTRACTION_MODEL = "openai/phi-4-reasoning-plus-mlx"  # Winner from evaluation
EXTRACTION_API_BASE = "http://localhost:1234/v1"
```

**🏆 Tested Models** (in order of extraction performance):
1. `openai/phi-4-reasoning-plus-mlx` - ⭐ **Recommended**
2. `openai/glm-4.5-air`
3. `openai/qwen3-32b-mlx`
4. `openai/llama-3.2-3b-instruct`
5. `openai/deepseek-r1-0528-qwen3-8b-mlx`

### 🖥️ LM Studio Integration

Requires **LM Studio** running locally:
1. Start LM Studio
2. Load your chosen model
3. Ensure server runs on `http://localhost:1234/v1`

## 📁 File Organization

```
├── benchmark_runner.py          # Primary benchmark engine
├── answer_extractor.py          # LLM extraction post-processor
├── view_results.py              # Results analysis with failure diagnosis
├── view_dataset.py              # Dataset inspection utility
├── reasoning1.yaml              # 22-problem reasoning test
├── balanced1.yaml               # 30-problem mixed test
├── sample1.yaml                 # 825+ comprehensive test
└── JSON_RESULTS/                # All benchmark outputs
    ├── benchmark_results_*.json      # Original results
    └── benchmark_results_*_llm_extracted.json  # Enhanced results
```

## 🚀 Advanced Features

### 🎨 Custom Extraction Prompts

Edit `create_extraction_prompt()` in `answer_extractor.py` to customize extraction logic:

```python
format_guides = {
    "gsm8k": "Expected: PURE NUMBER ONLY like 42 or 3.5 (NO units, NO dollar signs)",
    "your_dataset": "Expected: Your custom format instructions"
}
```

### 📦 Batch Processing

```bash
# Process multiple configurations
for config in reasoning1.yaml balanced1.yaml; do
    python benchmark_runner.py $config
    python answer_extractor.py
done
```

### 🤖 Automated Analysis Pipeline

```bash
#!/bin/bash
# Run test and analyze in one script
python benchmark_runner.py reasoning1.yaml
python answer_extractor.py
python view_results.py --failures
```

## 🔧 Troubleshooting

### ⚠️ Common Issues

**"No benchmark results files found"**:
- Ensure `benchmark_runner.py` completed successfully
- Check `JSON_RESULTS/` directory exists
- Verify file naming pattern: `benchmark_results_*.json`

**LLM extraction failures**:
- Confirm LM Studio is running with model loaded
- Check `http://localhost:1234/v1/health` responds
- Verify model name has `openai/` prefix

**Analysis shows "Unable to determine cause"**:
- LLM response may be too verbose/unclear
- Consider refining extraction patterns in `analyze_failure()`

### 🐛 Debug Mode

```bash
# Enable detailed logging
LITELLM_LOG=DEBUG python answer_extractor.py
```

## ⚡ Performance Notes

- **Regex extraction**: ~0.1s per problem
- **LLM extraction**: ~1-5s per problem (depends on model)
- **Analysis**: Instant for cached results
- **Cost**: LLM extraction uses local models (free after setup)

The two-stage approach balances speed (regex for quick feedback) with accuracy (LLM for precision when needed).