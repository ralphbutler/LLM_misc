#!/usr/bin/env python3
"""
Answer Extractor - LLM-Based Answer Extraction from Benchmark Results

Reads benchmark results produced by benchmark_runner.py and re-extracts answers
using a local LLM with Pydantic structured output. Compares LLM extraction vs
original regex extraction accuracy.

Usage:
    python answer_extractor.py [optional_results_file.json]

If no file specified, automatically finds the newest benchmark_results_*.json file.
"""

import sys
import os
import json
import time
import glob
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from litellm import completion
import litellm

# Enable JSON schema validation
litellm.enable_json_schema_validation = True

# Configuration - make model easily changeable
EXTRACTION_MODEL = "openai/phi-4-reasoning-plus-mlx"  # Winner from our evaluation
EXTRACTION_API_BASE = "http://localhost:1234/v1"

class ExtractedAnswer(BaseModel):
    final_answer: str          # "A", "42", "yes", etc.
    confidence: float          # 0.0-1.0 extraction confidence
    reasoning_excerpt: Optional[str] = None  # Key reasoning steps from response
    extraction_source: str     # Exact text that led to answer
    ambiguous: bool           # Multiple valid interpretations?
    answer_format: str        # "multiple_choice", "numerical", "yes_no", "text"

def find_latest_benchmark_results() -> str:
    """Find the most recent benchmark_results_*.json file (excluding already processed ones)"""
    pattern = "JSON_RESULTS/benchmark_results_*.json"
    files = glob.glob(pattern)

    # Filter out already-processed files (those ending with _llm_extracted.json)
    original_files = [f for f in files if not f.endswith('_llm_extracted.json')]

    if not original_files:
        print(f"ERROR: No original benchmark results files found matching pattern: {pattern}")
        print("Please run benchmark_runner.py first to generate results.")
        if files:
            print(f"Found {len(files)} already-processed files, but no original results to process.")
        sys.exit(1)

    # Sort by modification time, newest first
    original_files.sort(key=os.path.getmtime, reverse=True)
    latest_file = original_files[0]

    print(f"Auto-discovered latest benchmark results: {latest_file}")
    return latest_file

def create_extraction_prompt(original_response: str, dataset_type: str, question_text: str = "") -> str:
    """Create prompt for answer extraction based on dataset type"""

    # Dataset-specific guidance with emphasis on clean extraction
    format_guides = {
        "gsm8k": "Expected: PURE NUMBER ONLY like 42 or 3.5 (NO units, NO dollar signs, NO words)",
        "winogrande": "Expected: Single letter A or B",
        "piqa": "Expected: Single number 0 or 1",
        "hellaswag": "Expected: Single number 0, 1, 2, or 3",
        "logiqa": "Expected: Single letter A, B, C, or D",
        "arc_challenge": "Expected: Single letter A, B, C, or D",
        "mmlu": "Expected: Single letter A, B, C, or D",
        "competition_math": "Expected: PURE NUMBER, fraction, or expression (NO units, NO words)",
        "aime_2024": "Expected: Integer from 0 to 999 (NO units, NO words)"
    }

    format_guide = format_guides.get(dataset_type, "Expected: Extract the final answer")

    # Special instructions for mathematical datasets
    math_datasets = {"gsm8k", "competition_math", "aime_2024"}
    extra_math_instruction = ""
    if dataset_type in math_datasets:
        extra_math_instruction = """

CRITICAL FOR MATH PROBLEMS: Extract ONLY the pure number, no units or formatting:
- "The answer is 42 dollars" → extract "42"
- "Final answer: 5 hours" → extract "5"
- "Total cost is $12" → extract "12"
- "They used 273 yards" → extract "273"
- "Answer: 5/6" → extract "5/6"

DO NOT include: dollars signs ($), units (hours, yards, etc.), commas, or extra words."""

    return f"""You are an expert at extracting final answers from LLM responses to benchmark questions.

DATASET: {dataset_type}
{format_guide}{extra_math_instruction}

ORIGINAL LLM RESPONSE:
{original_response}

Your task: Extract the FINAL ANSWER and provide structured information about the extraction.

Look for clear indicators like:
- "The answer is X"
- "**Answer: X**" or "**X**"
- Numbers at the end of calculations
- Multiple choice selections like "Choose A" or "Option B"
- Final statements or conclusions

Be very precise in your extraction. Return:
- final_answer: Just the core answer (e.g., "A", "42", "5/6")
- confidence: 0.0-1.0 based on how clear the extraction was
- reasoning_excerpt: Key reasoning steps if visible (optional)
- extraction_source: The exact text that led you to this answer
- ambiguous: True if multiple valid interpretations exist
- answer_format: Category like "multiple_choice", "numerical", "fraction", etc.

Focus on getting the cleanest possible final answer with no extra formatting."""

def extract_answer_with_llm(response_text: str, dataset: str, question: str = "") -> Optional[ExtractedAnswer]:
    """Use LLM to extract structured answer from response text"""

    try:
        prompt = create_extraction_prompt(response_text, dataset, question)
        messages = [{"role": "user", "content": prompt}]

        # Try Pydantic approach first
        try:
            response = completion(
                model=EXTRACTION_MODEL,
                messages=messages,
                api_base=EXTRACTION_API_BASE,
                max_tokens=1024,
                temperature=0.0,
                response_format=ExtractedAnswer
            )

            content = response.choices[0].message.content
            json_content = json.loads(content)
            return ExtractedAnswer(**json_content)

        except Exception as e:
            # Fallback to JSON schema
            print(f"    Pydantic failed, trying JSON schema: {str(e)[:100]}...")

            response = completion(
                model=EXTRACTION_MODEL,
                messages=messages,
                api_base=EXTRACTION_API_BASE,
                max_tokens=1024,
                temperature=0.0,
                response_format={
                    "type": "json_object",
                    "response_schema": ExtractedAnswer.model_json_schema()
                }
            )

            content = response.choices[0].message.content
            json_content = json.loads(content)
            return ExtractedAnswer(**json_content)

    except Exception as e:
        print(f"    Extraction failed: {str(e)[:100]}...")
        return None

def process_benchmark_results(results_file: str) -> Dict[str, Any]:
    """Process benchmark results and re-extract answers using LLM"""

    print(f"Processing benchmark results from: {results_file}")

    # Load original results
    with open(results_file, 'r') as f:
        original_results = json.load(f)

    print(f"Original results:")
    print(f"  Total problems: {original_results['metadata']['total_problems']}")
    print(f"  Original accuracy: {original_results['metadata']['accuracy_overall']:.1f}%")

    # Count datasets
    datasets = {}
    for result in original_results['results']:
        dataset = result['dataset']
        if dataset not in datasets:
            datasets[dataset] = 0
        datasets[dataset] += 1

    print(f"  Datasets: {list(datasets.keys())} ({datasets})")

    # Create enhanced results structure
    enhanced_results = {
        "metadata": {
            "original_file": results_file,
            "extraction_model": EXTRACTION_MODEL,
            "extraction_timestamp": datetime.now().isoformat(),
            "extraction_method": "llm_pydantic"
        },
        "original_metadata": original_results["metadata"],
        "enhanced_results": {},
        "comparison": {},
        "extraction_stats": {
            "total_extractions": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "avg_confidence": 0.0,
            "low_confidence_count": 0  # confidence < 0.7
        }
    }

    total_extractions = 0
    successful_extractions = 0
    total_confidence = 0.0
    low_confidence_count = 0

    # Group results by dataset first
    datasets_data = {}
    for result in original_results["results"]:
        dataset_name = result["dataset"]
        if dataset_name not in datasets_data:
            datasets_data[dataset_name] = []
        datasets_data[dataset_name].append(result)

    # Process each dataset
    for dataset_name, dataset_problems in datasets_data.items():
        print(f"\nProcessing {dataset_name}...")

        enhanced_dataset = {
            "original_problems": dataset_problems,
            "llm_extractions": [],
            "comparison": {
                "original_correct": 0,
                "llm_correct": 0,
                "both_correct": 0,
                "both_wrong": 0,
                "original_right_llm_wrong": 0,
                "llm_right_original_wrong": 0
            }
        }

        # Process each problem in the dataset
        for i, problem in enumerate(dataset_problems):
            print(f"  Problem {i+1}/{len(dataset_problems)}")

            total_extractions += 1

            # Extract answer using LLM
            extracted = extract_answer_with_llm(
                problem["llm_result"]["content"],
                dataset_name,
                problem["problem_data"].get("question", "")
            )

            if extracted:
                successful_extractions += 1
                total_confidence += extracted.confidence

                if extracted.confidence < 0.7:
                    low_confidence_count += 1

                # Compare with correct answer
                ground_truth = str(problem["evaluation"]["ground_truth"]).lower().strip()
                llm_correct = (extracted.final_answer.lower().strip() == ground_truth)
                original_correct = problem["evaluation"]["correct"]

                # Update comparison stats
                if original_correct and llm_correct:
                    enhanced_dataset["comparison"]["both_correct"] += 1
                elif not original_correct and not llm_correct:
                    enhanced_dataset["comparison"]["both_wrong"] += 1
                elif original_correct and not llm_correct:
                    enhanced_dataset["comparison"]["original_right_llm_wrong"] += 1
                elif not original_correct and llm_correct:
                    enhanced_dataset["comparison"]["llm_right_original_wrong"] += 1

                enhanced_dataset["comparison"]["original_correct"] += int(original_correct)
                enhanced_dataset["comparison"]["llm_correct"] += int(llm_correct)

                # Store extraction result
                extraction_record = {
                    "test_id": problem["test_id"],
                    "original_extracted": problem["evaluation"]["extracted_answer"],
                    "original_correct": original_correct,
                    "llm_extracted": extracted.final_answer,
                    "llm_correct": llm_correct,
                    "ground_truth": ground_truth,
                    "confidence": extracted.confidence,
                    "ambiguous": extracted.ambiguous,
                    "answer_format": extracted.answer_format,
                    "extraction_source": extracted.extraction_source,
                    "reasoning_excerpt": extracted.reasoning_excerpt
                }

                enhanced_dataset["llm_extractions"].append(extraction_record)
            else:
                # Extraction failed
                enhanced_dataset["llm_extractions"].append({
                    "test_id": problem["test_id"],
                    "extraction_failed": True,
                    "original_extracted": problem["evaluation"]["extracted_answer"],
                    "original_correct": problem["evaluation"]["correct"]
                })

        enhanced_results["enhanced_results"][dataset_name] = enhanced_dataset

        # Print dataset summary
        total_problems = len(dataset_problems)
        original_acc = enhanced_dataset["comparison"]["original_correct"] / total_problems
        llm_acc = enhanced_dataset["comparison"]["llm_correct"] / total_problems if successful_extractions > 0 else 0

        print(f"  {dataset_name} Results:")
        print(f"    Original accuracy: {original_acc:.1%}")
        print(f"    LLM accuracy: {llm_acc:.1%}")
        print(f"    Successful extractions: {len([e for e in enhanced_dataset['llm_extractions'] if not e.get('extraction_failed', False)])}/{total_problems}")

    # Update overall extraction stats
    enhanced_results["extraction_stats"] = {
        "total_extractions": total_extractions,
        "successful_extractions": successful_extractions,
        "failed_extractions": total_extractions - successful_extractions,
        "success_rate": successful_extractions / total_extractions if total_extractions > 0 else 0,
        "avg_confidence": total_confidence / successful_extractions if successful_extractions > 0 else 0,
        "low_confidence_count": low_confidence_count,
        "low_confidence_rate": low_confidence_count / successful_extractions if successful_extractions > 0 else 0
    }

    return enhanced_results

def print_summary_report(enhanced_results: Dict[str, Any]):
    """Print a summary comparison report"""

    print("\n" + "="*60)
    print("EXTRACTION COMPARISON SUMMARY")
    print("="*60)

    # Overall stats
    original_metadata = enhanced_results["original_metadata"]
    extraction_stats = enhanced_results["extraction_stats"]

    print(f"Original benchmark accuracy: {original_metadata['accuracy_overall']:.1f}%")
    print(f"Extraction success rate: {extraction_stats['success_rate']:.1%}")
    print(f"Average extraction confidence: {extraction_stats['avg_confidence']:.2f}")
    print(f"Low confidence extractions: {extraction_stats['low_confidence_count']}/{extraction_stats['successful_extractions']} ({extraction_stats['low_confidence_rate']:.1%})")

    print(f"\nDataset Comparison:")
    print(f"{'Dataset':<20} {'Original':<10} {'LLM':<10} {'Improved':<10} {'Degraded':<10}")
    print("-" * 60)

    total_original = 0
    total_llm = 0
    total_problems = 0

    for dataset_name, results in enhanced_results["enhanced_results"].items():
        comp = results["comparison"]
        dataset_problems = len(results["original_problems"])

        original_acc = comp["original_correct"] / dataset_problems
        llm_acc = comp["llm_correct"] / dataset_problems
        improved = comp["llm_right_original_wrong"]
        degraded = comp["original_right_llm_wrong"]

        total_original += comp["original_correct"]
        total_llm += comp["llm_correct"]
        total_problems += dataset_problems

        print(f"{dataset_name:<20} {original_acc:<9.1%} {llm_acc:<9.1%} {improved:<9d} {degraded:<9d}")

    overall_original_acc = total_original / total_problems
    overall_llm_acc = total_llm / total_problems

    print("-" * 60)
    print(f"{'OVERALL':<20} {overall_original_acc:<9.1%} {overall_llm_acc:<9.1%}")

    improvement = overall_llm_acc - overall_original_acc
    print(f"\nNet improvement: {improvement:+.1%}")

def main():
    print("Answer Extractor - LLM-Based Answer Extraction")
    print("=" * 50)
    print(f"Using extraction model: {EXTRACTION_MODEL}")

    # Determine input file
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
        if not os.path.exists(results_file):
            print(f"ERROR: File not found: {results_file}")
            sys.exit(1)
    else:
        results_file = find_latest_benchmark_results()

    # Process the results
    try:
        enhanced_results = process_benchmark_results(results_file)

        # Print summary report
        print_summary_report(enhanced_results)

        # Save enhanced results
        base_name = os.path.splitext(os.path.basename(results_file))[0]
        output_file = f"JSON_RESULTS/{base_name}_llm_extracted.json"

        # Ensure JSON_RESULTS directory exists
        os.makedirs("JSON_RESULTS", exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(enhanced_results, f, indent=2)

        print(f"\nEnhanced results saved to: {output_file}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()